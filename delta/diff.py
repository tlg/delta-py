"""
Deterministic typed snapshot diff between two documents.

``snapshot_diff(base, target)`` returns a normalized change satisfying
``base.compose(base.diff(target)) == target``, built from retains,
inserts, deletes and typed embed-retain patches only — never cut/paste
(explicit move intent is composed separately by the caller).

Documents are normalized into *atoms* — one UTF-16 code unit per text
atom, one unit per embed — and aligned in two deterministic stages:
first the maximum number of *exact* matches (value and attributes),
then, inside each unmatched gap, the maximum number of *compatible*
matches (same text unit ignoring attributes; same embed type ignoring
payload).  Ties break by a canonical bidirectional Myers ordering, so
output never depends on elapsed time, dictionary ordering, or how the
input text was chunked into ops.  Exact-first alignment keeps an
unchanged embed retained verbatim rather than patching its repeated
siblings pairwise.

Exact common document edges are removed structurally before atomization,
so a small edit in a large snapshot only atomizes the changed middle.

Globally smallest wire output is *not* a goal: a slightly larger but
stable delta is preferred over heuristic cleanup.
"""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import chain
from typing import Any, Protocol, Self

from . import op
from .embed import handlers, walk_move_ops
from .op import Attributes, Op, Ops, RetainValue


type DiffContext = Mapping[str, Any]


class DeltaLike(Protocol):
    """The document/edit surface used by the snapshot zipper."""

    ops: Ops

    def __init__(self, ops: Ops | None = None) -> None: ...

    def __len__(self) -> int: ...

    def push(self, operation: Op) -> Self: ...

    def delete(self, length: int) -> Self: ...

    def retain(self, length: RetainValue, **attrs: Any) -> Self: ...

    def extend(self, ops: Ops | Self) -> Self: ...

    def chop(self) -> Self: ...


def _freeze(value: Any) -> str:
    """A stable equality key: canonical JSON, independent of dict
    insertion order; lone surrogates survive via ASCII escapes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class _Atom:
    """One document unit: a UTF-16 code unit of text, or an embed."""

    value: Any  # the code unit, or the embed's {type: payload} value
    attributes: Attributes | None
    frozen_value: str
    frozen_attributes: str
    compatible_value: str | None  # text unit, or embed type


def _insert_op(operation: Op | None) -> op.InsertOp:
    if operation is None or not op.is_insert(operation):
        raise ValueError("diff() requires documents with only insert ops")
    return operation


def _atomize(delta: DeltaLike) -> list[_Atom]:
    atoms: list[_Atom] = []
    for operation in delta.ops:
        operation = _insert_op(operation)
        insert = operation["insert"]
        attributes = operation.get("attributes")
        frozen_attrs = _freeze(attributes)
        if isinstance(insert, str):
            for index in range(op.str_length(insert)):
                unit = op.str_slice(insert, index, index + 1)
                atoms.append(_Atom(unit, attributes, unit, frozen_attrs, unit))
        else:
            frozen = _freeze(insert)
            compatible = next(iter(insert), None) if isinstance(insert, dict) else type(insert).__name__
            atoms.append(_Atom(insert, attributes, frozen, frozen_attrs, compatible))
    return atoms


def _same(left: _Atom, right: _Atom, compatible: bool) -> bool:
    if isinstance(left.value, str) != isinstance(right.value, str):
        return False
    if compatible:
        return left.compatible_value == right.compatible_value
    return left.frozen_value == right.frozen_value and left.frozen_attributes == right.frozen_attributes


def _exact_key(atom: _Atom) -> tuple[bool, str, str]:
    return isinstance(atom.value, str), atom.frozen_value, atom.frozen_attributes


def _text_edge(left: str, right: str, suffix: bool = False) -> int:
    """Common UTF-16 prefix/suffix length, using C-level slice
    comparisons rather than visiting equal units in Python."""
    if left.isascii() and right.isascii():
        a, b, width = left, right, 1
    else:
        a = left.encode("utf-16-le", "surrogatepass")
        b = right.encode("utf-16-le", "surrogatepass")
        width = 2
    high = min(len(a), len(b)) // width
    if not high or (a[-width:] != b[-width:] if suffix else a[:width] != b[:width]):
        return 0
    low = 1
    while low < high:
        middle = (low + high + 1) // 2
        boundary = middle * width
        same = a[-boundary:] == b[-boundary:] if suffix else a[:boundary] == b[:boundary]
        low, high = (middle, high) if same else (low, middle - 1)
    return low


def _common_piece(left: Any, right: Any, left_attrs: Any, right_attrs: Any, suffix: bool = False) -> int:
    if _freeze(left_attrs) != _freeze(right_attrs):
        return 0
    if isinstance(left, str) and isinstance(right, str):
        return _text_edge(left, right, suffix)
    return int(not isinstance(left, str) and not isinstance(right, str) and _freeze(left) == _freeze(right))


def _common_prefix(left_ops: list[Op], right_ops: list[Op]) -> int:
    left, right = op.iterator(left_ops), op.iterator(right_ops)
    length = 0
    while left.has_next() and right.has_next():
        if left.peek_length() == 0:
            left.next(0)
            continue
        if right.peek_length() == 0:
            right.next(0)
            continue
        left_raw, right_raw = _insert_op(left.peek()), _insert_op(right.peek())
        size = int(min(left.peek_length(), right.peek_length()))
        left_piece, right_piece = _insert_op(left.next(size))["insert"], _insert_op(right.next(size))["insert"]
        common = _common_piece(left_piece, right_piece, left_raw.get("attributes"), right_raw.get("attributes"))
        length += common
        if common < size:
            return length
    return length


def _common_suffix(left_ops: list[Op], right_ops: list[Op]) -> int:
    left_index, right_index = len(left_ops) - 1, len(right_ops) - 1
    left_offset = right_offset = length = 0
    while left_index >= 0 and right_index >= 0:
        left_raw, right_raw = _insert_op(left_ops[left_index]), _insert_op(right_ops[right_index])
        left_size, right_size = op.length(left_raw), op.length(right_raw)
        if left_offset == left_size:
            left_index, left_offset = left_index - 1, 0
            continue
        if right_offset == right_size:
            right_index, right_offset = right_index - 1, 0
            continue
        left_insert, right_insert = left_raw["insert"], right_raw["insert"]
        size = min(left_size - left_offset, right_size - right_offset)
        left_piece = (
            op.str_slice(left_insert, left_size - left_offset - size, left_size - left_offset)
            if isinstance(left_insert, str)
            else left_insert
        )
        right_piece = (
            op.str_slice(right_insert, right_size - right_offset - size, right_size - right_offset)
            if isinstance(right_insert, str)
            else right_insert
        )
        common = _common_piece(left_piece, right_piece, left_raw.get("attributes"), right_raw.get("attributes"), True)
        length += common
        if common < size:
            return length
        left_offset += size
        right_offset += size
    return length


def _document_slice[DeltaT: DeltaLike](delta: DeltaT, start: int, stop: int | None = None) -> DeltaT:
    """Document-only slice preserving even empty attribute mappings;
    the general op iterator intentionally canonicalizes those away."""
    result: list[Op] = []
    position = 0
    for operation in delta.ops:
        operation = _insert_op(operation)
        size = op.length(operation)
        if stop is not None and position >= stop:
            break
        if size and position + size > start:
            value = operation["insert"]
            if isinstance(value, str):
                low = max(start - position, 0)
                high = size if stop is None else min(stop - position, size)
                piece: Op = {"insert": op.str_slice(value, low, high)}
                if "attributes" in operation:
                    piece["attributes"] = operation["attributes"]
                result.append(piece)
            else:
                result.append(operation)
        position += size
    return delta.__class__(result)


def _matched(
    a: list[_Atom],
    b: list[_Atom],
    compatible: bool = False,
    a_start: int = 0,
    a_end: int | None = None,
    b_start: int = 0,
    b_end: int | None = None,
) -> list[tuple[int, int]]:
    """A deterministic LCS as matched absolute indexes.

    Bidirectional Myers finds a middle overlap, then an explicit stack
    bisects each half.  Only the two current frontiers are retained:
    O(N + M) memory rather than one O(D)-wide snapshot for every D.
    Leading and one-unit matches are leftmost, then trailing snakes are
    preferred; remaining ties take the first diagonal overlap, with an
    x-advancing edge winning equal frontier lengths.
    """
    a_end, b_end = len(a) if a_end is None else a_end, len(b) if b_end is None else b_end

    def equal(i: int, j: int) -> bool:
        return _same(a[i], b[j], compatible)

    def bisect(alo: int, ahi: int, blo: int, bhi: int) -> tuple[int, int]:
        n, m = ahi - alo, bhi - blo
        max_distance = (n + m + 1) // 2
        offset = max_distance + 1
        size = 2 * max_distance + 3
        forward = [-1] * size
        backward = [-1] * size
        forward[offset + 1] = backward[offset + 1] = 0
        delta = n - m
        front_overlap = bool(delta & 1)
        f_start = f_end = back_start = back_end = 0

        for distance in range(max_distance + 1):
            # Higher diagonals first preserves Quill's deletion-first
            # ambiguity rule (the earliest target occurrence survives).
            for diagonal in reversed(range(-distance + f_start, distance + 1 - f_end, 2)):
                index = offset + diagonal
                down = diagonal == -distance or (diagonal != distance and forward[index - 1] < forward[index + 1])
                x = forward[index + 1] if down else forward[index - 1] + 1
                y = x - diagonal
                while 0 <= x < n and 0 <= y < m and equal(alo + x, blo + y):
                    x += 1
                    y += 1
                forward[index] = x
                if x > n:
                    f_end += 2
                elif y > m:
                    f_start += 2
                elif front_overlap:
                    reverse_index = offset + delta - diagonal
                    if 0 <= reverse_index < size and backward[reverse_index] != -1:
                        if x >= n - backward[reverse_index]:
                            return alo + x, blo + y

            for diagonal in range(-distance + back_start, distance + 1 - back_end, 2):
                index = offset + diagonal
                down = diagonal == -distance or (diagonal != distance and backward[index - 1] < backward[index + 1])
                x = backward[index + 1] if down else backward[index - 1] + 1
                y = x - diagonal
                while 0 <= x < n and 0 <= y < m and equal(ahi - x - 1, bhi - y - 1):
                    x += 1
                    y += 1
                backward[index] = x
                if x > n:
                    back_end += 2
                elif y > m:
                    back_start += 2
                elif not front_overlap:
                    forward_diagonal = delta - diagonal
                    forward_index = offset + forward_diagonal
                    if 0 <= forward_index < size and forward[forward_index] != -1:
                        forward_x = forward[forward_index]
                        if forward_x >= n - x:
                            return alo + forward_x, blo + forward_x - forward_diagonal
        raise RuntimeError("Myers bisect found no overlap")  # pragma: no cover - algorithm invariant

    matches: list[tuple[int, int]] = []
    # The flag marks an already matched suffix, emitted after its left
    # sibling.  A stack avoids Python recursion on highly fragmented text.
    stack = [(a_start, a_end, b_start, b_end, False)]
    while stack:
        alo, ahi, blo, bhi, emit = stack.pop()
        if emit:
            matches.extend(zip(range(alo, ahi), range(blo, bhi)))
            continue
        while alo < ahi and blo < bhi and equal(alo, blo):
            matches.append((alo, blo))
            alo += 1
            blo += 1
        if alo == ahi or blo == bhi:
            continue
        if ahi - alo == 1:
            if (j := next((j for j in range(blo, bhi) if equal(alo, j)), None)) is not None:
                matches.append((alo, j))
            continue
        if bhi - blo == 1:
            if (i := next((i for i in range(alo, ahi) if equal(i, blo)), None)) is not None:
                matches.append((i, blo))
            continue

        suffix = 0
        while alo < ahi and blo < bhi and equal(ahi - 1, bhi - 1):
            ahi -= 1
            bhi -= 1
            suffix += 1
        if suffix:
            stack.append((ahi, ahi + suffix, bhi, bhi + suffix, True))
            if alo < ahi and blo < bhi:
                stack.append((alo, ahi, blo, bhi, False))
            continue

        x, y = bisect(alo, ahi, blo, bhi)
        if (x, y) in ((alo, blo), (ahi, bhi)):
            raise RuntimeError("Myers bisect did not advance")  # pragma: no cover - no common edge snake
        stack.append((x, ahi, y, bhi, False))
        stack.append((alo, x, blo, y, False))
    return matches


def _aligned(base: list[_Atom], target: list[_Atom]) -> Iterator[tuple[int, int, bool]]:
    """(base index, target index, is_exact) pairs in document order:
    maximum exact matches first, then compatible matches per gap."""
    previous = (-1, -1)
    for low, high in chain(_matched(base, target), ((len(base), len(target)),)):
        for i, j in _matched(base, target, True, previous[0] + 1, low, previous[1] + 1, high):
            yield i, j, False
        if low < len(base):
            yield low, high, True
        previous = (low, high)


def _embed_pair(delta: DeltaLike, ours: _Atom, theirs: _Atom, context: DiffContext) -> None:
    """Emit one aligned same-type embed pair: a retain, a handler patch,
    or an explicit replacement."""
    attributes = op.diff(ours.attributes, theirs.attributes)
    if ours.frozen_value == theirs.frozen_value:
        delta.retain(1, **(attributes or {}))
        return
    embed_type = next(iter(ours.value))
    handler = handlers.get(embed_type)
    diff = getattr(handler, "diff", None)
    patch = diff(ours.value[embed_type], theirs.value[embed_type], context) if diff is not None else NotImplemented
    if patch is NotImplemented:  # the handler asks for replacement
        _replace(delta, theirs)
        return
    if patch is None:
        raise ValueError(f"{embed_type!r} handler returned no diff for unequal snapshots (None means equality only)")
    if next(walk_move_ops({embed_type: patch}), None) is not None:
        raise ValueError(f"{embed_type!r} handler diff produced a move: snapshot diffs must not contain cut or paste")
    retain: Op = {"retain": {embed_type: patch}}
    if attributes:
        retain["attributes"] = attributes
    delta.push(retain)


def _replace(delta: DeltaLike, theirs: _Atom) -> None:
    insert: Op = {"insert": theirs.value}
    if theirs.attributes:
        insert["attributes"] = theirs.attributes
    delta.push(insert)
    delta.delete(1)


def _plain_retain(operation: Op | None) -> int | None:
    if operation is None or not op.is_retain_op(operation) or operation.get("attributes"):
        return None
    retained = operation["retain"]
    return retained if isinstance(retained, int) else None


def _anchored[DeltaT: DeltaLike](delta: DeltaT, base_atoms: list[_Atom], cursor: int) -> DeltaT:
    """Slide the first pure insert or delete toward ``cursor`` (a UTF-16
    position in the base document) when the swept span is a uniform run
    of identical units — the edit lands where the caret was instead of
    at the canonical end of the run.  An unreachable cursor (blocked by
    a differing unit or attributes) leaves the canonical placement."""
    position = 0
    for index, operation in enumerate(delta.ops):
        if op.is_retain_op(operation):
            retained = operation["retain"]
            # a typed embed patch retains exactly one unit
            position += retained if isinstance(retained, int) else 1
            continue
        if op.is_insert(operation):
            insert = operation["insert"]
            if not isinstance(insert, str):
                return delta
            length = 0
        elif op.is_delete(operation):
            insert = None
            length = operation["delete"]
        else:
            return delta
        # only a lone edit between retains is slidable
        neighbors = delta.ops[index + 1 : index + 2]
        if neighbors and op.type(neighbors[0]) in ("insert", "delete"):
            return delta
        low, high = min(cursor, position), max(cursor, position) + length
        if low < 0 or high > len(base_atoms):
            return delta
        span = [_exact_key(base_atoms[i]) for i in range(min(cursor, position), max(cursor, position))]
        if insert is not None:
            frozen_attrs = _freeze(operation.get("attributes"))
            units = [(True, op.str_slice(insert, i, i + 1), frozen_attrs) for i in range(op.str_length(insert))]
            # sliding is sound iff the affected region reads the same
            # both ways: the swept run commutes with the insertion
            # (uniform and periodic runs alike, emoji pairs included)
            if span + units != units + span:
                return delta
        else:
            # deleting [position, +L) vs [cursor, +L): sound iff the
            # swept run repeats itself L units later
            start = min(cursor, position)
            shifted = [_exact_key(base_atoms[i]) for i in range(start + length, max(cursor, position) + length)]
            if span != shifted:
                return delta
        move = cursor - position  # negative slides the edit left
        # the edit slides by trading units between its two neighboring
        # plain retains; an attributed retain pins a format boundary
        # and blocks the slide
        previous = delta.ops[index - 1] if index else None
        following = delta.ops[index + 1] if index + 1 < len(delta.ops) else None
        for neighbor, needed in ((previous, -move), (following, move)):
            if needed <= 0:
                continue  # this side only grows (or the tail is implicit)
            if neighbor is None and neighbor is following:
                continue  # growing into the implicit trailing retain
            if (retained := _plain_retain(neighbor)) is None or retained < needed:
                return delta
        # slide: the retain before the edit gains `move` units, the one
        # after loses them (absent retains are implicit tail)
        rebuilt = delta.__class__()
        for i, other in enumerate(delta.ops):
            retained = _plain_retain(other)
            if i == index - 1 and retained is not None:
                rebuilt.retain(retained + move)
            elif i == index:
                if _plain_retain(previous) is None:
                    rebuilt.retain(move)  # a new retain before the edit
                rebuilt.push(other)
            elif i == index + 1 and retained is not None:
                rebuilt.retain(retained - move)
            elif i == index + 1 and move < 0:
                rebuilt.retain(-move)  # vacated run before a pinned op
                rebuilt.push(other)
            else:
                rebuilt.push(other)
        if move < 0 and following is None:
            rebuilt.retain(-move)  # implicit tail; chop drops it
        return rebuilt.chop()
    return delta


def _atom_diff[DeltaT: DeltaLike](base: DeltaT, target: DeltaT, context: DiffContext) -> DeltaT:
    ours = _atomize(base)
    theirs = _atomize(target)
    delta = base.__class__()
    cursor = (0, 0)
    for i, j, is_exact in chain(_aligned(ours, theirs), ((len(ours), len(theirs), True),)):
        for index in range(cursor[0], i):
            atom = ours[index]
            delta.delete(1)
        for index in range(cursor[1], j):
            atom = theirs[index]
            insert: Op = {"insert": atom.value}
            if atom.attributes:
                insert["attributes"] = atom.attributes
            delta.push(insert)
        if i < len(ours):
            a, b = ours[i], theirs[j]
            if is_exact:
                delta.retain(1)
            elif isinstance(a.value, str):
                delta.retain(1, **(op.diff(a.attributes, b.attributes) or {}))
            else:
                _embed_pair(delta, a, b, context)
        cursor = (i + 1, j + 1)
    return delta.chop()


def snapshot_diff[DeltaT: DeltaLike](base: DeltaT, target: DeltaT, context: DiffContext) -> DeltaT:
    if base.ops == target.ops:
        return base.__class__()
    if any(operation.get("insert") is None for operation in chain(base.ops, target.ops)):
        raise ValueError("diff() requires documents with only insert ops")
    prefix = _common_prefix(base.ops, target.ops)
    ours, theirs = (_document_slice(base, prefix), _document_slice(target, prefix)) if prefix else (base, target)
    # The matcher resolves a one-unit range leftmost before considering
    # a common suffix; keep that canonical ambiguity rule unchanged.
    suffix = _common_suffix(ours.ops, theirs.ops) if min(len(ours), len(theirs)) > 1 else 0
    if suffix:
        ours = _document_slice(ours, 0, len(ours) - suffix)
        theirs = _document_slice(theirs, 0, len(theirs) - suffix)
    return base.__class__().retain(prefix).extend(_atom_diff(ours, theirs, context)).chop()


def snapshot_diff_at[DeltaT: DeltaLike](base: DeltaT, target: DeltaT, cursor: int, context: DiffContext) -> DeltaT:
    """``snapshot_diff`` with a caret hint: the base-document UTF-16
    position where the edit happened.  Within a uniform run — where
    several placements produce the same document — the edit is anchored
    at the caret instead of the canonical end of the run, so concurrent
    transforms reorder around the position the editor actually used.
    Same inputs always give the same output; an unreachable hint leaves
    the canonical placement."""
    delta = snapshot_diff(base, target, context)
    if not delta.ops:
        return delta
    return _anchored(delta, _atomize(base), cursor)
