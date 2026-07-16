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
payload).  Ties break by the canonical Myers ordering, so output never
depends on elapsed time, dictionary ordering, or how the input text was
chunked into ops.  Exact-first alignment keeps an unchanged embed
retained verbatim rather than patching its repeated siblings pairwise.

Globally smallest wire output is *not* a goal: a slightly larger but
stable delta is preferred over heuristic cleanup.
"""

import json
from dataclasses import dataclass
from typing import Any

from . import op
from .moves import Attributes, MoveDelta, Op, _walk_move_ops, handlers


def _freeze(value: Any) -> str:
    """A stable equality key: canonical JSON, independent of dict
    insertion order; lone surrogates survive via ASCII escapes."""
    return json.dumps(value, sort_keys=True)


@dataclass(frozen=True, slots=True)
class _Atom:
    """One document unit: a UTF-16 code unit of text, or an embed."""

    kind: str  # 'text' | 'embed'
    value: Any  # the code unit, or the embed's {type: payload} value
    attributes: Attributes | None
    frozen_value: str
    exact: str  # value + attributes
    compat: str  # value ignoring attributes / embed type only


def _atomize(delta: MoveDelta) -> list[_Atom]:
    atoms: list[_Atom] = []
    for operation in delta.ops:
        insert = operation.get('insert')
        if insert is None and insert != '':
            raise ValueError(
                'diff() requires documents with only insert ops')
        attributes = operation.get('attributes')
        frozen_attrs = _freeze(attributes)
        if isinstance(insert, str):
            for index in range(op.str_length(insert)):
                unit = op.str_slice(insert, index, index + 1)
                frozen = _freeze(unit)
                atoms.append(_Atom(
                    'text', unit, attributes, frozen,
                    exact=f'text:{frozen}:{frozen_attrs}',
                    compat=f'text:{frozen}'))
        else:
            embed_type = next(iter(insert), None)
            frozen = _freeze(insert)
            atoms.append(_Atom(
                'embed', insert, attributes, frozen,
                exact=f'embed:{frozen}:{frozen_attrs}',
                compat=f'embed:{embed_type}'))
    return atoms


def _matched(a: list[str], b: list[str]) -> list[tuple[int, int]]:
    """Matched index pairs of a longest common subsequence, by the
    canonical greedy forward Myers walk — deterministic, no timeout."""
    n, m = len(a), len(b)
    if not n or not m:
        return []
    v: dict[int, int] = {1: 0}
    trace: list[dict[int, int]] = []
    final_d = None
    for d in range(n + m + 1):
        trace.append(dict(v))
        for k in range(-d, d + 1, 2):
            if k == -d or (k != d and v.get(k - 1, -1) < v.get(k + 1, -1)):
                x = v[k + 1]
            else:
                x = v[k - 1] + 1
            y = x - k
            while x < n and y < m and a[x] == b[y]:
                x += 1
                y += 1
            v[k] = x
            if x >= n and y >= m:
                final_d = d
                break
        if final_d is not None:
            break
    assert final_d is not None
    matches: list[tuple[int, int]] = []
    x, y = n, m
    for d in range(final_d, 0, -1):
        state = trace[d]
        k = x - y
        if k == -d or (k != d and state.get(k - 1, -1) < state.get(k + 1, -1)):
            prev_k = k + 1  # the edit was an insertion (down)
        else:
            prev_k = k - 1  # the edit was a deletion (right)
        prev_x = state[prev_k]
        prev_y = prev_x - prev_k
        edit_x = prev_x if prev_k == k + 1 else prev_x + 1
        while x > edit_x:  # the snake: matched diagonal after the edit
            x -= 1
            y -= 1
            matches.append((x, y))
        x, y = prev_x, prev_y
    while x > 0 and y > 0:  # leading snake before any edit
        x -= 1
        y -= 1
        matches.append((x, y))
    matches.reverse()
    return matches


def _aligned(base: list[_Atom],
             target: list[_Atom]) -> list[tuple[int, int, bool]]:
    """(base index, target index, is_exact) pairs in document order:
    maximum exact matches first, then compatible matches per gap."""
    exact = _matched([atom.exact for atom in base],
                     [atom.exact for atom in target])
    pairs: list[tuple[int, int, bool]] = []
    previous = (-1, -1)
    for low, high in [*exact, (len(base), len(target))]:
        gap_a = base[previous[0] + 1:low]
        gap_b = target[previous[1] + 1:high]
        for i, j in _matched([atom.compat for atom in gap_a],
                             [atom.compat for atom in gap_b]):
            pairs.append((previous[0] + 1 + i, previous[1] + 1 + j, False))
        if low < len(base):
            pairs.append((low, high, True))
        previous = (low, high)
    return pairs


def _embed_pair(delta: MoveDelta, ours: _Atom, theirs: _Atom) -> None:
    """Emit one aligned same-type embed pair: a retain, a handler patch,
    or an explicit replacement."""
    attributes = op.diff(ours.attributes, theirs.attributes)
    if ours.frozen_value == theirs.frozen_value:
        delta.retain(1, **(attributes or {}))
        return
    embed_type = next(iter(ours.value))
    handler = handlers.get(embed_type)
    diff = getattr(handler, 'diff', None)
    patch = diff(ours.value[embed_type], theirs.value[embed_type]) \
        if diff is not None else NotImplemented
    if patch is NotImplemented:  # the handler asks for replacement
        _replace(delta, theirs)
        return
    if patch is None:
        raise ValueError(
            f'{embed_type!r} handler returned no diff for unequal '
            'snapshots (None means equality only)')
    if next(_walk_move_ops([patch]), None) is not None:
        raise ValueError(
            f'{embed_type!r} handler diff produced a move: snapshot '
            'diffs must not contain cut or paste')
    retain: Op = {'retain': {embed_type: patch}}
    if attributes:
        retain['attributes'] = attributes
    delta.push(retain)


def _replace(delta: MoveDelta, theirs: _Atom) -> None:
    insert: Op = {'insert': theirs.value}
    if theirs.attributes:
        insert['attributes'] = theirs.attributes
    delta.push(insert)
    delta.delete(1)


def _anchored(delta: MoveDelta, base_atoms: list[_Atom],
              cursor: int) -> MoveDelta:
    """Slide the first pure insert or delete toward ``cursor`` (a UTF-16
    position in the base document) when the swept span is a uniform run
    of identical units — the edit lands where the caret was instead of
    at the canonical end of the run.  An unreachable cursor (blocked by
    a differing unit or attributes) leaves the canonical placement."""
    position = 0
    for index, operation in enumerate(delta.ops):
        kind = op.type(operation)
        if kind == 'retain':
            retained = operation['retain']
            # a typed embed patch retains exactly one unit
            position += retained if isinstance(retained, int) else 1
            continue
        if kind not in ('insert', 'delete'):
            return delta
        # only a lone edit between retains is slidable
        neighbors = delta.ops[index + 1:index + 2]
        if neighbors and op.type(neighbors[0]) in ('insert', 'delete'):
            return delta
        length = operation.get('delete', 0)
        low, high = min(cursor, position), max(cursor, position) + length
        if low < 0 or high > len(base_atoms):
            return delta
        span = [atom.exact for atom in
                base_atoms[min(cursor, position):max(cursor, position)]]
        if kind == 'insert':
            insert = operation.get('insert')
            if not isinstance(insert, str):
                return delta
            frozen_attrs = _freeze(operation.get('attributes'))
            units = [
                f'text:{_freeze(op.str_slice(insert, i, i + 1))}:{frozen_attrs}'
                for i in range(op.str_length(insert))]
            # sliding is sound iff the affected region reads the same
            # both ways: the swept run commutes with the insertion
            # (uniform and periodic runs alike, emoji pairs included)
            if span + units != units + span:
                return delta
        else:
            # deleting [position, +L) vs [cursor, +L): sound iff the
            # swept run repeats itself L units later
            start = min(cursor, position)
            shifted = [atom.exact for atom in
                       base_atoms[start + length:
                                  max(cursor, position) + length]]
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
            if (neighbor is None
                    or not isinstance(neighbor.get('retain'), int)
                    or neighbor.get('attributes')
                    or neighbor['retain'] < needed):
                return delta
        # slide: the retain before the edit gains `move` units, the one
        # after loses them (absent retains are implicit tail)
        rebuilt = delta.__class__()
        for i, other in enumerate(delta.ops):
            if i == index - 1 and isinstance(other.get('retain'), int) \
                    and not other.get('attributes'):
                rebuilt.retain(other['retain'] + move)
            elif i == index:
                if previous is None or not isinstance(
                        previous.get('retain'), int) \
                        or previous.get('attributes'):
                    rebuilt.retain(move)  # a new retain before the edit
                rebuilt.push(other)
            elif i == index + 1 and isinstance(other.get('retain'), int) \
                    and not other.get('attributes'):
                rebuilt.retain(other['retain'] - move)
            elif i == index + 1 and move < 0:
                rebuilt.retain(-move)  # vacated run before a pinned op
                rebuilt.push(other)
            else:
                rebuilt.push(other)
        if move < 0 and following is None:
            rebuilt.retain(-move)  # implicit tail; chop drops it
        return rebuilt.chop()
    return delta


def snapshot_diff(base: MoveDelta, target: MoveDelta) -> MoveDelta:
    if base.ops == target.ops:
        return base.__class__()
    ours = _atomize(base)
    theirs = _atomize(target)
    delta = base.__class__()
    cursor = (0, 0)
    for i, j, is_exact in [*_aligned(ours, theirs),
                           (len(ours), len(theirs), True)]:
        for atom in ours[cursor[0]:i]:
            delta.delete(1)
        for atom in theirs[cursor[1]:j]:
            insert: Op = {'insert': atom.value}
            if atom.attributes:
                insert['attributes'] = atom.attributes
            delta.push(insert)
        if i < len(ours):
            a, b = ours[i], theirs[j]
            if is_exact:
                delta.retain(1)
            elif a.kind == 'text':
                delta.retain(1, **(op.diff(a.attributes, b.attributes) or {}))
            else:
                _embed_pair(delta, a, b)
        cursor = (i + 1, j + 1)
    return delta.chop()


def snapshot_diff_at(base: MoveDelta, target: MoveDelta,
                     cursor: int) -> MoveDelta:
    """``snapshot_diff`` with a caret hint: the base-document UTF-16
    position where the edit happened.  Within a uniform run — where
    several placements produce the same document — the edit is anchored
    at the caret instead of the canonical end of the run, so concurrent
    transforms reorder around the position the editor actually used.
    Same inputs always give the same output; an unreachable hint leaves
    the canonical placement."""
    delta = snapshot_diff(base, target)
    if not delta.ops:
        return delta
    return _anchored(delta, _atomize(base), cursor)
