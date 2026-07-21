"""
Standalone executable reference for semantic cut/paste move OT.

This file uses :mod:`delta.op` for operation typing, UTF-16 splitting,
attribute algebra, and the Quill-style iterator.  It intentionally does not
import or depend on ``delta.theory``.

Two additional operation types express moving content without buffering it:

    {'cut':   {'ref': 'r', 'length': 10}}
    {'paste': {'ref': 'r', 'start': 0, 'length': 10}, 'attributes': {...}}

``cut`` consumes ``length`` characters of its input — like a delete — and
remembers the removed span under ``ref``.  ``paste`` produces ``length``
characters starting at ``start`` *within that remembered span*, optionally
applying an attribute patch (retain semantics: ``None`` removes a key) on
top of whatever attributes the content carries.  A paste of a single embed
may also carry ``change``, a retain-style embed patch applied to the
underlying payload at apply time — the embed analogue of ``attributes``.

Because a paste addresses its source by position instead of by value,
deltas stay base-free and closed under composition:

* an insert into a pasted span splits the window:
  ``paste(r, 0, 10)`` -> ``paste(r, 0, 2) . insert('x') . paste(r, 2, 8)``
* a delete inside a pasted span shrinks or splits the window,
* a format over a pasted span becomes the paste's attribute patch,
* cutting a region an earlier delta edited sends those edits along:
  inserts reappear literally between paste windows, formats become paste
  attribute patches, and deleted characters are cut but never pasted.

Invariants (see ``check``): a ref has exactly one cut; every paste window
fits inside its cut; windows of one ref are pairwise disjoint (move, not
copy); and every public cut has a paste consumer.  A zipper may transiently
produce an orphan cut, but normalization degrades that internal fragment to
a plain delete before returning it.

All offsets and lengths count UTF-16 code units, matching the upstream
JavaScript library: astral characters (most emoji) count 2, and a
boundary may fall inside a surrogate pair — the lone halves re-pair
into the real character when pushed back together.

Inverting keeps the semantic: each paste window inverts to a cut of the
pasted span, and the source position pastes everything back.

Transform convention: concurrent edits *to moved content* (formats and
deletes) follow the content to its paste site; concurrent inserts *at a
position* inside the moved region stay at the source position.

Cuts split when they must stay contiguous: composing a cut over a region
containing an earlier cut, or transforming one against a concurrent
insert into its source, yields parts ``r``, ``r:1``, ... with the paste
windows renumbered across them.

Concurrent moves transform by rebasing: the priority side keeps contested
content and its cut re-targets the other side's paste windows (the content
is re-cut from wherever the loser put it); the losing side keeps only the
parts of its move that were never contested, while its deletions (window
gaps), formats (window attributes) and embed patches (window changes)
still apply to the content.

Moves also cross sequence levels, recursively: a cut inside an embed's
child sequence (a table cell — or a cell inside a cell, to any depth)
may pair with a paste at root, in a sibling, or levels deeper, sharing
one ref namespace per delta.  Handlers that carry child sequences opt in
by running them through ``MoveDelta`` and passing their explicit operation
context to structural child compose/transform/invert/diff calls at any
depth, so captures, window renumbering and routed edits flow between levels.
Unrelated algebra calls omit that context and start their own transaction.
Routed edits whose destination window lives inside embeds are composed back
in as minimal embed changes nested along the destination's hop chain,
following the embeds if they were themselves moved.

A paste may also ride a *newly inserted* embed's child sequence, at any
depth — moving existing content into a table the same delta creates.
The insert carries the window: compose expands it in place (splitting
it around earlier edits like any other window), concurrent formats and
deletes route into the inserted payload, and the inverse restores the
moved span at its source from ``base`` — the pasted copy simply dies
with the insert's inverse delete.

Deleting an embed that still *sources* a live move is handled with a
trash bin, in the spirit of tree-OT deleted-subtree buffers (Davis, Sun
& Lu 2002) — except no buffer is needed, because a cut already *is* one:
captured-but-never-pasted content stays positionally addressable.  The
deletion becomes a trash cut of the embed, and the orphaned pastes read
through the capture by coordinate:

    {'paste': {'ref': 'trash', 'unit': 0, 'path': ['ops'],
               'start': 2, 'length': 5}}

i.e. characters [2, 7) of the child sequence at ``path`` inside the
embed captured at offset ``unit`` of cut ``trash``.  Inverting restores
the embed whole from the base, so trash-read copies invert to deletes.
Transform emits the same reads when a move's winning contested content
sits in an embed the concurrent delta deletes, and rebases moves and
formats racing an existing read onto the surviving copy.  A read racing
a concurrent claim on its source follows the delete-beats-move rule,
under either priority: a concurrent cut that re-homes the unit leaves
the salvage standing (only the read's own trash deletes it), while one
that gap-drops the unit is a deletion, and the copy dies with the
content.  Compose, transform and invert are total up to one precise
refusal: the list-shaped-payload case noted below.

Out of scope, deliberately: ``diff`` never emits moves (move detection is
a separate concern); ``Delta.length()`` counts a move twice (its cut and
its paste — use ``change_length`` for the net effect); transform cannot
rebuild a minimal embed patch whose destination path crosses a raw JSON
array (``{'rows': [{'ops': ...}]}``) — arrays are handler-opaque, so the
core can neither rebase their indices through concurrent mutations nor
encode an element-addressed patch, and routed edits into such windows
raise.  Model ordered collections as child *sequences* instead (an
``ops`` list of row embeds): sequence units rebase like any other
position, routed patches address them by retain hops, and moving a row
becomes an ordinary unit move.  And, as in the upstream quill algebra, a
delta that consumes more than its document is garbage in, garbage out.
"""

import copy
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Self

from delta import op
from delta.op import Attributes, Op

type Ops = list[Op]
type Payload = dict[str, Any]  # an embed's value or embed-change patch
type Ref = str


NULL_CHARACTER = chr(0)


def get_embed_type_and_data(a, b):
    if not isinstance(a, dict) or a is None:
        raise TypeError(f"cannot retain a {type(a).__name__}")
    if not isinstance(b, dict) or b is None:
        raise TypeError(f"cannot retain a {type(b).__name__}")
    if len(a) != 1 or len(b) != 1:
        raise ValueError("embed values must contain exactly one type")

    embed_type = next(iter(a))
    b_type = next(iter(b))
    if not embed_type or embed_type != b_type:
        raise ValueError(f"embed types not matched: {embed_type} != {b_type}")
    return [embed_type, a[embed_type], b[embed_type]]


handlers = {}


def _streams(payload):
    """Yield handler-declared ``(path, ops)`` child streams."""
    if not isinstance(payload, dict) or len(payload) != 1:
        return
    embed_type, data = next(iter(payload.items()))
    paths = getattr(handlers.get(embed_type), "stream_paths", None)
    if paths is None:
        return
    for path in paths(data):
        path = tuple(path)
        value = data
        for step in path:
            value = value[step]
        if not isinstance(value, list):
            raise TypeError(f"{embed_type!r} handler stream path {path!r} does not address a list")
        yield path, value


def _replaced(value, path, replacement):
    if not path:
        return replacement
    result = list(value) if isinstance(value, list) else dict(value)
    result[path[0]] = _replaced(result[path[0]], path[1:], replacement)
    return result


def _map_streams(payload, transform):
    if not isinstance(payload, dict) or len(payload) != 1:
        return None
    embed_type, data = next(iter(payload.items()))
    updated = data
    for path, _ops in list(_streams(payload)):
        current = updated
        for step in path:
            current = current[step]
        replacement = transform(current)
        if replacement is not None:
            updated = _replaced(updated, path, replacement)
    return None if updated is data else {embed_type: updated}


def _get_handler(embed_type):
    handler = handlers.get(embed_type)
    if handler is None:
        raise ValueError(f'no handlers for embed type "{embed_type}"')
    return handler


# Cross-level moves (a cut in an embed's child sequence paired with a
# paste at root, or vice versa) share one transaction per outermost call.
# Handlers that carry child sequences join it by composing/transforming/
# inverting them through MoveDelta, exactly like the fixtures' handler.
@dataclass(frozen=True, slots=True)
class Window:
    """
    One paste window of a cut: the [start, start+length) slice of the
    captured span, with the inline patch and embed change it applies.

    ``location`` is where the window pastes: ``('root', out_position)``
    or ``('child', hops, prefix)`` for a window inside embed payloads —
    one ``(unit, embed_type, keys)`` hop per nesting level.  Invert
    collects location-less windows (it only needs the source ordering).
    """

    start: int
    length: int
    attributes: Attributes | None
    change: Payload | None
    location: tuple[Any, ...] | None = None


@dataclass(slots=True)
class ComposeState:
    """One compose transaction, shared by every nesting level."""

    tables: dict[Ref, list[tuple]]  # ref -> segments its cut consumed
    taken: set[Ref]  # every ref in the transaction (part allocation)
    cuts: set[Ref]  # other's cut refs (paste-before-cut retry gate)
    trash: dict[int, Any]  # trash sites for deleted sourcing embeds
    # refs of paste destinations nested below a root operation: their
    # cuts must emit fresh part refs so the owner's recursive payload
    # expansion can never mistake generated output for an input window
    nested_pastes: set[Ref] = field(default_factory=set)
    retry: bool = False


@dataclass(slots=True)
class TransformState:
    """One transform transaction, shared by every nesting level."""

    self_windows: dict[Ref, list[tuple]]
    other_windows: dict[Ref, list[tuple]]
    other_reads: dict[Ref, list[dict[str, Any]]]
    taken: set[Ref]
    # other's refs -> how their cut sources fared
    state: dict[Ref, dict[str, Any]] = field(default_factory=dict)
    # (our ref, index) -> edits routed to a root window
    buckets: dict[tuple[Ref, int], list[tuple]] = field(default_factory=dict)
    # (our ref, index) -> edits routed into an embed
    overlays: dict[tuple[Ref, int], list[tuple]] = field(default_factory=dict)
    # read key -> edits routed into a trash-read output (owner pass only)
    read_buckets: dict[str, list[tuple]] = field(default_factory=dict)


@dataclass(slots=True)
class InvertState:
    """One invert transaction, shared by every nesting level."""

    windows: dict[Ref, list[tuple]]
    inverse_refs: dict[tuple[Ref, int], Ref]


@dataclass(frozen=True, slots=True)
class DiffState:
    """One recursive snapshot-diff transaction."""


def _checked_context(value, expected, operation):
    if value is not None and not isinstance(value, expected):
        raise TypeError(f"{operation} received a {type(value).__name__} context")
    return value


def _walk_move_ops(value, skip_inserts=False):
    """Yield moves from root or handler-owned streams, never opaque data."""
    if isinstance(value, list):
        for operation in value:
            if not isinstance(operation, dict):
                continue
            if isinstance(operation.get("cut"), dict) or isinstance(operation.get("paste"), dict):
                yield operation
            for carrier in ("insert", "retain"):
                if carrier == "insert" and skip_inserts:
                    continue
                payload = operation.get(carrier)
                if isinstance(payload, dict):
                    yield from _walk_move_ops(payload, skip_inserts)
            spec = operation.get("paste")
            if isinstance(spec, dict) and isinstance(spec.get("change"), dict):
                yield from _walk_move_ops(spec["change"], skip_inserts)
    elif isinstance(value, dict):
        for _path, child_ops in _streams(value):
            yield from _walk_move_ops(child_ops, skip_inserts)


def has_moves(delta):
    return next(_walk_move_ops(list(delta.ops)), None) is not None


def check(delta):
    """Validate the move invariants of a delta, transaction-wide: cut and
    paste halves may live at root or inside embed-change payloads."""
    cuts = {}
    windows = {}
    pathed = set()
    for operation in _walk_move_ops(list(delta.ops)):
        spec = operation.get("cut")
        if isinstance(spec, dict):
            if not isinstance(spec.get("ref"), str):
                raise ValueError("a cut needs a string reference")
            if not isinstance(spec.get("length"), int) or spec["length"] <= 0:
                raise ValueError("a cut needs a positive integer length")
            if spec["ref"] in cuts:
                raise ValueError(f"duplicate cut reference {spec['ref']!r}")
            cuts[spec["ref"]] = spec["length"]
        spec = operation.get("paste")
        if isinstance(spec, dict):
            if not isinstance(spec.get("ref"), str):
                raise ValueError("a paste needs a string reference")
            if (
                not isinstance(spec.get("start"), int)
                or spec["start"] < 0
                or not isinstance(spec.get("length"), int)
                or spec["length"] <= 0
            ):
                raise ValueError("a paste needs a non-negative integer start and a positive integer length")
            if spec.get("change") is not None and spec["length"] != 1:
                raise ValueError("a paste change must address one embed")
            if "path" in spec:  # reads through a trashed embed: no flat span
                pathed.add(spec["ref"])
            else:
                windows.setdefault(spec["ref"], []).append((spec["start"], spec["length"]))
    for ref in pathed:
        if ref not in cuts:
            raise ValueError(f"paste {ref!r} has no cut")
    for ref in cuts:
        if ref not in windows and ref not in pathed:
            raise ValueError(f"cut {ref!r} has no paste")
    for ref, spans in windows.items():
        if ref not in cuts:
            raise ValueError(f"paste {ref!r} has no cut")
        position = 0
        for start, length in sorted(spans):
            if start < position:
                raise ValueError(f"paste windows for {ref!r} overlap")
            position = start + length
        if position > cuts[ref]:
            raise ValueError(f"paste window for {ref!r} exceeds its cut")
    return delta


def _refs(ops):
    refs = set()
    for operation in _walk_move_ops(list(ops)):
        for key in ("cut", "paste"):
            spec = operation.get(key)
            if isinstance(spec, dict) and isinstance(spec.get("ref"), str):
                refs.add(spec["ref"])
    return refs


def _renamed(first_refs, ops):
    """Rename refs of ``ops`` that collide with ``first_refs``."""
    collisions = _refs(ops) & first_refs
    if not collisions:
        return list(ops)
    taken = first_refs | _refs(ops)
    renames = {}
    for ref in sorted(collisions):
        index = 2
        while f"{ref}~{index}" in taken:
            index += 1
        renames[ref] = f"{ref}~{index}"
        taken.add(renames[ref])
    renamed = copy.deepcopy(list(ops))
    for operation in _walk_move_ops(renamed):
        for key in ("cut", "paste"):
            spec = operation.get(key)
            if isinstance(spec, dict) and spec.get("ref") in renames:
                spec["ref"] = renames[spec["ref"]]
    return renamed


def _fresh_ref(ref, taken):
    """Claim a ref name, decorating it until it is unused."""
    while ref in taken:
        ref += "'"
    taken.add(ref)
    return ref


def _compose_embed_change(first, second, context):
    """Compose two retain-style embed changes into one."""
    if first is None:
        return second
    if second is None:
        return first
    embed_type, first_data, second_data = get_embed_type_and_data(first, second)
    return {embed_type: _get_handler(embed_type).compose(first_data, second_data, context)}


def _apply_embed_change(insert, change, context):
    """Apply a retain-style embed change to an embed insert payload."""
    if change is None:
        return insert
    embed_type, insert_data, change_data = get_embed_type_and_data(insert, change)
    return {embed_type: _get_handler(embed_type).apply(insert_data, change_data, context)}


def _invert_embed_change(change, base_insert, context):
    embed_type, change_data, base_data = get_embed_type_and_data(change, base_insert)
    payload = _get_handler(embed_type).invert(change_data, base_data, context)
    return {embed_type: payload} if payload else None


def _transform_embed_change(applied, other, priority, context):
    """Transform an embed change against a concurrently applied one."""
    if other is None:
        return None
    if applied is None:
        return other
    embed_type = next(iter(applied))
    if embed_type != next(iter(other)):
        return other
    payload = _get_handler(embed_type).transform(applied[embed_type], other[embed_type], priority, context)
    return {embed_type: payload} if payload else None


def _consume_cut(spec, self_it, out, shared, nested):
    """Consume the region a later cut covers, recording what it was made of.

    Base-backed characters become the composed cut (deleted spans are
    absorbed: they are cut but never pasted), earlier inserts and earlier
    paste windows are remembered so the paste sites can replay them.  An
    earlier cut inside the region passes through and splits the composed
    cut into parts, since the base spans around it stay contiguous.
    """
    remaining = spec["length"]
    taken = shared.taken
    segments = []  # (span, kind, ...) over the cut-local coordinate line
    current = None  # cut op of the open part, extended in place
    parts = 0

    def open_part():
        nonlocal current, parts
        if current is None:
            parts += 1
            ref = spec["ref"]
            if nested or ref in shared.nested_pastes:
                # a nested destination means the owner's recursive
                # payload expansion will see the emitted pieces: fresh
                # refs keep them distinct from the capture-table key
                ref = _fresh_ref(f"{ref}:{parts}", taken)
            elif parts > 1:
                ref = _fresh_ref(f"{ref}:{parts - 1}", taken)
            current = {"cut": {"ref": ref, "length": 0}}
            out.append(current)
        return current["cut"]

    while remaining > 0:
        self_type = self_it.peek_type()
        if self_type == "delete":
            open_part()["length"] += self_it.next()["delete"]
            continue
        if self_type == "cut":
            out.append(self_it.next())  # the earlier cut passes through whole
            current = None  # and splits this cut into a new part
            continue
        length = min(remaining, self_it.peek_length())
        piece = self_it.next(length) if self_it.peek() is not None else {"retain": length}
        piece_type = op.type(piece)
        if piece_type == "insert":
            segments.append((length, "insert", piece))
        elif piece_type == "paste":
            segments.append((length, "chain", piece))
        else:
            change = piece["retain"] if isinstance(piece.get("retain"), dict) else None
            part = open_part()
            if change is not None:
                for ref, path, inner_offset, _inner in _payload_cut_sites(change):
                    # if no window keeps this embed, its pending moves
                    # re-target through the outer cut by path
                    shared.trash[ref] = {
                        "ref": part["ref"],
                        "unit": part["length"],
                        "path": list(path),
                        "offset": inner_offset,
                    }
            segments.append((length, "base", part["ref"], part["length"], piece.get("attributes"), change))
            part["length"] += length
        remaining -= length
    shared.tables[spec["ref"]] = segments


def _expand_pathed(operation, segments, context):
    """Resolve a paste that reads through a trashed embed: locate the
    captured unit, then extract the addressed child span."""
    spec = operation["paste"]
    position = 0
    for segment in segments:
        span, kind = segment[0], segment[1]
        offset = spec.get("unit", 0) - position
        if 0 <= offset < span:
            if kind == "chain":
                inner = segment[2]["paste"]
                piece = copy.deepcopy(operation)
                piece["paste"]["ref"] = inner["ref"]
                piece["paste"]["unit"] = inner["start"] + offset
                return [piece]
            if kind == "base":
                # symbolic, but re-targeted at the composed cut part so
                # the unit survives absorbed inserts and deletes
                piece = copy.deepcopy(operation)
                piece["paste"].update(ref=segment[2], unit=segment[3] + offset)
                return [piece]
            if kind != "insert":
                return None
            insert = segment[2]["insert"]
            if not isinstance(insert, dict):
                return None
            value = _navigate_payload(insert[next(iter(insert))], spec["path"])
            if not isinstance(value, list):
                return None
            pieces = []
            child = MoveDelta._from_owned_ops(copy.deepcopy(value))
            for piece in child[spec["start"] : spec["start"] + spec["length"]].ops:
                new_op = {"insert": _apply_embed_change(piece["insert"], spec.get("change"), context)}
                attributes = op.compose(piece.get("attributes"), operation.get("attributes"), False)
                if attributes:
                    new_op["attributes"] = attributes
                pieces.append(new_op)
            return pieces
        position += span
    return None


def _expand(operation, tables, context):
    """Replay one paste window over the segments its cut consumed."""
    spec = operation["paste"]
    if "path" in spec:
        return _expand_pathed(operation, tables[spec["ref"]], context)
    patch = operation.get("attributes")
    own_change = spec.get("change")
    pieces = []
    position = 0
    for segment in tables[spec["ref"]]:
        span, kind = segment[0], segment[1]
        low = max(spec["start"], position)
        high = min(spec["start"] + spec["length"], position + span)
        if low < high:
            offset, size = low - position, high - low
            if kind == "base":
                piece_spec = {"ref": segment[2], "start": segment[3] + offset, "length": size}
                change = _compose_embed_change(segment[5], own_change, context)
                if change is not None:
                    piece_spec["change"] = change
                piece = {"paste": piece_spec}
                attributes = op.compose(segment[4], patch, True)
            elif kind == "insert":
                source = segment[2]
                insert = source["insert"]
                if isinstance(insert, str):
                    piece = {"insert": op.str_slice(insert, offset, offset + size)}
                else:
                    piece = {"insert": _apply_embed_change(insert, own_change, context)}
                attributes = op.compose(source.get("attributes"), patch, False)
            else:
                inner = segment[2]["paste"]
                piece_spec = {**inner, "start": inner["start"] + offset, "length": size}
                change = _compose_embed_change(inner.get("change"), own_change, context)
                if change is not None:
                    piece_spec["change"] = change
                piece = {"paste": piece_spec}
                attributes = op.compose(segment[2].get("attributes"), patch, True)
            if attributes:
                piece["attributes"] = attributes
            pieces.append(piece)
        position += span
    return pieces


def _expanded_payload(value, tables, shared):
    """Expand paste windows carried inside a freshly inserted embed's
    child sequences against the transaction's capture tables.  Returns
    the rewritten value, or None when nothing needed expanding."""

    def expand_sequence(sequence):
        changed = False
        merged = MoveDelta()
        for child_op in sequence:
            if not isinstance(child_op, dict):
                merged.ops.append(child_op)
                continue
            rewritten = child_op
            spec = child_op.get("paste")
            if isinstance(spec, dict) and isinstance(spec.get("change"), dict):
                change = _expanded_payload(spec["change"], tables, shared)
                if change is not None:
                    rewritten = {**rewritten, "paste": {**spec, "change": change}}
                    spec = rewritten["paste"]
                    changed = True
            for carrier in ("insert", "retain"):
                payload = rewritten.get(carrier)
                payload = _expanded_payload(payload, tables, shared) if isinstance(payload, dict) else None
                if payload is not None:
                    rewritten = {**rewritten, carrier: payload}
                    changed = True
            if isinstance(spec, dict):
                ref = spec.get("ref")
                if ref in tables:
                    for piece in _expand(rewritten, tables, shared):
                        merged.push(piece)
                    changed = True
                    continue
                if ref in shared.cuts:
                    shared.retry = True
            merged.push(rewritten)
        return merged.ops if changed else None

    return _map_streams(value, expand_sequence)


def _nested_paste_refs(ops):
    """Refs of paste operations occurring below a root operation — in
    embed payloads, child sequences, or a paste's change."""
    refs = set()
    for operation in ops:
        if not isinstance(operation, dict):
            continue
        payloads = [operation.get(carrier) for carrier in ("insert", "retain")]
        spec = operation.get("paste")
        if isinstance(spec, dict):
            payloads.append(spec.get("change"))
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            for nested in _walk_move_ops(payload):
                inner = nested.get("paste")
                if isinstance(inner, dict) and isinstance(inner.get("ref"), str):
                    refs.add(inner["ref"])
    return refs


def _composed_retain(self_op, other_op, length, context):
    """Compose one retain pairing."""
    if op.type(self_op) == "paste":
        spec = dict(self_op["paste"])
        if isinstance(other_op.get("retain"), dict):
            spec["change"] = _compose_embed_change(spec.get("change"), other_op["retain"], context)
        new_op = {"paste": spec}
        attributes = op.compose(self_op.get("attributes"), other_op.get("attributes"), True)
        if attributes:
            new_op["attributes"] = attributes
        return new_op

    new_op = {}
    if isinstance(self_op.get("retain"), (int, float)):
        new_op["retain"] = length if isinstance(other_op.get("retain"), (int, float)) else other_op["retain"]
    elif isinstance(other_op.get("retain"), (int, float)):
        if self_op.get("retain") is None:
            new_op["insert"] = self_op.get("insert")
        else:
            new_op["retain"] = self_op.get("retain")
    else:
        action = "insert" if self_op.get("retain") is None else "retain"
        embed_type, self_data, other_data = get_embed_type_and_data(self_op.get(action), other_op.get("retain"))
        handler = _get_handler(embed_type)
        method = handler.compose if action == "retain" else handler.apply
        new_op[action] = {embed_type: method(self_data, other_data, context)}
    attributes = op.compose(
        self_op.get("attributes"), other_op.get("attributes"), isinstance(self_op.get("retain"), (int, float))
    )
    if attributes:
        new_op["attributes"] = attributes
    return new_op


def _transformed_retain(self_op, other_op, length, priority, context):
    self_data = self_op.get("retain")
    other_data = other_op.get("retain")
    transformed_data = other_data if isinstance(other_data, dict) else length
    if isinstance(self_data, dict) and isinstance(other_data, dict):
        embed_type = next(iter(self_data))
        if embed_type == next(iter(other_data)):
            handler = _get_handler(embed_type)
            if handler:
                transformed_data = {
                    embed_type: handler.transform(self_data[embed_type], other_data[embed_type], priority, context)
                }
    new_op = {"retain": transformed_data}
    attributes = op.transform(self_op.get("attributes"), other_op.get("attributes"), priority)
    if attributes:
        new_op["attributes"] = attributes
    return new_op


def _covered_runs(windows, low, high):
    """Intersections of [low, high) with a ref's sorted paste windows."""
    runs = []
    for index, window in enumerate(windows):
        start, size = window.start, window.length
        run_low, run_high = max(low, start), min(high, start + size)
        if run_low < run_high:
            runs.append((index, run_low, run_high))
    return runs


def _collect_windows(ops):
    """Transaction-wide paste windows: {ref: sorted [(start, length,
    attributes, change, location)]}.  A location is coordinate-shaped:
    ``('root', output position)`` for a root window, or ``('child',
    hops, prefix)`` for a window inside embed changes, where each hop is
    ``(unit position, embed type, keys to the child sequence)`` — one
    hop per nesting level — and ``prefix`` is the paste's output offset
    inside the innermost sequence."""
    windows = {}
    position = 0
    for operation in ops:
        if op.type(operation) == "paste":
            spec = operation["paste"]
            if "path" not in spec:  # trash reads are opaque to routing
                windows.setdefault(spec["ref"], []).append(
                    Window(
                        spec["start"],
                        spec["length"],
                        operation.get("attributes"),
                        spec.get("change"),
                        ("root", position),
                    )
                )
            _collect_change_windows(spec, position, windows)
        elif isinstance(operation.get("retain"), dict):
            embed_type = next(iter(operation["retain"]))
            _collect_child_windows(operation["retain"], ((position, embed_type),), windows)
        elif isinstance(operation.get("insert"), dict):
            embed_type = next(iter(operation["insert"]))
            _collect_child_windows(operation["insert"], ((position, embed_type),), windows)
        position += op.output_length(operation)
    for spans in windows.values():
        spans.sort(key=lambda span: span.start)
    return windows


def _collect_change_windows(spec, position, windows, hops=()):
    """Moves may ride a paste's embed change; anchor them at the pasted
    embed's destination."""
    change = spec.get("change")
    if isinstance(change, dict) and change:
        change_type = next(iter(change))
        _collect_child_windows(change, (*hops, (position, change_type)), windows)


def _collect_child_windows(payload, hops, windows):
    for path, sequence in _streams(payload):
        unit, embed_type = hops[-1]
        here = (*hops[:-1], (unit, embed_type, path))
        prefix = 0
        for child_op in sequence:
            if not isinstance(child_op, dict):
                continue
            spec = child_op.get("paste")
            if isinstance(spec, dict) and isinstance(spec.get("ref"), str):
                if "path" not in spec:
                    windows.setdefault(spec["ref"], []).append(
                        Window(
                            spec["start"],
                            spec["length"],
                            child_op.get("attributes"),
                            spec.get("change"),
                            ("child", here, prefix),
                        )
                    )
                _collect_change_windows(spec, prefix, windows, here)
            elif isinstance(child_op.get("retain"), dict):
                inner_type = next(iter(child_op["retain"]))
                _collect_child_windows(child_op["retain"], (*here, (prefix, inner_type)), windows)
            elif isinstance(child_op.get("insert"), dict):
                inner_type = next(iter(child_op["insert"]))
                _collect_child_windows(child_op["insert"], (*here, (prefix, inner_type)), windows)
            prefix += op.output_length(child_op)


def _unit_patch(delta, unit, windows=None):
    """The embed patch a delta applies to the unit at input ``unit``: an
    embed-change payload, or the change riding its covering paste."""
    if windows is None:
        windows = _collect_windows(list(delta.ops))
    input_position = 0
    for operation in delta.ops:
        length = op.input_length(operation)
        if input_position <= unit < input_position + length:
            if isinstance(operation.get("retain"), dict):
                data = operation["retain"]
                return data[next(iter(data))]
            if op.type(operation) == "cut":
                offset = unit - input_position
                ref = operation["cut"]["ref"]
                for window in windows.get(ref, ()):
                    if window.start <= offset < window.start + window.length:
                        for candidate in delta.ops:
                            if op.type(candidate) != "paste":
                                continue
                            spec = candidate["paste"]
                            if spec["ref"] == ref and spec["start"] == window.start and spec.get("change") is not None:
                                change = spec["change"]
                                return change[next(iter(change))]
                        return None
            return None
        input_position += length
    return None


def _deposit(shared, ref, windows, index, offset, routed, local_pastes=frozenset()):
    """File a routed edit under its window.  Windows expanding in the
    routing sequence itself (root windows, or same-sequence child moves)
    lay out in stream; windows in other sequences become destination
    overlays composed at the end."""
    window = windows[index]
    in_stream = window.location[0] == "root" or (ref, window.start) in local_pastes
    target = shared.buckets if in_stream else shared.overlays
    target.setdefault((ref, index), []).append((offset, routed))


def _route_delete(shared, ref, windows, low, high, local_pastes=frozenset()):
    """Route a deletion of our moved content into our paste windows."""
    for index, run_low, run_high in _covered_runs(windows, low, high):
        _deposit(
            shared, ref, windows, index, run_low - windows[index].start, {"delete": run_high - run_low}, local_pastes
        )


def _payload_has_pastes(payload):
    return any(isinstance(operation.get("paste"), dict) for operation in _walk_move_ops(payload))


def _drop_other_embed(payload, state, rebased=frozenset()):
    """The other side's embed was deleted under it: its cut sources are
    gone, so its windows renumber to nothing — except moves already
    rebased onto a trash read of the same content."""
    for operation in _walk_move_ops(payload):
        spec = operation.get("cut")
        if isinstance(spec, dict) and spec["ref"] not in rebased:
            mapping = _mapping(state, spec["ref"])
            mapping["current"] = None
            mapping["segments"].append((spec["length"], None, 0, None, None))


def _drop_self_embed(payload, shared, reads=()):
    """Our embed was deleted by the other side: a delete beats a move, so
    content we moved out of it dies at our windows — except sub-ranges
    the other side's trash reads rescue, whose moves get rebased onto the
    surviving copies and whose read attributes still format the content.
    (Windows *inside* the embed need no marking — their destination unit
    maps to nothing, which the overlay pass detects.)"""
    for ref, path, offset, length in _payload_cut_sites(payload):
        spans = shared.self_windows.get(ref, [])
        rescued = []
        for read in reads:
            if tuple(read["path"]) != tuple(path):
                continue
            low = max(read["start"], offset) - offset
            high = min(read["start"] + read["length"], offset + length) - offset
            if low < high:
                rescued.append((low, high, read.get("attributes")))
        rescued.sort()
        for index, span in enumerate(spans):
            window_low, window_length = span.start, span.length
            cursor = window_low
            for low, high, attributes in rescued:
                low = max(low, window_low)
                high = min(high, window_low + window_length)
                if low < high:
                    if low > cursor:
                        _deposit(shared, ref, spans, index, cursor - window_low, {"delete": low - cursor})
                    trimmed = op.transform(span.attributes, attributes, True)
                    if trimmed:
                        _deposit(
                            shared, ref, spans, index, low - window_low, {"retain": high - low, "attributes": trimmed}
                        )
                    cursor = high
            if cursor < window_low + window_length:
                _deposit(
                    shared, ref, spans, index, cursor - window_low, {"delete": window_low + window_length - cursor}
                )


def _child_patch(path, sequence):
    """Rebuild a minimal embed patch nesting a child ops list along the
    JSON path where the destination sequence was found."""
    value = sequence
    for part in reversed(path):
        if not isinstance(part, str):
            raise ValueError("cannot rebuild a list-indexed child sequence patch")
        value = {part: value}
    return value


def _wrap_hops(hops, child_ops):
    """Wrap innermost child ops outward through a hop chain into the
    level-one child ops list of the outermost embed."""
    for unit, embed_type, keys in reversed(hops[1:]):
        inner = [{"retain": {embed_type: _child_patch(keys, child_ops)}}]
        child_ops = [{"retain": unit}, *inner] if unit else inner
    return child_ops


def _flatten_hops(hops):
    """A hop chain as a trash-read path: keys with integer unit offsets
    marking each descent into a deeper sequence."""
    path = list(hops[0][2])
    for unit, _embed_type, keys in hops[1:]:
        path.append(unit)
        path.extend(keys)
    return path


def _rewrite_payload(value, state, priority, context):
    """Renumber paste windows inside an embed payload the transform loop
    passed through untouched."""

    def rewrite(sequence):
        rebuilt, changed = [], False
        for child_op in sequence:
            if not isinstance(child_op, dict):
                rebuilt.append(child_op)
                continue
            spec = child_op.get("paste")
            if isinstance(spec, dict) and spec.get("ref") in state:
                pieces = _renumber(child_op, state, priority, context)
                if pieces is not None:
                    rebuilt.extend(pieces)
                    changed = True
                    continue
            rewritten = child_op
            for carrier in ("retain", "insert"):
                payload = rewritten.get(carrier)
                replacement = _rewrite_payload(payload, state, priority, context) if isinstance(payload, dict) else None
                if replacement is not None:
                    rewritten = {**rewritten, carrier: replacement}
                    changed = True
            spec = rewritten.get("paste")
            change = spec.get("change") if isinstance(spec, dict) else None
            replacement = _rewrite_payload(change, state, priority, context) if isinstance(change, dict) else None
            if replacement is not None:
                rewritten = {**rewritten, "paste": {**spec, "change": replacement}}
                changed = True
            rebuilt.append(rewritten)
        if not changed:
            return None
        merged = MoveDelta()
        for child_op in rebuilt:
            if isinstance(child_op, dict):
                merged.push(child_op)
            else:
                merged.ops.append(child_op)
        return merged.ops

    return _map_streams(value, rewrite)


def _new_part(mapping, ref, taken):
    """Allocate the next split-part ref: r, r:1, r:2, ..."""
    mapping["parts"] += 1
    if mapping["parts"] == 1:
        return ref
    return _fresh_ref(f"{ref}:{mapping['parts'] - 1}", taken)


def _mapping(state, ref):
    return state.setdefault(ref, {"segments": [], "parts": 0, "current": None})


def _unit_coordinate(delta, index):
    """Where the input unit at ``index`` lives after the delta: an integer
    root output position, ``('nested', hops, offset)`` when a move carried
    it inside embed payloads, or None when it no longer exists."""
    input_position = 0
    for operation in delta.ops:
        kind = op.type(operation)
        offset = index - input_position
        if kind == "cut" and 0 <= offset < operation["cut"]["length"]:
            windows = _collect_windows(list(delta.ops))
            for window in windows.get(operation["cut"]["ref"], ()):
                start, location = window.start, window.location
                if start <= offset < start + window.length:
                    if location[0] == "root":
                        return location[1] + offset - start
                    _, hops, prefix = location
                    return ("nested", hops, prefix + offset - start)
            return None  # cut but never pasted
        if kind == "delete" and 0 <= offset < operation["delete"]:
            return None
        input_position += op.input_length(operation)
    return delta.transform_position(index)


def _unit_position(delta, index):
    """Map one input *unit* through a delta: unlike a caret, a unit at a
    window's first offset belongs to the window and follows the move.
    Returns None when the unit no longer exists at root after the delta."""
    located = _unit_coordinate(delta, index)
    return located if isinstance(located, int) else None


def _read_spans(edit_ops, start, end):
    """Output runs holding the surviving images of child input
    [start, end) after a concurrent child patch: retained content maps
    across, inserts strictly inside join, and content a concurrent move
    or delete claims leaves the span."""
    runs = []
    input_position = 0
    output_position = 0
    for child_op in edit_ops:
        if not isinstance(child_op, dict):
            continue
        kind = op.type(child_op)
        input_length = op.input_length(child_op)
        output_length = op.output_length(child_op)
        if kind == "retain":
            low = max(input_position, start)
            high = min(input_position + input_length, end)
            if low < high:
                runs.append([output_position + low - input_position, high - low])
        elif kind == "insert" and start < input_position < end:
            runs.append([output_position, output_length])
        input_position += input_length
        output_position += output_length
    if input_position < end:  # the implicit tail retain
        low = max(input_position, start)
        runs.append([output_position + low - input_position, end - low])
    runs.sort()
    merged = []
    for low, length in runs:
        if length <= 0:
            continue
        if merged and merged[-1][0] + merged[-1][1] == low:
            merged[-1][1] += length
        else:
            merged.append([low, length])
    return merged


def _route_into_reads(edit_ops, reads, read_buckets, state, taken, alive_refs=frozenset()):
    """Route one concurrent child patch over trash reads of the same
    sequence into the reads' output: formats and deletes apply to the
    surviving copies, inserts strictly inside join them, and a concurrent
    move whose destination survives elsewhere re-cuts its claim out of
    the copy.  Moves confined to the dying content lose it, like the rest
    of the trash.  Returns the refs of moves rebased onto the reads."""
    reads = sorted(reads, key=lambda read: read["start"])
    rebased = set()
    position = 0
    for child_op in edit_ops:
        if not isinstance(child_op, dict):
            continue
        kind = op.type(child_op)
        if kind == "insert":
            for read in reads:
                if read["start"] < position < read["start"] + read["length"]:
                    joined = dict(child_op)
                    attributes = op.compose(child_op.get("attributes"), read.get("attributes"), False)
                    joined.pop("attributes", None)
                    if attributes:
                        joined["attributes"] = attributes
                    read_buckets[read["key"]].append((position - read["start"], joined))
            continue
        if kind == "paste":
            continue  # its content dies with the cell or was re-cut
        length = op.input_length(child_op)
        if kind == "cut" and child_op["cut"]["ref"] in alive_refs:
            inner_ref = child_op["cut"]["ref"]
            rebased.add(inner_ref)
            mapping = _mapping(state, inner_ref)
            mapping["current"] = None
            cursor = position
            for read in reads:
                low = max(position, read["start"])
                high = min(position + length, read["start"] + read["length"])
                if low >= high:
                    continue
                if low > cursor:
                    mapping["segments"].append((low - cursor, None, 0, None, None))
                part = _new_part(mapping, inner_ref, taken)
                read_buckets[read["key"]].append((low - read["start"], {"cut": {"ref": part, "length": high - low}}))
                mapping["segments"].append((high - low, part, 0, None, None))
                cursor = high
            if cursor < position + length:
                mapping["segments"].append((position + length - cursor, None, 0, None, None))
        elif kind == "cut":
            # a move confined to the dying content loses it entirely
            for read in reads:
                low = max(position, read["start"])
                high = min(position + length, read["start"] + read["length"])
                if low < high:
                    read_buckets[read["key"]].append((low - read["start"], {"delete": high - low}))
        else:
            for read in reads:
                low = max(position, read["start"])
                high = min(position + length, read["start"] + read["length"])
                if low >= high:
                    continue
                if kind == "delete":
                    read_buckets[read["key"]].append((low - read["start"], {"delete": high - low}))
                elif kind == "retain" and (child_op.get("attributes") or isinstance(child_op.get("retain"), dict)):
                    routed = (
                        {"retain": child_op["retain"]}
                        if isinstance(child_op.get("retain"), dict)
                        else {"retain": high - low}
                    )
                    if child_op.get("attributes"):
                        routed["attributes"] = child_op["attributes"]
                    read_buckets[read["key"]].append((low - read["start"], routed))
        position += length
    return rebased


def _routed_entry(routed, state, priority, context):
    """Unwrap a deferred routed edit, renumbering payload windows now
    that the transaction state is complete."""
    if isinstance(routed, tuple):
        rewritten = copy.deepcopy(routed[1])
        replacement = _rewrite_payload(rewritten["retain"], state, priority, context)
        if replacement is not None:
            rewritten["retain"] = replacement
        return rewritten
    return routed


def _capture_trash(result, position):
    """The capture holding the deleted unit at input ``position`` of
    ``result``, minting a trash cut out of a plain delete if needed.
    Returns (result, ref, unit) or None."""
    input_position = 0
    for index, operation in enumerate(result.ops):
        length = op.input_length(operation)
        if input_position <= position < input_position + length:
            offset = position - input_position
            if op.type(operation) == "cut":
                return result, operation["cut"]["ref"], offset
            if op.type(operation) == "delete":
                trash_ref = _fresh_ref("trash", _refs(result.ops))
                ops = list(result.ops)
                replacement = []
                if offset:
                    replacement.append({"delete": offset})
                replacement.append({"cut": {"ref": trash_ref, "length": 1}})
                if length - offset - 1:
                    replacement.append({"delete": length - offset - 1})
                ops[index : index + 1] = replacement
                return result.__class__(ops), trash_ref, 0
            return None
        input_position += length
    return None


def _read_from_trash(result, position, path, entries):
    """Re-target routed cut parts whose destination window died: their
    windows read the content out of the capture that swallowed the host
    embed.  Other routed edits die with it."""
    trash_map = {}
    for offset, routed in entries:
        if isinstance(routed, tuple):
            routed = routed[1]
        if not isinstance(routed.get("cut"), dict):
            continue
        found = _capture_trash(result, position)
        if found is None:
            raise ValueError("concurrent deletion of an embed that still sources a move is not representable")
        result, trash_ref, unit = found
        trash_map[routed["cut"]["ref"]] = {"ref": trash_ref, "unit": unit, "path": list(path), "offset": offset}
    return _retarget_trashed(result, trash_map) if trash_map else result


def _sequence_at(delta, position, path):
    """The child ops list at ``path`` inside the patch the delta applies
    to the unit at input ``position`` — whether the patch sits in an
    embed change or rides the unit's covering paste."""
    value = _unit_patch(delta, position)
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value if isinstance(value, list) else None


def _navigate_patch(value, path):
    """Navigate a change payload along a trash-read path: string parts
    are payload keys, integer parts descend through the retain covering
    that input offset of a child patch."""
    for part in path:
        if isinstance(part, int):
            if not isinstance(value, list):
                return None
            cursor = 0
            found = None
            for child_op in value:
                if not isinstance(child_op, dict):
                    return None
                size = op.input_length(child_op)
                if cursor <= part < cursor + size:
                    if isinstance(child_op.get("retain"), dict):
                        found = child_op["retain"]
                    break
                cursor += size
            if not isinstance(found, dict):
                return None
            value = found[next(iter(found))]
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    return value


def _navigate_payload(value, path):
    """Navigate content along a trash-read path: string parts are payload
    keys, integer parts descend into the unit at that offset of a child
    snapshot."""
    for part in path:
        if isinstance(part, int):
            if not isinstance(value, list):
                return None
            cursor = 0
            unit = None
            for child_op in value:
                if not isinstance(child_op, dict):
                    return None
                size = op.output_length(child_op)
                if cursor <= part < cursor + size:
                    unit = child_op.get("insert")
                    break
                cursor += size
            if not isinstance(unit, dict):
                return None
            value = unit[next(iter(unit))]
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    return value


def _collect_captures(ops, base, captures):
    """Record every cut's captured content — root or nested — by walking
    change payloads against the matching base payloads."""
    position = 0
    for operation in ops:
        if not isinstance(operation, dict):
            continue
        if op.type(operation) == "cut":
            captures[operation["cut"]["ref"]] = base[position : position + operation["cut"]["length"]].ops
        elif isinstance(operation.get("retain"), dict):
            unit = base[position : position + 1].ops
            snapshot = unit[0].get("insert") if unit else None
            if isinstance(snapshot, dict):
                _collect_payload_captures(operation["retain"], snapshot, captures)
        position += op.input_length(operation)


def _collect_payload_captures(change, snapshot, captures):
    if not isinstance(change, dict) or not isinstance(snapshot, dict) or len(change) != 1:
        return
    embed_type = next(iter(change))
    snap_data = snapshot.get(embed_type)
    for path, sequence in _streams(change):
        snap = snap_data
        try:
            for step in path:
                snap = snap[step]
        except KeyError, IndexError, TypeError:
            continue
        if isinstance(snap, list):
            _collect_captures(sequence, MoveDelta(list(snap)), captures)


def _lower_ops(ops, captures):
    lowered = MoveDelta()
    for operation in ops:
        kind = op.type(operation)
        if kind == "cut":
            lowered.delete(operation["cut"]["length"])
        elif kind == "paste":
            for piece in _capture_read(operation, captures):
                lowered.push(piece)
        elif isinstance(operation.get("retain"), dict):
            payload = copy.deepcopy(operation["retain"])
            new_op = dict(operation)
            new_op["retain"] = _lower_payload(payload, captures) or payload
            lowered.push(new_op)
        elif isinstance(operation.get("insert"), dict):
            payload = copy.deepcopy(operation["insert"])
            new_op = dict(operation)
            new_op["insert"] = _lower_payload(payload, captures) or payload
            lowered.push(new_op)
        else:
            lowered.push(operation)
    return lowered.chop()


def _lower_payload(value, captures):
    return _map_streams(value, lambda sequence: _lower_ops(sequence, captures).ops)


def _capture_read(operation, captures):
    """Materialize one paste window out of its capture."""
    spec = operation["paste"]
    content = captures.get(spec["ref"], [])
    if "path" in spec:
        content = _navigate_payload(list(content), [spec.get("unit", 0), *spec["path"]])
        if not isinstance(content, list):
            return []
    pieces = []
    window = MoveDelta._from_owned_ops(copy.deepcopy(list(content)))[spec["start"] : spec["start"] + spec["length"]]
    for piece in window.ops:
        new_op = {"insert": _apply_embed_change(piece["insert"], spec.get("change"), None)}
        attributes = op.compose(piece.get("attributes"), operation.get("attributes"), False)
        if attributes:
            new_op["attributes"] = attributes
        pieces.append(new_op)
    return pieces


def _apply_overlays(result, shared, priority):
    """Compose routed edits whose destination windows live inside embed
    payloads onto the transformed delta, as minimal embed changes.  When
    the destination embed itself moved, the patch rides its window; when
    the transformed delta already patches the destination sequence, the
    overlay is rebased through that patch first."""
    grouped = {}
    for (ref, index), entries in shared.overlays.items():
        window = shared.self_windows[ref][index]
        _, hops, prefix = window.location
        bucket = grouped.setdefault(hops, [])
        bucket.extend((prefix + offset, routed) for offset, routed in entries)
    items = []
    for hops, entries in grouped.items():
        child_ops, _span = _laid_out(entries, lambda routed: _routed_entry(routed, shared.state, priority, shared))
        items.append((hops, child_ops))
    items.sort(key=lambda item: item[0][0][0])
    for hops, child_ops in items:
        located = _unit_coordinate(result, hops[0][0])
        if located is None:  # the destination embed no longer exists
            offsets = []
            cursor = 0
            for child_op in child_ops:
                offsets.append((cursor, child_op))
                cursor += op.length(child_op)
            result = _read_from_trash(result, hops[0][0], _flatten_hops(hops), offsets)
            continue
        if isinstance(located, tuple):
            # the destination embed itself moved inside another embed:
            # extend the chain through its new host, whose anchor op sits
            # at a root output position of the result
            _, host_hops, offset = located
            hops = (*host_hops, (offset, hops[0][1], hops[0][2]), *hops[1:])
            mapped = host_hops[0][0]
            existing = None  # the host anchor is the mover's own op
        else:
            mapped = located
            existing = _sequence_at(result, hops[0][0], hops[0][2])
        _position, embed_type, path = hops[0]
        child_ops = _wrap_hops(hops, child_ops)
        if existing is not None:  # rebase through their own edits there
            child_ops = (
                MoveDelta._from_owned_ops(copy.deepcopy(list(existing)))
                ._transform_transaction(MoveDelta(child_ops), False)
                .ops
            )
            if not child_ops:
                continue
        overlay = result.__class__().retain(mapped)
        overlay.push({"retain": {embed_type: _child_patch(path, list(child_ops))}})
        result = result._compose_unchecked(overlay)
    return result


def _route_cut(spec, other_it, shared, out, priority, local_pastes, local_reads=None, read_buckets=None):
    """Route the other side's edits on our cut region to its paste windows.

    Formats and deletes address the moved content and follow it; inserts
    address a position and stay at the source.  A concurrent cut of the
    same content is contested: with priority our move keeps it (their
    windows shrink), without it their cut is rebased — re-targeted at our
    paste windows, split into parts per window.
    """
    ref = spec["ref"]
    windows = shared.self_windows.get(ref, [])
    state = shared.state
    length = spec["length"]
    offset = 0
    while offset < length:
        if other_it.peek_type() in ("insert", "paste"):
            out.append(other_it.next())  # stays at the source position
            continue
        piece_length = min(length - offset, other_it.peek_length())
        piece = other_it.next(piece_length) if other_it.peek() is not None else {"retain": piece_length}
        piece_type = op.type(piece)
        low, high = offset, offset + piece_length
        if piece_type == "cut":
            mapping = _mapping(state, piece["cut"]["ref"])
            mapping["current"] = None
            if priority:
                # Our move wins: their claim on this run drops — but their
                # window gaps are deletions and their window attributes
                # are formats, and both still apply to the content, so
                # they land on our paste windows.  Units their trash
                # reads address survive as re-cuts the reads re-anchor to.
                their_ref = piece["cut"]["ref"]
                run_start = sum(span for span, *_ in mapping["segments"])
                theirs = shared.other_windows.get(their_ref, [])
                read_units = sorted({read["unit"] for read in shared.other_reads.get(their_ref, ())})

                def kill_reads(our_low, our_high):
                    # their gap-delete follows the content into copies
                    # our reads salvage: the copies die with it
                    if not local_reads:
                        return
                    for read in local_reads.get(ref, ()):
                        if our_low <= read["unit"] < our_high:
                            read_buckets[read["key"]].append((0, {"delete": read["length"]}))

                def drop_run(their_low, their_high):
                    cursor = their_low
                    for unit in read_units:
                        if not their_low <= unit < their_high:
                            continue
                        our_unit = low + unit - run_start
                        if unit > cursor:
                            mapping["segments"].append((unit - cursor, None, 0, None, None))
                            _route_delete(shared, ref, windows, low + cursor - run_start, our_unit, local_pastes)
                            kill_reads(low + cursor - run_start, our_unit)
                        covering = _covered_runs(windows, our_unit, our_unit + 1)
                        if covering:
                            part = _new_part(mapping, their_ref, shared.taken)
                            for index, run_low, _run_high in covering:
                                _deposit(
                                    shared,
                                    ref,
                                    windows,
                                    index,
                                    run_low - windows[index].start,
                                    {"cut": {"ref": part, "length": 1}},
                                    local_pastes,
                                )
                            mapping["segments"].append((1, part, 0, None, None))
                        else:
                            # both sides dropped the unit and we win:
                            # their read loses it with everything else
                            mapping["segments"].append((1, None, 0, None, None))
                        cursor = unit + 1
                    if cursor < their_high:
                        mapping["segments"].append((their_high - cursor, None, 0, None, None))
                        _route_delete(
                            shared, ref, windows, low + cursor - run_start, low + their_high - run_start, local_pastes
                        )
                        kill_reads(low + cursor - run_start, low + their_high - run_start)

                position = run_start
                for their_index, cov_low, cov_high in _covered_runs(theirs, run_start, run_start + piece_length):
                    if cov_low > position:
                        drop_run(position, cov_low)
                    their_window = theirs[their_index]
                    if their_window.attributes or their_window.change:
                        for index, run_low, run_high in _covered_runs(
                            windows, low + cov_low - run_start, low + cov_high - run_start
                        ):
                            attributes = op.transform(
                                windows[index].attributes, theirs[their_index].attributes, priority
                            )
                            change = _transform_embed_change(
                                windows[index].change, theirs[their_index].change, priority, shared
                            )
                            if change is not None:
                                routed = {"retain": change}
                            elif attributes:
                                routed = {"retain": run_high - run_low}
                            else:
                                continue
                            if attributes:
                                routed["attributes"] = attributes
                            _deposit(shared, ref, windows, index, run_low - windows[index].start, routed, local_pastes)
                    mapping["segments"].append((cov_high - cov_low, None, 0, None, None))
                    position = cov_high
                if position < run_start + piece_length:
                    drop_run(position, run_start + piece_length)
            else:  # their move wins: re-cut from our paste windows
                their_ref = piece["cut"]["ref"]
                run_start = sum(span for span, *_ in mapping["segments"])
                read_units = sorted({read["unit"] for read in shared.other_reads.get(their_ref, ())})

                def lost_run(our_low, our_high):
                    their_low = run_start + our_low - low
                    their_high = run_start + our_high - low
                    # a read addressing units in this run dies with the
                    # content: a copy out of trash never survives a
                    # concurrent claim on its source
                    mapping["segments"].append((their_high - their_low, None, 0, None, None))

                position = low
                for index, run_low, run_high in _covered_runs(windows, low, high):
                    if run_low > position:
                        lost_run(position, run_low)
                    part_ref = _new_part(mapping, their_ref, shared.taken)
                    _deposit(
                        shared,
                        ref,
                        windows,
                        index,
                        run_low - windows[index].start,
                        {"cut": {"ref": part_ref, "length": run_high - run_low}},
                        local_pastes,
                    )
                    mapping["segments"].append((run_high - run_low, part_ref, 0, None, None))
                    position = run_high
                if position < high:
                    lost_run(position, high)
                if local_reads:
                    # their winning claim covers units our reads salvage:
                    # if their windows re-home a unit, only our own trash
                    # deletes it and the salvage stands; if they gap-drop
                    # it, their deletion beats the read and the copy dies
                    for read in local_reads.get(ref, ()):
                        if not low <= read["unit"] < high:
                            continue
                        their_unit = run_start + read["unit"] - low
                        if _covered_runs(shared.other_windows.get(their_ref, []), their_unit, their_unit + 1):
                            continue
                        read_buckets[read["key"]].append((0, {"delete": read["length"]}))
        elif piece_type == "delete":
            _route_delete(shared, ref, windows, low, high, local_pastes)
            if local_reads:  # deleting a trashed unit kills its reads
                for read in local_reads.get(ref, ()):
                    if low <= read["unit"] < high:
                        read_buckets[read["key"]].append((0, {"delete": read["length"]}))
        elif piece.get("attributes") or isinstance(piece.get("retain"), dict):
            rebased = set()
            if isinstance(piece.get("retain"), dict) and local_reads:
                # trash reads of this unit route the concurrent patch of
                # their addressed content into the surviving copies; a
                # concurrent move only keeps its claim if any of its
                # windows survive outside the dying payload
                payload_windows = set()
                for nested in _walk_move_ops(piece["retain"]):
                    nested_spec = nested.get("paste")
                    if isinstance(nested_spec, dict) and "path" not in nested_spec:
                        payload_windows.add((nested_spec["ref"], nested_spec["start"], nested_spec["length"]))
                alive = set()
                for nested in _walk_move_ops(piece["retain"]):
                    nested_spec = nested.get("cut")
                    if not isinstance(nested_spec, dict):
                        continue
                    for window in shared.other_windows.get(nested_spec["ref"], ()):
                        if (nested_spec["ref"], window.start, window.length) not in payload_windows:
                            alive.add(nested_spec["ref"])
                groups = {}
                for read in local_reads.get(ref, ()):
                    if read["unit"] == low:
                        groups.setdefault(read["path"], []).append(read)
                for path, group in groups.items():
                    data = piece["retain"][next(iter(piece["retain"]))]
                    edited = _navigate_patch(data, list(path))
                    if isinstance(edited, list):
                        rebased |= _route_into_reads(edited, group, read_buckets, state, shared.taken, alive)
            covered = _covered_runs(windows, low, high)
            if isinstance(piece.get("retain"), dict) and not covered:
                # the embed fell in a window gap: it is deleted, and any
                # moves it still sourced lose their content
                _drop_other_embed(piece["retain"], state, rebased)
            for index, run_low, run_high in covered:
                window = windows[index]
                start, window_change = window.start, window.change
                attributes = op.transform(window.attributes, piece.get("attributes"), priority)
                if isinstance(piece.get("retain"), dict):
                    change = _transform_embed_change(window_change, piece["retain"], priority, shared)
                    if change is None and not attributes:
                        continue
                    routed = {"retain": change if change is not None else run_high - run_low}
                else:
                    if not attributes:
                        continue
                    routed = {"retain": run_high - run_low}
                if attributes:
                    routed["attributes"] = attributes
                if isinstance(routed.get("retain"), dict) and _payload_has_pastes(routed["retain"]):
                    routed = ("rewrite", routed)  # renumber at expansion
                _deposit(shared, ref, windows, index, run_low - start, routed, local_pastes)
        offset += piece_length


def _cut_piece(out, shared, spec, deleted, attributes, change=None):
    """Record one transformed slice of a cut, splitting refs when the
    concurrent delta inserted inside the source region."""
    ref = spec["ref"]
    state = shared.state
    taken = shared.taken
    mapping = _mapping(state, ref)
    if change is not None and next(_walk_move_ops(change), None) is not None:
        # our embed rides inside their cut; if their windows drop it, the
        # moves it hosts lose their content and windows — unless their
        # trash reads rescue the addressed spans
        run_start = sum(span for span, *_ in mapping["segments"])
        if not _covered_runs(shared.other_windows.get(ref, []), run_start, run_start + spec["length"]):
            reads = [read for read in shared.other_reads.get(ref, ()) if read["unit"] == run_start]
            _drop_self_embed(change, shared, reads)
    if deleted:
        mapping["segments"].append((spec["length"], None, 0, None, None))
        return
    last = out[-1] if out else None
    if isinstance(last, dict) and op.type(last) == "cut" and last["cut"]["ref"] == mapping["current"]:
        offset = last["cut"]["length"]
        last["cut"]["length"] += spec["length"]
    else:
        mapping["current"] = _new_part(mapping, ref, taken)
        offset = 0
        out.append({"cut": {"ref": mapping["current"], "length": spec["length"]}})
    mapping["segments"].append((spec["length"], mapping["current"], offset, attributes, change))


def _renumber(operation, state, priority, context):
    """Map one paste window through what happened to its cut source.
    Returns None when the paste needs no rewriting."""
    spec = operation["paste"]
    mapping = state.get(spec["ref"])
    if mapping is None:
        return None
    if "path" in spec:
        # a trash read follows its captured unit through the rows, and
        # its window through any concurrent patch of the trashed content
        unit = spec.get("unit", 0)
        position = 0
        for span, part_ref, offset, _attrs, change in mapping["segments"]:
            if position <= unit < position + span:
                if part_ref is None:
                    return []  # the capture itself was deleted
                piece = copy.deepcopy(operation)
                piece["paste"].update(ref=part_ref, unit=offset + unit - position)
                if change is not None:
                    edited = _navigate_patch(change[next(iter(change))], spec["path"])
                    if isinstance(edited, list):
                        pieces = []
                        for start, length in _read_spans(edited, spec["start"], spec["start"] + spec["length"]):
                            fragment = copy.deepcopy(piece)
                            fragment["paste"].update(start=start, length=length)
                            pieces.append(fragment)
                        return pieces
                return [piece]
            position += span
        return None  # beyond the recorded rows: untouched
    pieces = []
    position = 0
    for span, part_ref, offset, attributes, change in mapping["segments"]:
        low = max(spec["start"], position)
        high = min(spec["start"] + spec["length"], position + span)
        if low < high and part_ref is not None:
            piece_spec = {"ref": part_ref, "start": offset + low - position, "length": high - low}
            transformed_change = _transform_embed_change(change, spec.get("change"), priority, context)
            if transformed_change is not None:
                piece_spec["change"] = transformed_change
            piece = {"paste": piece_spec}
            transformed = op.transform(attributes, operation.get("attributes"), priority)
            if transformed:
                piece["attributes"] = transformed
            pieces.append(piece)
        position += span
    return pieces


def _laid_out(entries, unwrap):
    """Lay routed entries over their span, sorted, with retain gaps.
    Joined inserts consume none of the span, so the cursor advances by
    input length."""
    output = []
    cursor = 0
    for offset, piece in sorted(entries, key=lambda entry: entry[0]):
        piece = unwrap(piece)
        if offset > cursor:
            output.append({"retain": offset - cursor})
        output.append(piece)
        cursor = offset + op.input_length(piece)
    return output, cursor


def _expand_window(size, entries, unwrap=None):
    """Lay one paste window's routed edits over its span.  Without an
    ``unwrap``, deferred payload rewrites surface raw for a later pass."""
    if unwrap is None:

        def unwrap(piece):
            return piece[1] if isinstance(piece, tuple) else piece

    output, cursor = _laid_out(entries, unwrap)
    if cursor < size:
        output.append({"retain": size - cursor})
    return output


def _renumber_list(items, state, priority, context):
    """Renumber every paste — root or inside embed payloads — through
    what happened to its cut source, once the whole transaction settled.
    Runs over the raw item list so marker tuples pass through and can be
    expanded afterwards, catching any edits the renumbering itself
    routes into them."""
    if not state:
        return items
    result = []
    for item in items:
        if not isinstance(item, dict):
            result.append(item)
            continue
        operation = copy.deepcopy(item)
        spec = operation.get("paste")
        if isinstance(spec, dict) and spec.get("ref") in state:
            pieces = _renumber(operation, state, priority, context)
            if pieces is not None:
                result.extend(pieces)
                continue
        if isinstance(operation.get("retain"), dict):
            replacement = _rewrite_payload(operation["retain"], state, priority, context)
            if replacement is not None:
                operation["retain"] = replacement
        if isinstance(operation.get("insert"), dict):
            replacement = _rewrite_payload(operation["insert"], state, priority, context)
            if replacement is not None:
                operation["insert"] = replacement
        result.append(operation)
    return result


def _expand_markers(out, buckets, read_buckets, cls, state=None, priority=False, context=None):
    """Expand window and read markers.  The owner passes the settled
    state so deferred payload rewrites happen right here; children leave
    them raw for the owner's renumbering pass to catch inside payloads."""
    unwrap = None
    if state is not None:

        def unwrap(piece):
            return _routed_entry(piece, state, priority, context)

    def expand(item):
        if isinstance(item, tuple):
            if item[0] == "read":
                return _expand_window(item[2], read_buckets.get(item[1], []), unwrap)
            return _expand_window(item[3], buckets.get((item[1], item[2]), []), unwrap)
        return None

    return _assemble(out, expand, cls)


def _assemble(out, expand, cls):
    """Expand pending items and merge; orphan cuts are normalized by the
    transaction owner once every level has settled."""
    expanded = []
    for item in out:
        pieces = expand(item)
        if pieces is None:
            expanded.append(item)
        else:
            expanded.extend(pieces)
    delta = cls()
    for operation in expanded:
        delta.push(operation)
    return delta.chop()


def _payload_cut_sites(retain):
    """(ref, path, offset) for each cut in the child sequences of an
    embed change: a cut only ever consumes input, so its offset in the
    underlying content is the sum of the input lengths before it.

    Every cut is recorded — a ref may have windows both inside the
    dying payload (they vanish with it) and outside (they re-target
    through the trash); the later passes sort the cases out, and a
    trash cut nothing ever reads degrades back to a plain delete."""
    sites = []
    for path, sequence in _streams(retain):
        offset = 0
        for child in sequence:
            if not isinstance(child, dict):
                continue
            spec = child.get("cut")
            if isinstance(spec, dict):
                sites.append((spec["ref"], path, offset, spec["length"]))
            nested = child.get("retain")
            if isinstance(nested, dict):
                for ref, inner_path, inner_offset, length in _payload_cut_sites(nested):
                    sites.append((ref, (*path, offset, *inner_path), inner_offset, length))
            offset += op.input_length(child)
    return sites


def _trash_embed(retain, shared, out):
    """Turn the deletion of a move-sourcing embed into a trash cut: the
    embed is captured (never pasted whole), and the moves it sourced will
    re-target through it by path."""
    sites = _payload_cut_sites(retain)
    if not sites:
        return False
    trash_ref = _fresh_ref("trash", shared.taken)
    out.append({"cut": {"ref": trash_ref, "length": 1}})
    for ref, path, offset, _length in sites:
        shared.trash[ref] = {"ref": trash_ref, "unit": 0, "path": list(path), "offset": offset}
    return True


def _retarget_trashed(delta, trash):
    """Rewrite pastes whose cut vanished into a trashed embed so they read
    through the capture by path."""
    if not trash:
        return delta
    cuts = {
        operation["cut"]["ref"]
        for operation in _walk_move_ops(list(delta.ops))
        if isinstance(operation.get("cut"), dict)
    }
    ops = copy.deepcopy(list(delta.ops))
    for operation in list(_walk_move_ops(ops)):
        spec = operation.get("paste")
        if isinstance(spec, dict) and spec["ref"] not in cuts and spec["ref"] in trash and "path" not in spec:
            site = trash[spec["ref"]]
            spec.update(
                ref=site["ref"], unit=site["unit"], path=list(site["path"]), start=site["offset"] + spec["start"]
            )
    result = delta.__class__()
    for operation in ops:
        result.push(operation)
    return result.chop()


def _sourced(delta):
    """Reject a composed delta whose paste lost its cut — the source
    embed of a still-referenced move was deleted along the way."""
    cuts = set()
    for operation in _walk_move_ops(list(delta.ops)):
        spec = operation.get("cut")
        if isinstance(spec, dict):
            cuts.add(spec["ref"])
    for operation in _walk_move_ops(list(delta.ops)):
        spec = operation.get("paste")
        if isinstance(spec, dict) and spec["ref"] not in cuts:
            raise ValueError("cannot compose the deletion of an embed that still sources a move")
    return delta


def _normalize_orphans(delta):
    """Degrade internally produced cuts whose ref is pasted nowhere —
    root or any child sequence — into plain deletes.  Public operands are
    checked before composition/transform and reject such orphan cuts; this
    pass only closes over transient cuts created by the zipper."""
    pasted = set()
    cut_refs = set()
    for operation in _walk_move_ops(list(delta.ops)):
        spec = operation.get("paste")
        if isinstance(spec, dict):
            pasted.add(spec["ref"])
        spec = operation.get("cut")
        if isinstance(spec, dict):
            cut_refs.add(spec["ref"])
    if cut_refs <= pasted:
        return delta
    ops = copy.deepcopy(list(delta.ops))
    for operation in list(_walk_move_ops(ops)):
        spec = operation.get("cut")
        if isinstance(spec, dict) and spec["ref"] not in pasted:
            operation.clear()
            operation["delete"] = spec["length"]
    result = delta.__class__()
    for operation in ops:
        result.push(operation)
    return result.chop()


class MoveDelta:
    @staticmethod
    def register_embed(embed_type: str, handler: Any) -> None:
        handlers[embed_type] = handler

    @staticmethod
    def unregister_embed(embed_type: str) -> None:
        handlers.pop(embed_type, None)

    get_handler = staticmethod(_get_handler)

    @classmethod
    def _from_owned_ops(cls, ops: Ops, **attrs: Any) -> Self:
        """Internal ownership transfer: ``ops`` must not be reused."""
        delta = cls(**attrs)
        delta.ops = ops if isinstance(ops, list) else list(ops)
        return delta

    def __init__(self, ops: Ops | Self | None = None, **attrs: Any) -> None:
        ops = getattr(ops, "ops", ops)
        self.ops = [op.clone(operation) for operation in ops] if ops else []
        self.__dict__.update(attrs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MoveDelta):
            return NotImplemented
        return self.ops == other.ops

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.ops})"

    def __iter__(self):
        return iter(self.ops)

    def __len__(self) -> int:
        return sum(op.length(o) for o in self.ops)

    def insert(self, text: str | Payload, **attrs: Any) -> Self:
        if text == "":
            return self
        new_op = {"insert": text}
        if attrs:
            new_op["attributes"] = attrs
        return self.push(new_op)

    def delete(self, length: int) -> Self:
        if length <= 0:
            return self
        return self.push({"delete": length})

    def retain(self, length: int | float | Payload, **attrs: Any) -> Self:
        if isinstance(length, (int, float)) and length <= 0:
            return self
        new_op = {"retain": length}
        if attrs:
            new_op["attributes"] = attrs
        return self.push(new_op)

    def extend(self, ops: Ops | Self) -> Self:
        ops = getattr(ops, "ops", ops)
        if not ops:
            return self
        ops = list(ops)
        self.push(ops[0])
        self.ops.extend(map(op.clone, ops[1:]))
        return self

    def concat(self, other: Self) -> Self:
        delta = self.__class__._from_owned_ops([op.clone(operation) for operation in self.ops])
        delta.extend(other)
        return delta

    def chop(self) -> Self:
        if self.ops:
            last_op = self.ops[-1]
            if isinstance(last_op.get("retain"), (int, float)) and not last_op.get("attributes"):
                self.ops.pop()
        return self

    def document(self) -> str:
        parts = []
        for o in self:
            insert = o.get("insert")
            if insert or insert == "":
                if isinstance(insert, str):
                    parts.append(insert)
                else:
                    parts.append(NULL_CHARACTER)
            else:
                raise ValueError("document() can only be called on Deltas that have only insert ops")
        return "".join(parts)

    def iterator(self) -> op.Iterator:
        return op.iterator(self.ops)

    def length(self) -> int | float:
        return sum(op.length(o) for o in self)

    def diff(self, other: Self, cursor: int | None = None, *, _context: DiffState | None = None) -> Self:
        """
        Deterministic typed snapshot diff between two *documents*
        (Deltas with only insert ops): retains, inserts, deletes and
        embed-retain patches only — never cut/paste.  ``cursor`` is an
        optional caret hint (a base-document UTF-16 position) anchoring
        an ambiguous edit where the editor actually made it.  See
        delta/diff.py for the atom model and alignment policy.
        """
        from delta.diff import snapshot_diff, snapshot_diff_at

        context = _checked_context(_context, DiffState, "diff") or DiffState()
        if cursor is None:
            return snapshot_diff(self, other, context)
        return snapshot_diff_at(self, other, cursor, context)

    def each_line(self, fn: Any, newline: str = "\n") -> None:
        for line, attributes, index in self.iter_lines(newline):
            if fn(line, attributes, index) is False:
                break

    def iter_lines(self, newline: str = "\n"):
        it = self.iterator()
        line = self.__class__()
        i = 0
        while it.has_next():
            if it.peek_type() != "insert":
                return
            current_op = it.peek()
            start = op.length(current_op) - it.peek_length()
            if isinstance(current_op.get("insert"), str):
                suffix = op.str_slice(current_op["insert"], start)
                found = suffix.find(newline)
                nl_index = op.str_length(suffix[:found]) if found >= 0 else -1
            else:
                nl_index = -1

            if nl_index < 0:
                line.push(it.next())
            elif nl_index > 0:
                line.push(it.next(nl_index))
            else:
                attributes = it.next(1).get("attributes", {})
                yield line, attributes, i
                i += 1
                line = self.__class__()
        if len(line) > 0:
            yield line, {}, i

    def cut(self, ref: Ref, length: int) -> Self:
        if length <= 0:
            return self
        return self.push({"cut": {"ref": ref, "length": length}})

    def paste(self, ref: Ref, start: int, length: int, change: Payload | None = None, **attrs: Any) -> Self:
        if length <= 0:
            return self
        new_op = {"paste": {"ref": ref, "start": start, "length": length}}
        if change is not None:
            if length != 1:
                raise ValueError("a paste change must address one embed")
            new_op["paste"]["change"] = change
        if attrs:
            new_op["attributes"] = attrs
        return self.push(new_op)

    def __getitem__(self, item: int | slice) -> Self:
        if isinstance(item, int):
            start = item
            stop = item + 1
        elif isinstance(item, slice):
            start = item.start or 0
            stop = item.stop
            if item.step is not None:
                raise ValueError("no support for step slices")
        else:
            raise TypeError("Invalid argument type.")

        if (start is not None and start < 0) or (stop is not None and stop < 0):
            raise ValueError("no support for negative indexing.")

        ops = []
        it = self.iterator()
        pos = 0
        while it.has_next():
            if stop is not None and pos >= stop:
                break
            if pos < start:
                next_op = it.next(start - pos)
            else:
                next_op = it.next(stop - pos if stop is not None else None)
                ops.append(next_op)
            pos += op.length(next_op)

        return self.__class__(ops)

    def push(self, operation: Op) -> Self:
        if self.ops:
            last_op = self.ops[-1]
            if op.type(operation) == op.type(last_op) == "paste":
                last, new = last_op["paste"], operation["paste"]
                if (
                    last["ref"] == new["ref"]
                    and last["start"] + last["length"] == new["start"]
                    and "change" not in last
                    and "change" not in new
                    and last.get("path") == new.get("path")
                    and last.get("unit") == new.get("unit")
                    and last_op.get("attributes") == operation.get("attributes")
                ):
                    last["length"] += new["length"]
                    return self
            if op.type(operation) == op.type(last_op) == "cut" and last_op["cut"]["ref"] == operation["cut"]["ref"]:
                last_op["cut"]["length"] += operation["cut"]["length"]
                return self

        index = len(self.ops)
        new_op = op.clone(operation)
        if index == 0:
            self.ops.append(new_op)
            return self

        last_op = self.ops[index - 1]

        if op.type(new_op) == op.type(last_op) == "delete":
            last_op["delete"] += new_op["delete"]
            return self

        if op.type(last_op) == "delete" and op.type(new_op) == "insert":
            index -= 1
            if index == 0:
                self.ops.insert(0, new_op)
                return self
            last_op = self.ops[index - 1]

        if new_op.get("attributes") == last_op.get("attributes"):
            if isinstance(new_op.get("insert"), str) and isinstance(last_op.get("insert"), str):
                last_op["insert"] = op.str_join(last_op["insert"], new_op["insert"])
                return self

            if isinstance(new_op.get("retain"), (int, float)) and isinstance(last_op.get("retain"), (int, float)):
                last_op["retain"] += new_op["retain"]
                if isinstance(new_op.get("attributes"), dict):
                    last_op["attributes"] = new_op["attributes"]
                return self

        self.ops.insert(index, new_op)
        return self

    def change_length(self) -> int:
        length = 0
        for operation in self:
            match op.type(operation):
                case "delete":
                    length -= operation["delete"]
                case "cut":
                    length -= operation["cut"]["length"]
                case "insert":
                    length += op.length(operation)
                case "paste":
                    length += operation["paste"]["length"]
        return length

    def compose(self, other: Self, *, _context: ComposeState | None = None) -> Self:
        context = _checked_context(_context, ComposeState, "compose")
        if context is not None:  # a child sequence joining the transaction
            return self._compose_with_moves(list(other.ops), context, nested=True)
        check(self)
        check(other)
        return _sourced(self._compose_transaction(_renamed(_refs(self.ops), other.ops)))

    def _compose_unchecked(self, other: Self) -> Self:
        """Compose an internally generated fragment whose move halves may
        deliberately share refs with ``self`` and span the two operands."""
        return self._compose_transaction(list(other.ops))

    def _compose_transaction(self, other_ops: Ops) -> Self:
        refs = _refs(self.ops) | _refs(other_ops)
        other_cuts = {
            operation["cut"]["ref"]
            for operation in _walk_move_ops(list(other_ops))
            if isinstance(operation.get("cut"), dict)
        }
        nested = _nested_paste_refs(self.ops) | _nested_paste_refs(other_ops)
        tables = {}
        trash = {}
        while True:
            shared = ComposeState(tables=tables, taken=set(refs), cuts=other_cuts, trash=trash, nested_pastes=nested)
            result = self._compose_with_moves(other_ops, shared, nested=False)
            if not shared.retry:
                return _normalize_orphans(_retarget_trashed(result, trash))
            # A child paste preceded its cut's consumption; rerun once
            # with the now-complete capture tables.
            tables = dict(shared.tables)
            other_cuts = set()

    def _compose_with_moves(self, other_ops: Ops, shared: ComposeState, nested: bool) -> Self:
        self_it = op.iterator(self.ops)
        other_it = op.iterator(other_ops)
        out = []  # ops, with pastes from `other` still unexpanded
        tables = shared.tables
        while self_it.has_next() or other_it.has_next():
            if not other_it.has_next():
                # the rest of self passes through unchanged
                out.extend(self_it.rest())
                break
            other_type = other_it.peek_type()
            if other_type in ("insert", "paste"):
                out.append(other_it.next())
                continue
            if self_it.peek_type() in ("delete", "cut"):
                out.append(self_it.next())
                continue
            if other_type == "cut":
                _consume_cut(other_it.next()["cut"], self_it, out, shared, nested)
                continue
            length = min(self_it.peek_length(), other_it.peek_length())
            self_op = self_it.next(length)
            other_op = other_it.next(length)
            if other_op.get("retain") is not None:
                out.append(_composed_retain(self_op, other_op, length, shared))
            elif op.type(other_op) == "delete" and op.type(self_op) == "paste":
                # the delete cancels a pasted window; cuts riding its
                # change still source windows elsewhere — they read on
                # through the window's own capture, trash-bin style
                spec = self_op["paste"]
                if isinstance(spec.get("change"), dict):
                    for ref, path, offset, _size in _payload_cut_sites(spec["change"]):
                        shared.trash[ref] = {
                            "ref": spec["ref"],
                            "unit": spec["start"],
                            "path": list(path),
                            "offset": offset,
                        }
            elif op.type(other_op) == "delete" and isinstance(self_op.get("retain"), (int, float, dict)):
                if isinstance(self_op.get("retain"), dict) and _trash_embed(self_op["retain"], shared, out):
                    continue  # deletion of a sourcing embed became a trash cut
                out.append(other_op)

        def expand(operation):
            for carrier in ("insert", "retain"):
                # newly inserted embeds and typed retain payloads may
                # both carry paste windows in their child sequences;
                # expand them against the same tables.  Windows a child
                # handler already expanded reference refs outside the
                # tables and pass through untouched.
                if isinstance(operation, dict) and isinstance(operation.get(carrier), dict):
                    payload = _expanded_payload(operation[carrier], tables, shared)
                    if payload is not None:
                        return [{**operation, carrier: payload}]
                    return None
            if op.type(operation) != "paste":
                return None
            spec = operation["paste"]
            rewritten = None
            if isinstance(spec.get("change"), dict):
                # a window change is a payload too: pastes riding it
                # (a move into an embed that itself moved) expand alike
                change = _expanded_payload(spec["change"], tables, shared)
                if change is not None:
                    operation = {**operation, "paste": {**spec, "change": change}}
                    rewritten = [operation]
            ref = operation["paste"]["ref"]
            if ref in tables:
                return _expand(operation, tables, shared)
            if ref in shared.cuts:
                shared.retry = True  # cut consumed later in the walk
            return rewritten

        return _assemble(out, expand, self.__class__)

    def transform(
        self, other: Self | int | float, priority: bool = False, *, _context: TransformState | None = None
    ) -> Self | int | float:
        context = _checked_context(_context, TransformState, "transform")
        if isinstance(other, (int, float)):
            return self.transform_position(other, priority)
        if context is not None:  # a child sequence joining the transaction
            return self._transform_with_moves(other, priority, context)
        check(self)
        check(other)
        return _normalize_orphans(self._transform_transaction(other, priority))

    def _transform_transaction(self, other: Self, priority: bool) -> Self:
        other_reads = {}
        for operation in other.ops:
            if op.type(operation) == "paste" and "path" in operation["paste"]:
                spec = operation["paste"]
                other_reads.setdefault(spec["ref"], []).append(
                    {
                        "unit": spec.get("unit", 0),
                        "path": tuple(spec["path"]),
                        "start": spec["start"],
                        "length": spec["length"],
                        "attributes": operation.get("attributes"),
                    }
                )
        shared = TransformState(
            self_windows=_collect_windows(list(self.ops)),
            other_windows=_collect_windows(list(other.ops)),
            other_reads=other_reads,
            taken=_refs(other.ops),
        )
        # Renumbering and overlays may transform transaction fragments
        # through handlers, which receive this same explicit state.
        out = self._transform_with_moves(other, priority, shared, raw=True)
        out = _renumber_list(out, shared.state, priority, shared)
        result = _expand_markers(
            out, shared.buckets, shared.read_buckets, self.__class__, shared.state, priority, shared
        )
        return _apply_overlays(result, shared, priority)

    def _transform_with_moves(self, other: Self, priority: bool, shared: TransformState, raw: bool = False):
        """Transform ``other`` against ``self`` when either contains moves.

        Our moves route the other side's edits — including its cuts, which
        get rebased onto our paste windows when they lose the contested
        content — and the other side's moves shrink, split and renumber
        around our edits.  Windows and rebasing state are shared across
        the transaction so child sequences participate.  With ``raw`` the
        owner receives the unexpanded item list to renumber first.
        """
        state = shared.state
        local_pastes = {(o["paste"]["ref"], o["paste"]["start"]) for o in self.ops if op.type(o) == "paste"}
        window_index = {
            (ref, span.start): index for ref, spans in shared.self_windows.items() for index, span in enumerate(spans)
        }
        local_reads = {}  # our trash reads, routable like windows
        read_buckets = {}
        for operation in self.ops:
            if op.type(operation) == "paste" and "path" in operation["paste"]:
                spec = operation["paste"]
                key = (spec["ref"], spec.get("unit", 0), tuple(spec["path"]), spec["start"])
                local_reads.setdefault(spec["ref"], []).append(
                    {
                        "unit": spec.get("unit", 0),
                        "path": tuple(spec["path"]),
                        "start": spec["start"],
                        "length": spec["length"],
                        "attributes": operation.get("attributes"),
                        "key": key,
                    }
                )
                read_buckets[key] = []

        self_it = op.iterator(self.ops)
        other_it = op.iterator(other.ops)
        out = []  # ops, window/rewrite markers, unrenumbered pastes
        while self_it.has_next() or other_it.has_next():
            self_type = self_it.peek_type()
            other_type = other_it.peek_type()
            if self_type in ("insert", "paste") and (priority or other_type not in ("insert", "paste")):
                if self_type == "insert":
                    out.append({"retain": op.length(self_it.next())})
                    continue
                spec = self_it.next()["paste"]
                if "path" in spec:  # a trash read hosts routed edits too
                    out.append(
                        ("read", (spec["ref"], spec.get("unit", 0), tuple(spec["path"]), spec["start"]), spec["length"])
                    )
                else:
                    out.append(("window", spec["ref"], window_index[(spec["ref"], spec["start"])], spec["length"]))
                continue
            if other_type in ("insert", "paste"):
                out.append(other_it.next())
                continue
            if self_type == "cut":
                _route_cut(
                    self_it.next()["cut"], other_it, shared, out, priority, local_pastes, local_reads, read_buckets
                )
                continue
            length = min(self_it.peek_length(), other_it.peek_length())
            self_op = self_it.next(length)
            other_op = other_it.next(length)
            if op.type(other_op) == "cut":
                _cut_piece(
                    out,
                    shared,
                    other_op["cut"],
                    deleted=bool(self_op.get("delete")),
                    attributes=self_op.get("attributes"),
                    change=self_op["retain"] if isinstance(self_op.get("retain"), dict) else None,
                )
            elif self_op.get("delete"):
                if isinstance(other_op.get("retain"), dict):
                    _drop_other_embed(other_op["retain"], state)
                continue
            elif other_op.get("delete"):
                if isinstance(self_op.get("retain"), dict):
                    _drop_self_embed(self_op["retain"], shared)
                out.append(other_op)
            else:
                out.append(_transformed_retain(self_op, other_op, length, priority, shared))

        if raw:
            shared.read_buckets = read_buckets
            return out
        return _expand_markers(out, shared.buckets, read_buckets, self.__class__)

    def transform_position(self, index: int, priority: bool = False) -> int:
        """Map a position through this delta.

        A position strictly inside moved content follows it to the
        covering paste window; a position at the region's start, or over
        content that is cut but never pasted, stays at the source like a
        deletion.
        """
        input_position = 0
        followed = None
        for operation in self.ops:
            if op.type(operation) == "cut":
                offset = index - input_position
                if 0 < offset < operation["cut"]["length"]:
                    followed = (operation["cut"]["ref"], offset)
            input_position += op.input_length(operation)
        if followed is not None:
            ref, offset = followed
            output_position = 0
            for operation in self.ops:
                if op.type(operation) == "paste" and operation["paste"]["ref"] == ref:
                    spec = operation["paste"]
                    if spec["start"] <= offset < spec["start"] + spec["length"]:
                        return output_position + offset - spec["start"]
                output_position += op.output_length(operation)
        position = index
        passed = 0
        for operation in self.ops:
            if passed > position:
                break
            kind = op.type(operation)
            length = op.length(operation)
            if kind in ("delete", "cut"):
                position -= min(length, position - passed)
            elif kind in ("insert", "paste"):
                if passed < position or not priority:
                    position += length
                passed += length
            else:
                passed += length
        return position

    def lower(self, base: Self) -> Self:
        """Rewrite moves as plain deletes, inserts and embed changes
        against a concrete document, at every nesting level."""
        check(self)
        captures = {}
        _collect_captures(list(self.ops), MoveDelta(list(base.ops)), captures)
        return _lower_ops(list(self.ops), captures)

    def invert(self, base: Self, *, _context: InvertState | None = None) -> Self:
        """Invert against ``base``; moves invert semantically.

        Each paste window becomes a cut of the pasted span, and the
        original cut position pastes those spans back in source order,
        restoring never-pasted gaps from ``base`` and reverting any
        attribute patches the pastes applied.
        """
        context = _checked_context(_context, InvertState, "invert")
        if context is None:
            check(self)
            windows = {}  # ref -> [(start, length, attributes, change)]
            # windows riding inserted payloads vanish with the insert's
            # inverse delete, so the cut restores their spans from base
            for operation in _walk_move_ops(list(self.ops), skip_inserts=True):
                spec = operation.get("paste")
                if isinstance(spec, dict) and "path" not in spec:
                    windows.setdefault(spec["ref"], []).append(
                        Window(spec["start"], spec["length"], operation.get("attributes"), spec.get("change"))
                    )
            inverse_refs = {}
            taken = set()
            for ref in sorted(windows):
                windows[ref].sort(key=lambda span: span.start)
                for index, window in enumerate(windows[ref]):
                    inverse_refs[(ref, window.start)] = _fresh_ref(ref if index == 0 else f"{ref}:{index}", taken)
            context = InvertState(windows=windows, inverse_refs=inverse_refs)
        return self._invert_with_moves(base, context)

    def _invert_with_moves(self, base: Self, shared: InvertState) -> Self:
        windows = shared.windows
        inverse_refs = shared.inverse_refs
        inverted = self.__class__()
        base_index = 0
        for operator in self.ops:
            kind = op.type(operator)
            if kind == "insert":
                inverted.delete(op.length(operator))
            elif kind == "paste":
                spec = operator["paste"]
                if "path" in spec:
                    # content read through a trashed embed: the inverse
                    # restores the embed whole, so this copy just dies
                    inverted.delete(spec["length"])
                else:
                    inverted.push(
                        {
                            "cut": {
                                "ref": inverse_refs[(spec["ref"], spec["start"])],
                                "length": spec["length"],
                            }
                        }
                    )
            elif kind == "cut":
                spec = operator["cut"]
                position = 0
                for window in windows.get(spec["ref"], []):
                    start, length = window.start, window.length
                    attributes, change = window.attributes, window.change
                    for base_op in base[base_index + position : base_index + start]:
                        inverted.push(base_op)  # dropped gap: restore content
                    inverse_ref = inverse_refs[(spec["ref"], start)]
                    offset = 0
                    for base_op in base[base_index + start : base_index + start + length]:
                        piece_length = op.length(base_op)
                        piece_spec = {"ref": inverse_ref, "start": offset, "length": piece_length}
                        if change is not None:
                            revert_change = _invert_embed_change(change, base_op.get("insert"), shared)
                            if revert_change is not None:
                                piece_spec["change"] = revert_change
                        piece = {"paste": piece_spec}
                        revert = attributes and op.invert(attributes, base_op.get("attributes"))
                        if revert:
                            piece["attributes"] = revert
                        inverted.push(piece)
                        offset += piece_length
                    position = start + length
                for base_op in base[base_index + position : base_index + spec["length"]]:
                    inverted.push(base_op)
                base_index += spec["length"]
            elif isinstance(operator.get("retain"), (int, float)) and operator.get("attributes") is None:
                inverted.retain(operator["retain"])
                base_index += operator["retain"]
            elif kind == "delete" or isinstance(operator.get("retain"), (int, float)):
                length = int(operator.get("delete") or operator.get("retain"))
                for base_op in base[base_index : base_index + length]:
                    if kind == "delete":
                        inverted.push(base_op)
                    else:
                        inverted.retain(
                            op.length(base_op),
                            **(op.invert(operator.get("attributes"), base_op.get("attributes")) or {}),
                        )
                base_index += length
            elif isinstance(operator.get("retain"), dict):
                base_op = op.iterator(base[base_index].ops).next()
                embed_type, op_data, base_op_data = get_embed_type_and_data(operator["retain"], base_op.get("insert"))
                handler = _get_handler(embed_type)
                inverted.retain(
                    {embed_type: handler.invert(op_data, base_op_data, shared)},
                    **(op.invert(operator.get("attributes"), base_op.get("attributes")) or {}),
                )
                base_index += 1
        return inverted.chop()


# ── Functional reference API and existing-suite adapter ────────────────────

Delta = MoveDelta


def normalize(delta: MoveDelta | Sequence[Op]) -> MoveDelta:
    """Return canonical builder form without applying a document."""

    operations = delta.ops if hasattr(delta, "ops") else delta
    result = MoveDelta()
    for operation in copy.deepcopy(list(operations)):
        result.push(operation)
    return result.chop()


def validate(delta: MoveDelta | Sequence[Op]) -> MoveDelta:
    """Validate every move ref transaction-wide and return the Delta."""

    candidate = delta if isinstance(delta, MoveDelta) else MoveDelta(list(delta))
    check(candidate)
    return candidate


def apply(document: MoveDelta, delta: MoveDelta) -> MoveDelta:
    """Apply ``delta`` to an insert-only document."""

    return MoveDelta._from_owned_ops(copy.deepcopy(document.ops)).compose(
        MoveDelta._from_owned_ops(copy.deepcopy(delta.ops))
    )


def compose(first: MoveDelta, second: MoveDelta) -> MoveDelta:
    return MoveDelta._from_owned_ops(copy.deepcopy(first.ops)).compose(
        MoveDelta._from_owned_ops(copy.deepcopy(second.ops))
    )


def transform(first: MoveDelta, second: MoveDelta, priority: bool = False) -> MoveDelta:
    """Return ``first_after_second`` using the task statement's direction.

    The class method follows quill-delta's historical convention
    ``applied.transform(other) == other_after_applied``.  This functional
    entry point presents the construction's requested direction.
    """

    return MoveDelta._from_owned_ops(copy.deepcopy(second.ops)).transform(
        MoveDelta._from_owned_ops(copy.deepcopy(first.ops)), not priority
    )


def install_as_package_delta() -> None:
    """Redirect the existing repository tests to this standalone module."""

    import delta as package
    import delta.base as package_base
    import delta.coords as coordinates
    import delta.diff as snapshot

    package.Delta = MoveDelta
    package.MoveDelta = MoveDelta

    # Explicit validation helpers remain available from the production
    # module while this standalone class replaces the package surface.
    package_base.check = check
    package_base.has_moves = has_moves

    # coords and diff cached their imports when delta's package initialized.
    coordinates.Delta = MoveDelta
    coordinates._collect_windows = _collect_windows
    coordinates._unit_patch = _unit_patch
    coordinates._unit_position = _unit_position
    snapshot.handlers = handlers
    snapshot.walk_move_ops = _walk_move_ops


def run_pytest(arguments: Sequence[str] = ()) -> int:
    """Run the repository's unchanged suite against this implementation."""

    import pytest

    install_as_package_delta()
    return int(pytest.main(list(arguments) or ["-q"]))


def main(arguments: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if arguments is None else arguments)
    if args and args[0] == "--pytest":
        return run_pytest(args[1:])
    print("usage: python move-ot-reference.py --pytest [pytest arguments]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
