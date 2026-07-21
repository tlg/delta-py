import copy
from collections.abc import Callable, Generator, Iterator
from typing import Any, Literal, Self, TypedDict, TypeIs, overload, override

from . import op
from .embed import EmbedHandler, handlers, map_streams, streams as child_streams, walk_move_ops as walk_moves
from .op import Attributes, Number, Op, Ops, Path, Payload, Ref


type StructuralValue = str | Number | Ops | list[StructuralValue] | dict[str, StructuralValue] | None
type Inventory = tuple[set[Ref], set[Ref], set[Ref]]
type PasteWindow = tuple[int, int, Attributes | None, Payload | None]
type PasteWindows = dict[Ref, list[PasteWindow]]
type RenamedPart = tuple[int, int, Ref]


class TrashSite(TypedDict):
    ref: Ref
    unit: int
    path: Path
    offset: int


class Claim(TypedDict):
    stream: "Delta"
    parts: int
    renamed: list[RenamedPart]


class ComposeContext(TypedDict):
    _kind: Literal["compose"]
    streams: dict[Ref, "Delta"]
    retry: bool
    taken: set[Ref]
    claimed: set[Ref]
    cuts: set[Ref]
    trash: dict[Ref, TrashSite]
    nested_pastes: set[Ref]


class TransformContext(TypedDict):
    _kind: Literal["transform"]
    streams: dict[Ref, "Delta"]
    claims: dict[Ref, Claim]
    taken: set[Ref]
    doomed: set[Ref]
    retry: bool
    self_cuts: set[Ref]
    self_windows: PasteWindows
    other_windows: PasteWindows


class InvertContext(TypedDict):
    _kind: Literal["invert"]
    windows: PasteWindows
    names: dict[tuple[Ref, int], Ref]


class DiffContext(TypedDict):
    _kind: Literal["diff"]


type HandlerContext = ComposeContext | TransformContext | InvertContext | DiffContext
type Handler = EmbedHandler[Any, Any, HandlerContext]
type ContextKind = Literal["compose", "transform", "invert", "diff"]


NULL_CHARACTER = chr(0)


def _is_dictionary(value: StructuralValue) -> TypeIs[dict[str, StructuralValue]]:
    return isinstance(value, dict)


def _is_operation_stream(value: StructuralValue) -> TypeIs[Ops]:
    return isinstance(value, list)


class _PendingRef(str):
    """Private ref tag, unequal to its otherwise identical public ref."""

    __slots__ = ()
    __hash__ = str.__hash__

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is _PendingRef and super().__eq__(other)

    @override
    def __ne__(self, other: object) -> bool:
        return not self == other


def pending_ref(operation: Op) -> Ref | None:
    """The original ref when ``operation`` is a pending paste marker."""
    paste = operation.get("paste")
    if paste is not None and isinstance(ref := paste["ref"], _PendingRef):
        return str(ref)
    return None


def checked_context[Context: HandlerContext](context: Context | None, kind: ContextKind) -> Context | None:
    if context is not None and context.get("_kind") != kind:
        raise TypeError(f"{kind} received an invalid handler context")
    return context


def move_inventory(ops: Ops, find_nested: bool = False) -> Inventory:
    """Collect cut/paste refs and optionally paste refs below the stream root."""
    cuts: set[Ref] = set()
    pastes: set[Ref] = set()
    nested: set[Ref] = set()

    def collect(operation: Op, below_root: bool) -> None:
        if (spec := operation.get("cut")) is not None:
            cuts.add(spec["ref"])
        if (spec := operation.get("paste")) is not None:
            pastes.add(spec["ref"])
            if below_root:
                nested.add(spec["ref"])

    for operation in ops:
        collect(operation, False)
        for payload in (operation.get("insert"), operation.get("retain")):
            for nested_op in walk_moves(payload):
                collect(nested_op, find_nested)
        spec = operation.get("paste")
        for nested_op in walk_moves(spec.get("change") if spec is not None else None):
            collect(nested_op, find_nested)
    return cuts, pastes, nested


def paste_windows(ops: Ops, skip_inserts: bool = False) -> PasteWindows:
    """Collect every ordinary paste window across nested sequences."""
    windows: PasteWindows = {}
    for operation in walk_moves(ops, skip_inserts):
        spec = operation.get("paste")
        if spec is not None and "path" not in spec:
            windows.setdefault(spec["ref"], []).append(
                (spec["start"], spec["length"], operation.get("attributes"), spec.get("change"))
            )
    for spans in windows.values():
        spans.sort(key=lambda span: span[0])
    return windows


def read_spans(edit_ops: Ops, start: int, end: int) -> Generator[tuple[int, int]]:
    """Output runs containing retained images of input [start, end)."""
    input_position = output_position = 0
    for operation in edit_ops:
        kind = op.type(operation)
        input_length = int(op.input_length(operation))
        output_length = int(op.output_length(operation))
        if kind == "retain":
            low = max(input_position, start)
            high = min(input_position + input_length, end)
            if low < high:
                yield output_position + low - input_position, high - low
        elif kind == "insert" and start < input_position < end:
            yield output_position, output_length
        input_position += input_length
        output_position += output_length
    if input_position < end:  # implicit tail retain
        low = max(input_position, start)
        yield output_position + low - input_position, end - low


def input_effect(edit_ops: Ops, start: int, end: int) -> "Delta":
    """The part of an edit consuming ``[start, end)``; strict-interior
    zero-input ops follow the copied range, while boundary ops stay outside."""
    result = Delta()
    position = 0
    for operation in edit_ops:
        kind = op.type(operation)
        if kind in ("insert", "paste"):
            if start < position < end:
                result.push(operation)
            continue
        length = int(op.length(operation))
        low, high = max(position, start), min(position + length, end)
        if low < high:
            for piece in op.sliced([operation], low - position, high - low):
                result.push(piece)
        position += length
    if position < end:
        result.retain(end - max(position, start))
    return result


def renamed_refs(first_refs: set[Ref], ops: Ops) -> tuple[Ops, Inventory]:
    """Alpha-rename refs in the second operand that collide with the first."""
    inventory = move_inventory(ops, find_nested=True)
    collisions = (inventory[0] | inventory[1]) & first_refs
    if not collisions:
        return list(ops), inventory
    taken = first_refs | inventory[0] | inventory[1]
    names: dict[Ref, Ref] = {}
    for ref in sorted(collisions):
        index = 2
        while f"{ref}~{index}" in taken:
            index += 1
        names[ref] = f"{ref}~{index}"
        taken.add(names[ref])
    result = copy.deepcopy(list(ops))
    for operation in walk_moves(result):
        for spec in (operation.get("cut"), operation.get("paste")):
            if spec is not None and spec["ref"] in names:
                spec["ref"] = names[spec["ref"]]
    renamed = tuple({names.get(ref, ref) for ref in refs} for refs in inventory)
    return result, (renamed[0], renamed[1], renamed[2])


def payload_cut_sites(payload: Payload) -> Generator[tuple[Ref, Path, int]]:
    """(ref, path, offset) for cuts in handler-declared child streams."""
    for path, sequence in child_streams(payload):
        offset = 0
        for item in sequence:
            spec = item.get("cut")
            if spec is not None:
                yield spec["ref"], list(path), offset
            retained = item.get("retain")
            if isinstance(retained, dict):
                for ref, inner_path, inner_offset in payload_cut_sites(retained):
                    yield ref, [*path, offset, *inner_path], inner_offset
            offset += int(op.input_length(item))


def navigate_stream(value: StructuralValue, path: Path, output: bool) -> Ops | None:
    """Follow payload keys and cumulative stream-unit offsets."""
    measure = op.output_length if output else op.input_length
    for step in path:
        if isinstance(step, int):
            if not _is_operation_stream(value):
                return None
            position = 0
            for item in value:
                length = measure(item)
                if position <= step < position + length:
                    value = item.get("insert") if output else item.get("retain")
                    break
                position += length
            else:
                return None
            if not _is_dictionary(value) or len(value) != 1:
                return None
            value = next(iter(value.values()))
        else:
            if not _is_dictionary(value) or step not in value:
                return None
            value = value[step]
    return value if _is_operation_stream(value) else None


def fresh_ref(name: Ref, taken: set[Ref]) -> Ref:
    """``name``, suffixed until it dodges every ref in ``taken`` — every
    generated ref must go through this or a sibling guard: each naming
    scheme in this module has grown the same collision bug once."""
    while name in taken:
        name = f"{name}:1"
    taken.add(name)
    return name


def embed_payload(value: op.InsertValue | op.RetainValue | None) -> Payload:
    if not isinstance(value, dict):
        raise TypeError(f"cannot retain a {type(value).__name__}")
    return value


def get_embed_type_and_data(a: Payload, b: Payload) -> tuple[str, Any, Any]:
    if len(a) != 1 or len(b) != 1:
        raise ValueError("embed values must contain exactly one type")

    (embed_type, left), (b_type, right) = next(iter(a.items())), next(iter(b.items()))
    if not embed_type or embed_type != b_type:
        raise ValueError(f"embed types not matched: {embed_type} != {b_type}")
    return embed_type, left, right


def compose_change(existing: Payload | None, patch: Payload, context: ComposeContext) -> Payload:
    """Compose two embed changes riding a paste window."""
    if existing is None:
        return patch
    embed_type, a, b = get_embed_type_and_data(existing, patch)
    handler = Delta.get_handler(embed_type)
    return {embed_type: handler.compose(a, b, context)}


def apply_change(insert: Payload, change: Payload, context: ComposeContext) -> Payload:
    """Apply an embed change riding a paste window to the embed insert
    it moves."""
    embed_type, data, patch = get_embed_type_and_data(insert, change)
    return {embed_type: Delta.get_handler(embed_type).apply(data, patch, context)}


def transform_change(
    applied: Payload | None, other: Payload | None, priority: bool, context: TransformContext
) -> Payload | None:
    """Transform an embed change against a concurrently applied one;
    None when nothing of it survives."""
    if applied is None or other is None:
        return other
    embed_type, a, b = get_embed_type_and_data(applied, other)
    payload = Delta.get_handler(embed_type).transform(a, b, priority, context)
    return {embed_type: payload} if payload else None


def invert_change(change: Payload, base_insert: Payload, context: InvertContext) -> Payload | None:
    """Invert an embed change riding a paste window against the embed
    it patched; None when there is nothing to revert."""
    embed_type, data, based = get_embed_type_and_data(change, base_insert)
    payload = Delta.get_handler(embed_type).invert(data, based, context)
    return {embed_type: payload} if payload else None


def normalize_orphans[DeltaT: Delta](delta: DeltaT) -> DeltaT:
    """Turn cut fragments with no recursive paste consumer into deletes."""
    cuts, pastes, _nested = move_inventory(delta.ops)
    if pastes - cuts:
        raise ValueError("cannot compose the deletion of an embed that still sources a move")
    orphaned = cuts - pastes
    if not orphaned:
        return delta

    def remove(item: Op, out: DeltaT) -> bool:
        spec = item.get("cut")
        if spec is not None and spec["ref"] in orphaned:
            out.push({"delete": spec["length"]})
            return True
        return False

    normalized = rewrite_nested(delta.ops, delta.__class__, remove)
    return delta if normalized is None else delta.__class__(normalized).chop()


def retarget_trashed[DeltaT: Delta](delta: DeltaT, trash: dict[Ref, TrashSite]) -> DeltaT:
    """Route pastes through deleted source embeds once surviving cuts are known."""
    if not trash:
        return delta
    cuts, _pastes, _nested = move_inventory(delta.ops)

    def retarget(item: Op, out: DeltaT) -> bool:
        spec = item.get("paste")
        if spec is None or "path" in spec:
            return False
        site = trash.get(spec["ref"])
        if site is None or spec["ref"] in cuts:
            return False
        paste = spec.copy()
        paste.update(ref=site["ref"], unit=site["unit"], path=site["path"], start=site["offset"] + spec["start"])
        out.push(op.replace_paste(item, paste))
        return True

    rewritten = rewrite_nested(delta.ops, delta.__class__, retarget)
    return delta if rewritten is None else delta.__class__(rewritten).chop()


@overload
def rewrite_nested[DeltaT: Delta](
    value: Ops, delta_type: type[DeltaT], visit: Callable[[Op, DeltaT], bool]
) -> Ops | None: ...


@overload
def rewrite_nested[DeltaT: Delta](
    value: Op, delta_type: type[DeltaT], visit: Callable[[Op, DeltaT], bool]
) -> Op | None: ...


def rewrite_nested[DeltaT: Delta](
    value: Ops | Op, delta_type: type[DeltaT], visit: Callable[[Op, DeltaT], bool]
) -> Ops | Op | None:
    """Copy-on-change traversal of handler-declared operation streams."""

    def rewrite_payload(payload: Payload) -> Payload | None:
        return map_streams(payload, rewrite_stream)

    def rewrite_operation(item: Op) -> Op | None:
        inserted, retained = item.get("insert"), item.get("retain")
        replacement = rewrite_payload(inserted) if isinstance(inserted, dict) else None
        if replacement is not None:
            return op.replace_insert(item, replacement)
        replacement = rewrite_payload(retained) if isinstance(retained, dict) else None
        if replacement is not None:
            return op.replace_retain(item, replacement)
        spec = item.get("paste")
        change = spec.get("change") if spec is not None else None
        replacement = rewrite_payload(change) if isinstance(change, dict) else None
        if replacement is not None and spec is not None:
            paste = spec.copy()
            paste["change"] = replacement
            return op.replace_paste(item, paste)
        return None

    def rewrite_stream(sequence: Ops) -> Ops | None:
        items: Ops = []
        changed = False
        for item in sequence:
            emitted = delta_type()
            if visit(item, emitted):
                items.extend(emitted.ops)
                changed = True
                continue
            replacement = rewrite_operation(item)
            changed = changed or replacement is not None
            item = item if replacement is None else replacement
            items.append(item)
        if not changed:
            return None
        rebuilt = delta_type()
        for item in items:
            rebuilt.push(item)
        return rebuilt.ops

    return rewrite_stream(value) if isinstance(value, list) else rewrite_operation(value)


class Delta:
    @staticmethod
    def register_embed(embed_type: str, handler: Handler) -> None:
        handlers[embed_type] = handler

    @staticmethod
    def unregister_embed(embed_type: str) -> None:
        handlers.pop(embed_type, None)

    @staticmethod
    def get_handler(embed_type: str) -> Handler:
        handler = handlers.get(embed_type)
        if handler is None:
            raise ValueError(f'no handlers for embed type "{embed_type}"')
        return handler

    @classmethod
    def _from_owned_ops(cls, ops: Ops, **attrs: Any) -> Self:
        """Internal ownership transfer: ``ops`` must not be reused."""
        delta = cls(**attrs)
        delta.ops = ops
        return delta

    def __init__(self, ops: Ops | Self | None = None, **attrs: Any) -> None:
        ops = getattr(ops, "ops", ops)
        self.ops = [op.clone(operation) for operation in ops] if ops else []
        self.__dict__.update(attrs)

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Delta):
            return NotImplemented
        return self.ops == other.ops

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.ops})"

    def cut(self, ref: Ref, length: int) -> Self:
        if length <= 0:
            return self
        return self.push({"cut": {"ref": ref, "length": length}})

    def paste(
        self,
        ref: Ref,
        start: int,
        length: int,
        change: Payload | None = None,
        **attrs: Any,
    ) -> Self:
        if length <= 0:
            return self
        new_op: op.PasteOp = {"paste": {"ref": ref, "start": start, "length": length}}
        if change is not None:
            if length != 1:
                raise ValueError("a paste change must address one embed")
            new_op["paste"]["change"] = change
        if attrs:
            new_op["attributes"] = attrs
        return self.push(new_op)

    def insert(self, text: op.InsertValue, **attrs: Any) -> Self:
        if text == "":
            return self
        new_op: op.InsertOp = {"insert": text}
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
        new_op: op.RetainOp = {"retain": length}
        if attrs:
            new_op["attributes"] = attrs
        return self.push(new_op)

    def push(self, operation: Op) -> Self:
        kind = op.type(operation)
        index = len(self.ops)
        last_op: Op | None = None
        last_kind = None
        if index:
            last_op = self.ops[-1]
            last_kind = op.type(last_op)
            if op.is_paste(last_op) and op.is_paste(operation):
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
            if op.is_cut(last_op) and op.is_cut(operation) and last_op["cut"]["ref"] == operation["cut"]["ref"]:
                last_op["cut"]["length"] += operation["cut"]["length"]
                return self
            if op.is_delete(last_op) and op.is_delete(operation):
                self.ops[-1] = op.replace_delete(last_op, last_op["delete"] + operation["delete"])
                return self

        new_op = op.clone(operation)
        if not index:
            self.ops.append(new_op)
            return self

        if last_op is not None and last_kind == "delete" and kind == "insert":
            index -= 1
            if index:
                last_op = self.ops[index - 1]

        if index and last_op is not None and new_op.get("attributes") == last_op.get("attributes"):
            inserted, last_inserted = new_op.get("insert"), last_op.get("insert")
            if isinstance(inserted, str) and isinstance(last_inserted, str):
                self.ops[index - 1] = op.replace_insert(last_op, op.str_join(last_inserted, inserted))
                return self

            retained, last_retained = new_op.get("retain"), last_op.get("retain")
            if isinstance(retained, (int, float)) and isinstance(last_retained, (int, float)):
                self.ops[index - 1] = op.replace_retain(last_op, last_retained + retained)
                return self

        self.ops.insert(index, new_op)
        return self

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
        parts: list[str] = []
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

    def __iter__(self) -> Iterator[Op]:
        return iter(self.ops)

    @overload
    def __getitem__(self, item: int) -> Self: ...

    @overload
    def __getitem__(self, item: slice[int | None, int | None, int | None]) -> Self: ...

    def __getitem__(self, item: int | slice[int | None, int | None, int | None]) -> Self:
        if isinstance(item, int):
            start = item
            stop = item + 1
        else:
            start = item.start or 0
            stop = item.stop
            if item.step is not None:
                raise ValueError("no support for step slices")

        if start < 0 or (stop is not None and stop < 0):
            raise ValueError("no support for negative indexing.")
        length = float("inf") if stop is None else stop - start
        return self.__class__(list(op.sliced(self.ops, start, length)))

    def __len__(self) -> int:
        return int(sum(op.length(o) for o in self.ops))

    def iterator(self) -> op.Iterator:
        return op.iterator(self.ops)

    def change_length(self) -> int:
        signs: dict[str, int] = {"delete": -1, "cut": -1, "insert": 1, "paste": 1}
        return int(sum(signs.get(op.type(operation), 0) * op.length(operation) for operation in self))

    def length(self) -> int | float:
        return sum(op.length(o) for o in self)

    def compose(self, other: Self, *, _context: ComposeContext | None = None) -> Self:
        context = checked_context(_context, "compose")
        if context is not None:  # a child sequence joining the transaction
            return self._compose_with(other, context, nested=True)
        self_refs, self_pastes, nested_pastes = move_inventory(self.ops, find_nested=True)
        self_refs.update(self_pastes)
        other_ops, (other_cuts, other_pastes, other_nested) = renamed_refs(self_refs, other.ops)
        other = self.__class__._from_owned_ops(other_ops)
        refs = self_refs
        refs.update(other_cuts, other_pastes)
        nested_pastes.update(other_nested)
        del self_pastes, other_pastes, other_nested
        # A window met before its cut lives in another sequence leaves
        # the first run unresolved; the rerun sees completed streams.
        streams: dict[Ref, Delta] = {}
        trash: dict[Ref, TrashSite] = {}
        for _ in range(2):
            # streams survive the rerun (a completed walk fills them
            # all); the taken names don't, so reruns name parts alike
            shared = ComposeContext(
                _kind="compose",
                streams=streams,
                retry=False,
                taken=set(refs),
                claimed=set(),
                cuts=other_cuts,
                trash=trash,
                nested_pastes=nested_pastes,
            )
            result = self._compose_with(other, shared, nested=False)
            if shared["retry"]:
                # Later siblings can settle a child request or preserve its move.
                result_cuts, result_pastes, _ = move_inventory(result.ops)
                shared["retry"] = bool((result_pastes & other_cuts) - result_cuts)
            if not shared["retry"]:
                break
        else:
            raise RuntimeError("compose remained unresolved after retry")
        return normalize_orphans(retarget_trashed(result, trash))

    def _compose_with(self, other: Self, shared: ComposeContext, nested: bool) -> Self:
        self_it = self.iterator()
        other_it = other.iterator()
        ops: Ops = []
        first_other = other_it.peek()

        # A cut in `other` riding over content this delta produced pushes
        # the captured units, in order, onto a stream keyed by the cut's
        # ref: inserts and pastes are carried verbatim, retained base
        # units become windows onto the composed cut.  Every paste of
        # `other` stays in the output as a pending marker and consumes
        # its slice of the stream through the op iterator at the end.
        # The streams are transaction-wide: a cut in one sequence feeds
        # windows in any other.
        refs_map = shared["streams"]
        # taken names are transaction-wide: sequences must not mint
        # colliding part names for each other's refs
        used_refs = shared["taken"]
        nested_pastes = shared["nested_pastes"]
        trashed = shared["trash"]

        def fresh_name(ref: Ref, force: bool = False) -> Ref:
            if not force and ref not in shared["claimed"]:
                shared["claimed"].add(ref)
                used_refs.add(ref)
                return ref
            count = 1
            while f"{ref}:{count}" in used_refs:
                count += 1
            name = f"{ref}:{count}"
            used_refs.add(name)
            return name

        def pass_self_cut() -> None:
            # an earlier cut of self passes through whole; the iterator
            # stamps a start offset on it that the wire form doesn't have
            whole = self_it.next()
            if op.is_cut(whole):
                whole["cut"].pop("start", None)
            delta.push(whole)

        def consume_cut(spec: op.CutSpec) -> None:
            # Walk self under the whole span of a cut of `other`,
            # streaming what the span was made of: base units become the
            # composed cut (windowed for the paste sites, deleted spans
            # absorbed), content this delta produced is carried verbatim.
            # the stream is itself a delta: push gives captures the same
            # normalization as output ops (adjacent windows merge, split
            # text re-joins).  Assignment, not setdefault: a transaction
            # rerun consumes every cut afresh
            stream = refs_map[spec["ref"]] = self.__class__()
            remaining = spec["length"]
            part: op.CutOp | None = None  # the open composed cut, extended in place

            def cut_part() -> op.CutSpec:
                nonlocal part
                if part is None:
                    part = {
                        "cut": {
                            "ref": fresh_name(spec["ref"], force=(nested or spec["ref"] in nested_pastes)),
                            "length": 0,
                        }
                    }
                    delta.ops.append(part)
                return part["cut"]

            while remaining > 0:
                self_type = self_it.peek_type()
                if self_type == "delete":
                    # cut but never pasted: absorbed, the windows skip it
                    cut_part()["length"] += int(op.length(self_it.next()))
                    continue
                if self_type == "cut":
                    pass_self_cut()  # an earlier cut splits this one
                    part = None
                    continue
                length = int(min(remaining, self_it.peek_length()))
                piece = op.take(self_it, length)
                retained = piece.get("retain")
                if isinstance(retained, (int, float, dict)):
                    # a retained base unit is re-typed to its moved
                    # mirror: retain -> window onto the composed cut,
                    # embed patch -> the window's change, attributes ride
                    cut = cut_part()
                    window: op.PasteOp = {"paste": {"ref": cut["ref"], "start": cut["length"], "length": length}}
                    if isinstance(retained, dict):
                        for ref, path, offset in payload_cut_sites(retained):
                            trashed[ref] = {
                                "ref": cut["ref"],
                                "unit": cut["length"],
                                "path": path,
                                "offset": offset,
                            }
                        window["paste"]["change"] = retained
                    if attributes := piece.get("attributes"):
                        window["attributes"] = attributes
                    stream.push(window)
                    cut["length"] += length
                else:  # insert or paste rides to the paste sites
                    stream.push(piece)
                remaining -= length

        if (
            first_other
            and isinstance(first_retained := first_other.get("retain"), (int, float))
            and first_other.get("attributes") is None
        ):
            first_left = first_retained
            while self_it.peek_type() in ("insert", "paste") and self_it.peek_length() <= first_left:
                first_left -= self_it.peek_length()
                ops.append(self_it.next())
            if first_retained - first_left > 0:
                other_it.next(first_retained - first_left)

        # deleting an embed whose payload sources moves sends it to the
        # trash instead: an open trash cut collects such units, and the
        # sourced windows are rewritten to read through it by path
        trash_part: op.CutOp | None = None

        def trash_unit() -> tuple[Ref, int]:
            nonlocal trash_part
            if trash_part is None or delta.ops[-1] is not trash_part:
                trash_part = {"cut": {"ref": fresh_ref("trash", used_refs), "length": 0}}
                delta.ops.append(trash_part)
            spec = trash_part["cut"]
            spec["length"] += 1
            return spec["ref"], spec["length"] - 1

        delta = self.__class__(ops)
        while self_it.has_next() or other_it.has_next():
            self_type = self_it.peek_type()
            other_type = other_it.peek_type()
            if other_type in ("insert", "paste"):
                next_op = other_it.next()
                if op.is_paste(next_op):
                    paste = next_op["paste"]
                    paste["ref"] = _PendingRef(paste["ref"])
                delta.push(next_op)
            elif self_type == "delete":
                delta.push(self_it.next())
            elif self_type == "cut":
                pass_self_cut()
            elif other_type == "cut":
                next_op = other_it.next()
                if op.is_cut(next_op):
                    consume_cut(next_op["cut"])
            else:
                length = min(self_it.peek_length(), other_it.peek_length())
                self_op = op.take(self_it, length)
                other_op = op.take(other_it, length)
                retained, other_retained = (operation.get("retain") for operation in (self_op, other_op))
                if other_type == "retain":
                    if not isinstance(other_retained, (int, float, dict)):
                        raise TypeError("retain content is missing")
                    if isinstance(retained, (int, float)):
                        new_op = op.RetainOp(
                            retain=(length if isinstance(other_retained, (int, float)) else other_retained)
                        )
                    elif op.is_paste(self_op):
                        paste = self_op["paste"].copy()
                        if isinstance(other_retained, dict):
                            # an embed patch at the paste site rides the
                            # window as its change
                            paste["change"] = compose_change(paste.get("change"), other_retained, shared)
                        new_op = op.PasteOp(paste=paste)
                    else:
                        action = "insert" if retained is None else "retain"
                        if isinstance(other_retained, (int, float)):
                            value = self_op.get("insert") if action == "insert" else self_op.get("retain")
                        else:
                            source = self_op.get("insert") if action == "insert" else self_op.get("retain")
                            embed_type, left, right = get_embed_type_and_data(embed_payload(source), other_retained)
                            handler = Delta.get_handler(embed_type)
                            value = {
                                embed_type: (
                                    handler.compose(left, right, shared)
                                    if action == "retain"
                                    else handler.apply(left, right, shared)
                                )
                            }
                        if action == "insert":
                            if not isinstance(value, (str, dict)):
                                raise TypeError("insert content is missing")
                            new_op = op.InsertOp(insert=value)
                        else:
                            if not isinstance(value, (int, float, dict)):
                                raise TypeError("retain content is missing")
                            new_op = op.RetainOp(retain=value)

                    attributes = op.compose(
                        self_op.get("attributes"),
                        other_op.get("attributes"),
                        isinstance(retained, (int, float)) or self_type == "paste",
                    )
                    if attributes:
                        new_op["attributes"] = attributes
                    delta.push(new_op)

                    if not other_it.has_next() and delta.ops[-1] == new_op:
                        delta = delta.concat(self.__class__(self_it.rest()))
                        break
                else:  # B deletes content emitted or retained by A
                    paste = self_op["paste"] if op.is_paste(self_op) else None
                    payload = paste.get("change") if paste is not None else retained
                    sites = list(payload_cut_sites(payload)) if isinstance(payload, dict) else []
                    if sites:
                        if paste is not None:
                            name, unit = paste["ref"], paste["start"]
                        else:
                            name, unit = trash_unit()
                        for ref, path, offset in sites:
                            trashed[ref] = {"ref": name, "unit": unit, "path": path, "offset": offset}
                    elif retained is not None:
                        delta.push(other_op)

        if not nested_pastes and not any(op.type(o) in ("cut", "paste") for o in delta.ops):
            return delta.chop()

        def replay(operation: Op, spec: op.PasteSpec, out: Self) -> None:
            # one window over its capture stream, consumed through the
            # op iterator so it splits inserts and re-windows carried
            # pastes at any unit boundary, with the riders applied
            outer = operation.get("attributes")
            change = spec.get("change")
            if "path" in spec:
                # a read through a trashed embed: the window's footprint
                # in the cut is the single unit it reads through
                for piece in op.sliced(refs_map[spec["ref"]].ops, spec.get("unit", 0), 1):
                    if op.is_paste(piece):
                        # the embed stays cut: retarget unit and ref
                        carried = piece["paste"]
                        paste = spec.copy()
                        paste.update(ref=carried["ref"], unit=carried["start"])
                        out.push(op.replace_paste(operation, paste))
                    elif isinstance(inserted := piece.get("insert"), dict):
                        # a carried embed: read the slice out of its
                        # child sequence
                        data = navigate_stream(inserted[next(iter(inserted))], spec.get("path", []), output=True)
                        if data is None:
                            continue
                        for read in op.sliced(data, spec["start"], spec["length"]):
                            if outer:
                                attributes = op.compose(read.get("attributes"), outer, False)
                                read.pop("attributes", None)
                                if attributes:
                                    read["attributes"] = attributes
                            if change and isinstance(read_insert := read.get("insert"), dict):
                                read = op.replace_insert(read, apply_change(read_insert, change, shared))
                            out.push(read)
                return
            for piece in op.sliced(refs_map[spec["ref"]].ops, spec["start"], spec["length"]):
                kind = op.type(piece)
                if outer:
                    merged = op.compose(piece.get("attributes"), outer, kind == "paste")
                    piece.pop("attributes", None)
                    if merged:
                        piece["attributes"] = merged
                if change:
                    if op.is_paste(piece):
                        inner = piece["paste"]
                        paste = inner.copy()
                        paste["change"] = compose_change(inner.get("change"), change, shared)
                        piece = op.replace_paste(piece, paste)
                    elif isinstance(inserted := piece.get("insert"), dict):
                        piece = op.replace_insert(piece, apply_change(inserted, change, shared))
                out.push(piece)

        def expand(item: Op, out: Self) -> bool:
            spec = item.get("paste")
            marker = pending_ref(item)
            if marker is not None:
                if spec is None:
                    raise RuntimeError("pending marker is not a paste")
                spec = spec.copy()
                spec["ref"] = marker
                item = op.replace_paste(item, spec)
            if spec is not None and spec["ref"] in refs_map:
                replay(item, spec, out)
                return True
            if spec is not None and spec["ref"] in shared["cuts"]:
                shared["retry"] = True
            if marker is not None:
                out.push(item)  # strip the private pending-ref prefix
                return True
            return False

        # replace every pending paste with its slice of the captured
        # stream, and re-slice windows riding embed payloads alike —
        # expanding the whole op covers insert and retain payloads and
        # a window's own change (a move into an embed that itself moved)
        resolved = rewrite_nested(delta.ops, self.__class__, expand)
        return self.__class__(delta.ops if resolved is None else resolved).chop()

    def diff(
        self,
        other: Self,
        cursor: int | None = None,
        *,
        _context: DiffContext | None = None,
    ) -> Self:
        """
        Deterministic typed snapshot diff between two *documents*
        (Deltas with only insert ops).  ``cursor`` is an optional caret
        hint (a base-document UTF-16 position) anchoring an ambiguous
        edit where the editor actually made it.
        """
        from .diff import snapshot_diff, snapshot_diff_at

        context = checked_context(_context, "diff") or {"_kind": "diff"}
        if cursor is None:
            return snapshot_diff(self, other, context)
        return snapshot_diff_at(self, other, cursor, context)

    def each_line(self, fn: Callable[[Self, Attributes, int], object], newline: str = "\n") -> None:
        for line, attributes, index in self.iter_lines(newline):
            if fn(line, attributes, index) is False:
                break

    def iter_lines(self, newline: str = "\n") -> Iterator[tuple[Self, Attributes, int]]:
        it = self.iterator()
        line = self.__class__()
        i = 0
        while it.has_next():
            if it.peek_type() != "insert":
                return
            current_op = it.peek()
            if current_op is None:
                return
            start = op.length(current_op) - it.peek_length()
            inserted = current_op.get("insert")
            if isinstance(inserted, str):
                suffix = op.str_slice(inserted, int(start))
                found = suffix.find(newline)
                nl_index = op.str_length(suffix[:found]) if found >= 0 else -1
            else:
                nl_index = -1

            if nl_index < 0:
                line.push(it.next())
            elif nl_index > 0:
                line.push(it.next(nl_index))
            else:
                attributes = it.next(1).get("attributes") or {}
                yield line, attributes, i
                i += 1
                line = self.__class__()
        if len(line) > 0:
            yield line, {}, i

    def lower(self, base: Self) -> Self:
        """Materialize moves, then express the result as an ordinary diff."""
        document = self.__class__._from_owned_ops(copy.deepcopy(base.ops)).compose(self)
        lowered = base.diff(document)
        return self.__class__._from_owned_ops(copy.deepcopy(lowered.ops))

    def invert(self, base: Self, *, _context: InvertContext | None = None) -> Self:
        inverted = self.__class__()
        base_it = base.iterator()

        def read_base(length: Number) -> Generator[Op]:
            while length > 0 and base_it.has_next():
                piece = base_it.next(min(length, base_it.peek_length()))
                length -= op.length(piece)
                yield piece

        # A move inverts by swapping halves: each paste window becomes a
        # cut of the pasted span, and the cut site re-pastes the windows
        # back in base order, restoring dropped gaps from base.  Window
        # changes and attributes invert against the base content they
        # rode over.  The first window of a ref keeps its name; further
        # windows split off numbered refs.
        context = checked_context(_context, "invert")
        if context is None:
            windows = paste_windows(self.ops, skip_inserts=True)
            names: dict[tuple[Ref, int], Ref] = {}
            taken: set[Ref] = set()
            for ref in sorted(windows):
                for index, (start, *_) in enumerate(windows[ref]):
                    names[ref, start] = fresh_ref(ref if index == 0 else f"{ref}:{index}", taken)
            context = InvertContext(_kind="invert", windows=windows, names=names)
        windows, names = context["windows"], context["names"]

        for operator in self.ops:
            kind = op.type(operator)
            retained = operator.get("retain")
            attributes = operator.get("attributes")
            if kind == "insert":
                inverted.delete(int(op.length(operator)))
            elif op.is_cut(operator):
                spec = operator["cut"]
                position = 0
                for start, length, attrs, change in windows.get(spec["ref"], []):
                    for gap in read_base(start - position):
                        inverted.push(gap)  # dropped: restore from base
                    offset = 0
                    for base_op in read_base(length):
                        piece: op.PasteOp = {
                            "paste": {
                                "ref": names[spec["ref"], start],
                                "start": offset,
                                "length": int(op.length(base_op)),
                            }
                        }
                        if change:
                            revert = invert_change(change, embed_payload(base_op.get("insert")), context)
                            if revert:
                                piece["paste"]["change"] = revert
                        if attrs:
                            reverted = op.invert(attrs, base_op.get("attributes"))
                            if reverted:
                                piece["attributes"] = reverted
                        inverted.push(piece)
                        offset += int(op.length(base_op))
                    position = start + length
                for gap in read_base(spec["length"] - position):
                    inverted.push(gap)
            elif op.is_paste(operator):
                spec = operator["paste"]
                if "path" in spec:
                    # the inverse restores the trashed embed whole
                    inverted.delete(spec["length"])
                else:
                    inverted.push({"cut": {"ref": names[spec["ref"], spec["start"]], "length": spec["length"]}})
            elif isinstance(retained, (int, float)) and attributes is None:
                inverted.retain(retained)
                for _base_op in read_base(retained):
                    pass
            elif op.is_delete(operator):
                for base_op in read_base(operator["delete"]):
                    inverted.push(base_op)
            elif isinstance(retained, (int, float)):
                for base_op in read_base(retained):
                    inverted.retain(op.length(base_op), **(op.invert(attributes, base_op.get("attributes")) or {}))
            elif isinstance(retained, dict):
                base_op = base_it.next(1)
                embed_type, data, based = get_embed_type_and_data(retained, embed_payload(base_op.get("insert")))
                change = {embed_type: Delta.get_handler(embed_type).invert(data, based, context)}
                inverted.retain(change, **(op.invert(attributes, base_op.get("attributes")) or {}))
        return inverted.chop()

    @overload
    def transform(self, other: Self, priority: bool = False, *, _context: TransformContext | None = None) -> Self: ...

    @overload
    def transform(self, other: int | float, priority: bool = False, *, _context: None = None) -> int | float: ...

    def transform(
        self,
        other: Self | int | float,
        priority: bool = False,
        *,
        _context: TransformContext | None = None,
    ) -> Self | int | float:
        context = checked_context(_context, "transform")
        if isinstance(other, (int, float)):
            return self.transform_position(other, priority)
        if context is not None:  # a child sequence joining the transaction
            return self._transform_with(other, priority, context, root=False)
        streams: dict[Ref, Delta] = {}
        self_cuts, taken, _self_nested = move_inventory(self.ops)
        other_cuts, other_pastes, _other_nested = move_inventory(other.ops)
        taken.update(self_cuts, other_cuts, other_pastes)
        self_windows = paste_windows(self.ops)
        other_windows = paste_windows(other.ops)
        result: Self | None = None
        for _ in range(2):
            shared = TransformContext(
                _kind="transform",
                streams=streams,
                claims={},
                taken=set(taken),
                doomed=set(),
                retry=False,
                self_cuts=self_cuts,
                self_windows=self_windows,
                other_windows=other_windows,
            )
            result = self._transform_with(other, priority, shared, root=True)
            if shared["retry"]:
                shared["retry"] = any(pending_ref(o) is not None for o in walk_moves(result.ops))
            if not shared["retry"]:
                break
            streams = {
                ref: self.__class__._from_owned_ops(copy.deepcopy(stream.ops))
                for ref, stream in shared["streams"].items()
            }
        else:
            raise RuntimeError("transform remained unresolved after retry")
        return normalize_orphans(result)

    def _transform_with(self, other: Self, priority: bool, shared: TransformContext, root: bool) -> Self:
        self_it = self.iterator()
        other_it = other.iterator()
        delta = self.__class__()

        # The compose stream idea with roles rotated: a cut of self
        # captures other's concurrent ops over the moved span — formats,
        # embed patches and deletes follow the content — and self's
        # paste windows replay the captured slice at the destination.
        # Inserts by other inside the span stay at the source.  Each cut
        # of other keeps a claim: one more stream mapping its old span
        # to what became of the units — pastes onto the re-emitted cut
        # parts where its claim holds, deletes where the units vanished
        # or were lost to a priority cut of self — and other's windows
        # re-slice that stream at the end.  Both are transaction-wide:
        # a cut in one sequence routes ops to windows in any other.
        streams = shared["streams"]
        claims = shared["claims"]
        doomed = shared["doomed"]

        def claim(ref: Ref) -> Claim:
            if ref not in claims:
                claims[ref] = Claim(stream=self.__class__(), parts=0, renamed=[])
            return claims[ref]

        # other's windows by ref, for relocating a dropped claim's
        # formats, embed changes and gap deletions onto the units at
        # our paste site
        other_windows = shared["other_windows"]

        def is_windowed(windows: PasteWindows, ref: Ref, start: int, length: int) -> bool:
            return any(
                max(start, window) < min(start + length, window + span)
                for window, span, _attrs, _change in windows.get(ref, ())
            )

        def doom_nested(change: Payload) -> None:
            for moved in walk_moves(change):
                if (spec := moved.get("cut")) is not None:
                    doomed.add(spec["ref"])

        def doom_payload_sources(retained: op.RetainValue | None) -> None:
            if isinstance(retained, dict):
                doomed.update(ref for ref, _path, _offset in payload_cut_sites(retained))

        taken = shared["taken"]

        # a retry starts with completed streams: rebuild each ref once
        building: set[Ref] = set()

        def capture_stream(ref: Ref) -> Delta:
            if ref not in building:
                streams[ref] = self.__class__()
                building.add(ref)
            return streams[ref]

        def claim_part(state: Claim, ref: Ref) -> Ref:
            # the first part inherits the claim's own ref; later splits
            # take numbered names, dodging other's literal refs
            name = ref if state["parts"] == 0 else fresh_ref(f"{ref}:{state['parts']}", taken)
            state["parts"] += 1
            return name

        def reclaim(
            state: Claim,
            ref: Ref,
            length: int,
            cache: dict[Ref, op.CutOp],
            attributes: Attributes | None = None,
            change: Payload | None = None,
        ) -> None:
            # open or extend the re-emitted cut part for ``ref`` in the
            # output, windowing the units onto it in the claim stream;
            # ``attributes`` and ``change`` carry self's concurrent
            # format and embed patch on them
            part = cache.get(ref)
            if part is None:
                part = op.CutOp(cut={"ref": claim_part(state, ref), "length": 0})
                delta.ops.append(part)
                cache[ref] = part
            cut = part["cut"]
            window: op.PasteOp = {"paste": {"ref": cut["ref"], "start": cut["length"], "length": length}}
            if change:
                window["paste"]["change"] = change
            if attributes:
                window["attributes"] = attributes
            state["stream"].push(window)
            cut["length"] += length

        def contested(self_ref: Ref, other_ref: Ref, length: int) -> None:
            # both moves claim these units: the priority side keeps them
            stream = capture_stream(self_ref)
            state = claim(other_ref)
            if priority:
                # their move claim drops and their windows go
                # positionally dead — but their formats land on the
                # units at our paste site, and their dropped gaps
                # remain deletions
                position = len(state["stream"])
                end = position + length
                state["stream"].delete(length)
                for start, span, attrs, change in other_windows.get(other_ref, []):
                    low = max(start, position)
                    high = min(start + span, end)
                    if low >= high:
                        continue
                    if low > position:
                        stream.delete(low - position)
                    # a change window addresses one embed: it relocates
                    # as an embed patch on the unit at our site
                    piece = op.RetainOp(retain=change if change else high - low)
                    if attrs:
                        piece["attributes"] = attrs
                    stream.push(piece)
                    position = high
                if position < end:
                    stream.delete(end - position)
            else:
                # theirs: their cut re-emerges wherever our windows put
                # these units — parts are named at resolution, once the
                # slicing is known; until then the claim holds a marker
                seen = len(state["stream"])
                stream.push({"cut": {"ref": other_ref, "start": seen, "length": length}})
                state["stream"].push({"paste": {"ref": _PendingRef(other_ref), "start": seen, "length": length}})

        def transformed_payload(value: Payload) -> Payload:
            # rebuild an embed payload, transforming each child op
            # sequence against an empty concurrent edit — the nested
            # walk joins the transaction, so nested cuts capture and
            # nested pastes replay across levels
            def transform_stream(sequence: Ops) -> Ops | None:
                if next(walk_moves(sequence), None) is None:
                    return None
                return list(self.__class__(sequence).transform(self.__class__(), priority, _context=shared).ops)

            replacement = map_streams(value, transform_stream)
            return value if replacement is None else replacement

        def pass_self_content() -> None:
            # self content occupies output space: an insert becomes
            # plain retained space, a paste waits for its capture
            next_op = self_it.next()
            if op.is_paste(next_op):
                inner = next_op["paste"]
                paste = inner.copy()
                paste["ref"] = _PendingRef(inner["ref"])
                delta.push(op.replace_paste(next_op, paste))
            elif isinstance(inserted := next_op.get("insert"), dict) and next(walk_moves(inserted), None) is not None:
                # A newly inserted embed can itself host the destination.
                # Replaying the shared capture against an empty child edit
                # yields the retain-style patch to apply to that new value.
                delta.retain(transformed_payload(inserted))
            else:
                delta.retain(op.length(next_op))

        def capture_cut(spec: op.CutSpec) -> None:
            stream = capture_stream(spec["ref"])
            remaining = spec["length"]
            position = spec.get("start", 0)
            while remaining > 0:
                if other_it.peek_type() in ("insert", "paste"):
                    delta.push(other_it.next())  # stays at the source
                    continue
                length = int(min(remaining, other_it.peek_length()))
                piece = op.take(other_it, length)
                if op.is_cut(piece):
                    contested(spec["ref"], piece["cut"]["ref"], length)
                else:
                    retained = piece.get("retain")
                    if isinstance(retained, dict) and not is_windowed(
                        shared["self_windows"], spec["ref"], position, length
                    ):
                        doom_nested(retained)
                    stream.push(piece)
                remaining -= length
                position += length

        def pass_cut(spec: op.CutSpec) -> None:
            # a move by other survives; its span shrinks by what self
            # deleted, splits into parts around what self inserted, and
            # loses contested units to a priority cut of self
            state = claim(spec["ref"])
            survived = state["stream"]
            position = 0
            parts: dict[Ref, op.CutOp] = {}
            while position < spec["length"]:
                if self_it.peek_type() in ("insert", "paste"):
                    pass_self_content()
                    parts.clear()  # self content splits the cut
                    continue
                length = int(min(spec["length"] - position, self_it.peek_length()))
                piece = op.take(self_it, length)
                if op.is_delete(piece):
                    survived.delete(length)
                elif op.is_cut(piece):
                    # contested units leave with self, so the in-place
                    # part stays positionally contiguous across them
                    contested(piece["cut"]["ref"], spec["ref"], length)
                else:
                    retained = piece.get("retain")
                    if isinstance(retained, dict) and not is_windowed(
                        other_windows, spec["ref"], spec.get("start", 0) + position, length
                    ):
                        doom_nested(retained)
                    reclaim(
                        state,
                        spec["ref"],
                        length,
                        parts,
                        attributes=piece.get("attributes"),
                        change=(retained if isinstance(retained, dict) else None),
                    )
                position += length
            if survived.ops == [{"paste": {"ref": spec["ref"], "start": 0, "length": spec["length"]}}]:
                del claims[spec["ref"]]  # untouched: windows stand as-is

        while self_it.has_next() or other_it.has_next():
            self_type = self_it.peek_type()
            other_type = other_it.peek_type()
            if self_type in ("insert", "paste") and (priority or other_type not in ("insert", "paste")):
                pass_self_content()
            elif other_type in ("insert", "paste"):
                delta.push(other_it.next())
            elif self_type == "cut":
                next_op = self_it.next()
                if op.is_cut(next_op):
                    capture_cut(next_op["cut"])
            elif other_type == "cut":
                next_op = other_it.next()
                if op.is_cut(next_op):
                    pass_cut(next_op["cut"])
            else:
                length = min(self_it.peek_length(), other_it.peek_length())
                self_op = op.take(self_it, length)
                other_op = op.take(other_it, length)
                if self_type == "delete":
                    # other's patch dies with the embed self deleted,
                    # along with moves whose sources live in its payload
                    doom_payload_sources(other_op.get("retain"))
                    continue
                elif other_type == "delete":
                    # deleting self's patch source kills its move windows
                    doom_payload_sources(self_op.get("retain"))
                    delta.push(other_op)
                else:
                    self_data = self_op.get("retain")
                    other_data = other_op.get("retain")
                    transformed_data = other_data if isinstance(other_data, dict) else length

                    if isinstance(self_data, dict) and isinstance(other_data, dict):
                        embed_type = next(iter(self_data))
                        if embed_type == next(iter(other_data)):
                            handler = Delta.get_handler(embed_type)
                            transformed_data = {
                                embed_type: handler.transform(
                                    self_data[embed_type], other_data[embed_type], priority, shared
                                )
                            }
                    elif isinstance(self_data, dict) and next(walk_moves(self_data), None) is not None:
                        # self's patch moves content across levels and
                        # other has no patch here: the transformed op
                        # grows the payload structure, each child
                        # sequence transformed against nothing so its
                        # captures fill and replays land through the
                        # shared transaction
                        transformed_data = transformed_payload(self_data)
                    delta.retain(
                        transformed_data,
                        **(op.transform(self_op.get("attributes"), other_op.get("attributes"), priority) or {}),
                    )

        if not claims and not doomed and all(pending_ref(o) is None for o in delta.ops):
            return delta.chop()

        # pass one: replay each captured slice at its pending window —
        # a re-emerging cut of other gets its final part name here, once
        # the slicing is known, recorded for the windows in pass two
        resolved = self.__class__()
        for operation in delta.ops:
            ref = pending_ref(operation)
            if ref is None:
                spec = operation.get("paste")
                if spec is not None and spec["ref"] in doomed:
                    continue  # the move died with its deleted source
                resolved.push(operation)
                continue
            spec = operation.get("paste")
            if spec is None:
                raise RuntimeError("pending marker is not a paste")
            if ref in doomed:
                # the moved content dies where it would have landed
                resolved.delete(spec["length"])
                continue
            moved_change: Payload | None = None
            own_change = spec.get("change")
            if isinstance(own_change, dict) and next(walk_moves(own_change), None) is not None:
                routed = transformed_payload(own_change)
                if any(sequence for _path, sequence in child_streams(routed)):
                    moved_change = routed
            stream = streams.get(ref)
            if stream is None:  # no cut for this ref: no capture
                if ref in shared["self_cuts"]:
                    shared["retry"] = True
                    resolved.push(operation)
                    continue
                resolved.retain(spec["length"])
                continue
            outer = operation.get("attributes")
            if "path" in spec:
                # a copy out of trash occupies its length in the output;
                # other's stance comes from its op on the source unit
                for piece in op.sliced(stream.ops, spec.get("unit", 0), 1):
                    retained = piece.get("retain")
                    if op.type(piece) in ("delete", "cut"):
                        # the source is deleted or concurrently claimed:
                        # with either priority the copy dies with it
                        resolved.delete(spec["length"])
                    elif isinstance(retained, dict):
                        # other's interior patch routes onto the copy
                        data = navigate_stream(retained[next(iter(retained))], spec.get("path", []), output=False)
                        if data is None:
                            resolved.retain(spec["length"])
                            continue
                        # A move wholly inside the trashed sequence loses to
                        # the read: leave its source in place and drop its
                        # local destination before slicing the child patch.
                        local_moves = {
                            paste["ref"] for item in walk_moves(data) if (paste := item.get("paste")) is not None
                        }
                        if local_moves:
                            neutral = self.__class__()
                            for item in data:
                                cut = item.get("cut")
                                paste = item.get("paste")
                                if cut is not None and cut["ref"] in local_moves:
                                    neutral.retain(cut["length"])
                                elif paste is None or paste["ref"] not in local_moves:
                                    neutral.push(item)
                            data = neutral.ops
                        window = input_effect(data, spec["start"], spec["start"] + spec["length"])
                        for routed in window.ops:
                            resolved.push(routed)
                    else:
                        resolved.retain(spec["length"])
                continue
            for piece in op.sliced(stream.ops, spec["start"], spec["length"]):
                kind = op.type(piece)
                if kind == "retain":
                    # a captured format or embed patch meets the
                    # window's own attributes and change
                    attributes = op.transform(outer, piece.get("attributes"), priority)
                    retained = piece.get("retain")
                    if not isinstance(retained, (int, float, dict)):
                        raise TypeError("retain content is missing")
                    if isinstance(retained, dict):
                        retained = transform_change(own_change, retained, priority, shared) or moved_change or 1
                    elif moved_change is not None:
                        retained = moved_change
                    piece = op.RetainOp(retain=retained)
                    if attributes:
                        piece["attributes"] = attributes
                elif op.is_cut(piece):
                    inner = piece["cut"]
                    state = claim(inner["ref"])
                    name = claim_part(state, inner["ref"])
                    state["renamed"].append((inner.get("start", 0), inner["length"], name))
                    piece = op.CutOp(cut={"ref": name, "length": inner["length"]})
                resolved.push(piece)

        # Sources hidden inside an enclosing unit can be declared doomed
        # after a sibling child zipper already emitted their destination.
        # Settle those recursive destinations once the root walk is complete.
        def settle(item: Op, out: Self) -> bool:
            marker = pending_ref(item)
            spec = item.get("paste")
            if marker is not None and marker in doomed:
                if spec is None:
                    raise RuntimeError("pending marker is not a paste")
                out.delete(spec["length"])
                return True
            return spec is not None and spec["ref"] in doomed

        if doomed:
            replacement = rewrite_nested(resolved.ops, self.__class__, settle)
            if replacement is not None:
                resolved = self.__class__(replacement)

        # pass two: re-slice other's windows through their claim streams
        # — at root and riding embed payloads alike (self's own pastes
        # are resolved, so real paste ops are other's)
        if claims and root:
            final = self.__class__()

            def place(
                out: Self,
                operation: Op,
                spec: op.PasteSpec,
                piece_spec: op.PasteSpec,
                attributes: Attributes | None,
                change: Payload | None,
            ) -> None:
                paste = spec.copy()
                paste.update(piece_spec)
                paste.pop("change", None)
                if change:
                    paste["change"] = change
                placed = op.replace_paste(operation, paste)
                if attributes:
                    placed["attributes"] = attributes
                else:
                    placed.pop("attributes", None)
                out.push(placed)

            def renamed_slices(inner: op.PasteSpec, parts: list[RenamedPart]) -> Generator[tuple[Ref, int, int]]:
                for start, length, name in parts:
                    low = max(start, inner["start"])
                    high = min(start + length, inner["start"] + inner["length"])
                    if low < high:
                        yield name, low - start, high - low

            def re_slice(operation: Op, spec: op.PasteSpec, out: Self) -> None:
                state = claims[spec["ref"]]
                if "path" in spec:
                    # a read's footprint in the cut is the one unit it
                    # reads through; its start renumbers through self's
                    # interior patch on the source
                    for piece in op.sliced(state["stream"].ops, spec.get("unit", 0), 1):
                        if not op.is_paste(piece):
                            continue  # source is gone: the read dies
                        inner = piece["paste"]
                        targets = [(inner["ref"], inner["start"])]
                        if pending_ref(piece) is not None:
                            targets = [
                                (name, start) for name, start, _length in renamed_slices(inner, state["renamed"])
                            ]
                        start = spec["start"]
                        length = spec["length"]
                        spans = [(start, length)]
                        applied = inner.get("change")
                        if applied is not None:
                            data = navigate_stream(applied[next(iter(applied))], spec.get("path", []), output=False)
                            if data is not None:
                                spans = list(read_spans(data, start, start + length))
                        for target, unit in targets:
                            for start, length in spans:
                                paste = spec.copy()
                                paste.update(ref=target, unit=unit, start=start, length=length)
                                out.push(op.replace_paste(operation, paste))
                    return
                pieces = list(op.sliced(state["stream"].ops, spec["start"], spec["length"]))
                # Resolve riders first: they can name cut parts used by an
                # earlier pending piece in this same claim.
                moved_changes: dict[int, Payload] = {}
                for index, piece in enumerate(pieces):
                    inner = piece.get("paste")
                    applied = inner.get("change") if inner is not None else None
                    if applied is not None and next(walk_moves(applied), None) is not None:
                        moved_changes[index] = transformed_payload(applied)
                for index, piece in enumerate(pieces):
                    if not op.is_paste(piece):
                        continue
                    # the window's attributes and change meet self's
                    # format and patch on the units this slice carries
                    attributes = op.transform(piece.get("attributes"), operation.get("attributes"), priority)
                    inner = piece["paste"].copy()
                    applied = inner.pop("change", None)
                    # A move inside self's patch rides the carried embed
                    # when no ordinary transformed change survives.
                    change = transform_change(applied, spec.get("change"), priority, shared) or moved_changes.get(index)
                    if pending_ref(piece) is None:
                        place(out, operation, spec, inner, attributes, change)
                        continue
                    # units re-claimed under self's windows: map them
                    # through the parts named in pass one, in the
                    # window's own coordinate order
                    for name, start, length in renamed_slices(inner, sorted(state["renamed"])):
                        place(
                            out,
                            operation,
                            spec,
                            {"ref": name, "start": start, "length": length},
                            attributes,
                            change,
                        )

            def grow(item: Op, out: Self) -> bool:
                if settle(item, out):
                    return True
                inner = item.get("paste")
                if inner is not None and inner["ref"] in claims:
                    re_slice(item, inner, out)
                    return True
                return False

            for operation in resolved.ops:
                expanded = rewrite_nested(operation, self.__class__, grow)
                if expanded is not None:
                    operation = expanded
                spec = operation.get("paste")
                if spec is None or spec["ref"] not in claims:
                    final.push(operation)
                    continue
                re_slice(operation, spec, final)
            resolved = final
        return resolved.chop()

    def transform_position(self, index: int | float, priority: bool = False) -> int | float:
        # a position strictly inside a moved span follows its unit to
        # the window that carries it; dropped units and the span start
        # collapse at the cut site, which the classic walk below already
        # does once a cut counts as a delete (and a paste as an insert)
        offset = 0
        for operation in self.ops:
            kind = op.type(operation)
            if op.is_cut(operation) and offset < index < offset + op.length(operation):
                unit = index - offset
                out = 0
                for other in self.ops:
                    other_kind = op.type(other)
                    spec = other.get("paste")
                    if (
                        spec
                        and spec["ref"] == operation["cut"]["ref"]
                        and spec["start"] <= unit < spec["start"] + spec["length"]
                    ):
                        return out + unit - spec["start"]
                    if other_kind not in ("delete", "cut"):
                        out += op.length(other)
                break  # a dropped unit: collapses at the cut site
            if kind in ("retain", "delete", "cut"):
                offset += op.length(operation)

        it = self.iterator()
        offset = 0
        while it.has_next() and offset <= index:
            length = it.peek_length()
            next_type = it.peek_type()
            it.next()
            if next_type in ("delete", "cut"):
                index -= min(length, index - offset)
                continue
            elif next_type in ("insert", "paste") and (offset < index or not priority):
                index += length
            offset += length
        return index


def has_moves(delta: Delta) -> bool:
    """Whether ``delta`` contains a cut or paste in any declared stream."""
    return next(walk_moves(delta.ops), None) is not None


def _integer_value(value: object) -> int | None:
    """The integers accepted by JavaScript's ``Number.isInteger``.

    JSON integers may arrive as integral floats in Python, while booleans
    must not pass merely because ``bool`` subclasses ``int``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return int(value)


def _reference(value: object, operation: Literal["cut", "paste"]) -> Ref:
    if not isinstance(value, str):
        raise ValueError(f"a {operation} needs a string reference")
    return value


def check[DeltaT: Delta](delta: DeltaT) -> DeltaT:
    """Validate transfer references explicitly, without taxing algebra calls."""
    cuts: dict[Ref, int] = {}
    windows: dict[Ref, list[tuple[int, int]]] = {}
    pathed: set[Ref] = set()

    for operation in walk_moves(delta.ops):
        cut = operation.get("cut")
        if cut is not None:
            ref = _reference(cut.get("ref"), "cut")
            length = _integer_value(cut.get("length"))
            if length is None or length <= 0:
                raise ValueError("a cut needs a positive integer length")
            if ref in cuts:
                raise ValueError(f"duplicate cut reference {ref!r}")
            cuts[ref] = length

        paste = operation.get("paste")
        if paste is None:
            continue
        ref = _reference(paste.get("ref"), "paste")
        start = _integer_value(paste.get("start"))
        length = _integer_value(paste.get("length"))
        if start is None or start < 0 or length is None or length <= 0:
            raise ValueError("a paste needs a non-negative integer start and a positive integer length")
        if paste.get("change") is not None and length != 1:
            raise ValueError("a paste change must address one embed")
        if "path" in paste:
            pathed.add(ref)
        else:
            windows.setdefault(ref, []).append((start, length))

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
