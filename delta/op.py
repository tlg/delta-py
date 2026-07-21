"""
Op-level primitives: attribute algebra, UTF-16 text units, and the
splitting iterator every delta pass is built on.

Lengths and offsets are UTF-16 code units, matching the upstream
JavaScript library; ops are plain JSON-shaped dicts (the wire format).
"""

import copy
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Never, NotRequired, Self, TypedDict, TypeGuard, TypeIs, overload

type Attributes = dict[str, Any]
type OpKind = Literal["insert", "retain", "delete", "cut", "paste"]
type Payload = dict[str, Any]
type Ref = str
type Number = int | float
type Unit = int
type Distance = Number  # float is reserved for the iterator's infinity sentinel
type PathStep = str | Unit
type Path = list[PathStep]
type InsertValue = str | Payload | Number
type RetainValue = Distance | Payload


class CutSpec(TypedDict):
    ref: Ref
    length: Unit
    start: NotRequired[Unit]


class PasteSpec(TypedDict):
    ref: Ref
    start: Unit
    length: Unit
    change: NotRequired[Payload]
    unit: NotRequired[Unit]
    path: NotRequired[Path]


class AttributedOp(TypedDict, total=False):
    attributes: Attributes


class InsertOp(AttributedOp):
    insert: InsertValue
    retain: NotRequired[Never]
    delete: NotRequired[Never]
    cut: NotRequired[Never]
    paste: NotRequired[Never]


class RetainOp(AttributedOp):
    insert: NotRequired[Never]
    retain: RetainValue
    delete: NotRequired[Never]
    cut: NotRequired[Never]
    paste: NotRequired[Never]


class DeleteOp(AttributedOp):
    insert: NotRequired[Never]
    retain: NotRequired[Never]
    delete: Unit
    cut: NotRequired[Never]
    paste: NotRequired[Never]


class CutOp(AttributedOp):
    insert: NotRequired[Never]
    retain: NotRequired[Never]
    delete: NotRequired[Never]
    cut: CutSpec
    paste: NotRequired[Never]


class PasteOp(AttributedOp):
    insert: NotRequired[Never]
    retain: NotRequired[Never]
    delete: NotRequired[Never]
    cut: NotRequired[Never]
    paste: PasteSpec


type Op = InsertOp | RetainOp | DeleteOp | CutOp | PasteOp
type Ops = list[Op]


# ── UTF-16 text units ──


def str_length(text: str) -> int:
    """
    Length in UTF-16 code units — the unit all delta offsets count,
    matching the upstream JavaScript library (astral characters, like
    most emoji, count 2).
    """
    if text.isascii():
        return len(text)
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def str_slice(text: str, start: int, stop: int | None = None) -> str:
    """
    Slice by UTF-16 code-unit offsets.  A boundary may fall inside a
    surrogate pair, exactly as JavaScript allows; the lone surrogate is
    preserved so the halves reassemble.
    """
    if text.isascii():
        return text[start:stop]
    data = text.encode("utf-16-le", "surrogatepass")
    stop_at = None if stop is None else 2 * stop
    return data[2 * start : stop_at].decode("utf-16-le", "surrogatepass")


def str_join(left: str, right: str) -> str:
    """
    Concatenate two pieces of text, re-pairing a surrogate pair a UTF-16
    split may have pulled apart (Python's ``+`` would keep the two lone
    surrogates as two code points).
    """
    if left and right and "\ud800" <= left[-1] <= "\udbff" and "\udc00" <= right[0] <= "\udfff":
        return (left + right).encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")
    return left + right


# ── Attribute algebra ──
# Attribute sets are patches: a key set to None removes it on apply.


def compose(a: Attributes | None, b: Attributes | None, keep_null: bool = False) -> Attributes | None:
    """Compose two attribute patches; ``keep_null`` retains explicit removals."""
    a = a or {}
    b = b or {}
    attributes = {k: copy.deepcopy(v) for k, v in b.items() if keep_null or v is not None}
    for k, v in a.items():
        if k not in b:
            attributes[k] = copy.deepcopy(v)
    return attributes or None


def diff(a: Attributes | None, b: Attributes | None) -> Attributes | None:
    """The patch turning attribute set ``a`` into ``b``."""
    a = a or {}
    b = b or {}
    attributes = {k: b.get(k) for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    return attributes or None


def invert(attr: Attributes | None, base: Attributes | None) -> Attributes:
    """The patch undoing ``attr`` against the attributes it applied over."""
    attr = attr or {}
    base = base or {}
    result: Attributes = {k: v for k, v in base.items() if v != attr.get(k) and k in attr}
    result |= {k: None for k, v in attr.items() if v != base.get(k) and k not in base}
    return result


def transform(a: Attributes | None, b: Attributes | None, priority: bool = True) -> Attributes | None:
    """Transform patch ``b`` against concurrent ``a``; without priority ``b`` wins wholesale."""
    a = a or {}
    b = b or {}
    if not priority:
        return b or None
    attributes = {k: v for k, v in b.items() if k not in a}
    return attributes or None


# ── Op helpers ──
# Malformed move specs deliberately fail these guards and fall back to
# the opaque length-1 insert path: length/type never raise on bad input.


def clone(operation: Op) -> Op:
    """Detach one JSON-shaped operation, deep-copying only when it has
    a nested attribute, embed, or move value."""
    if any(isinstance(value, (dict, list)) for value in operation.values()):
        return copy.deepcopy(operation)
    return copy.copy(operation)


def is_cut(op: Op) -> TypeGuard[CutOp]:
    match op:
        case {"cut": {"length": int()}}:
            return True
        case _:
            return False


def is_paste(op: Op) -> TypeGuard[PasteOp]:
    match op:
        case {"paste": {"length": int(), "start": int()}}:
            return True
        case _:
            return False


def is_delete(op: Op) -> TypeGuard[DeleteOp]:
    match op:
        case {"delete": int()}:
            return True
        case _:
            return False


def is_retain(op: Op) -> bool:
    match op:
        case {"retain": int() | float()}:
            return True
        case {"retain": dict() as retained}:
            return bool(retained)
        case _:
            return False


def is_insert(op: Op) -> TypeIs[InsertOp]:
    return op.get("insert") is not None


def is_retain_op(op: Op) -> TypeIs[RetainOp]:
    return op.get("retain") is not None


def replace_insert(source: Op, value: InsertValue) -> InsertOp:
    attributes = source.get("attributes")
    return InsertOp(insert=value, attributes=attributes) if attributes is not None else InsertOp(insert=value)


def replace_retain(source: Op, value: RetainValue) -> RetainOp:
    attributes = source.get("attributes")
    return RetainOp(retain=value, attributes=attributes) if attributes is not None else RetainOp(retain=value)


def replace_delete(source: Op, length: Unit) -> DeleteOp:
    attributes = source.get("attributes")
    return DeleteOp(delete=length, attributes=attributes) if attributes is not None else DeleteOp(delete=length)


def replace_paste(source: Op, spec: PasteSpec) -> PasteOp:
    attributes = source.get("attributes")
    return PasteOp(paste=spec, attributes=attributes) if attributes is not None else PasteOp(paste=spec)


@overload
def length_of(op: InsertOp | DeleteOp | CutOp | PasteOp) -> int: ...


@overload
def length_of(op: RetainOp) -> Distance: ...


@overload
def length_of(op: Op) -> Distance: ...


def length_of(op: Op) -> Distance:
    if is_delete(op):
        return op["delete"]
    if is_retain(op):
        value = op.get("retain")
        if isinstance(value, (int, float)):
            return value
        return 1
    if is_cut(op):
        return op["cut"]["length"]
    if is_paste(op):
        return op["paste"]["length"]
    value = op.get("insert")
    if isinstance(value, str):
        return str_length(value)
    return 1


@overload
def type_of(op: Op) -> OpKind: ...


@overload
def type_of(op: None) -> None: ...


def type_of(op: Op | None) -> OpKind | None:
    if not op:
        return None
    if is_delete(op):
        return "delete"
    if is_cut(op):
        return "cut"
    if is_paste(op):
        return "paste"
    if is_retain(op):
        return "retain"
    return "insert"


def input_length(op: Op) -> int | float:
    """Units consumed by an operation."""
    return 0 if type_of(op) in ("insert", "paste") else length_of(op)


def output_length(op: Op) -> int | float:
    """Units emitted by an operation."""
    return 0 if type_of(op) in ("delete", "cut") else length_of(op)


# ── Splitting iterator ──


@dataclass(slots=True)
class Iterator:
    """
    A stateful cursor over ops that can split operations to exactly the
    length needed via ``next()``.
    """

    ops: list[Op] = field(default_factory=list[Op])
    index: int = 0
    offset: int | float = 0

    def reset(self) -> None:
        self.index = 0
        self.offset = 0

    def has_next(self) -> bool:
        return self.peek_length() < math.inf

    def next(self, length: int | float | None = None) -> Op:
        if length is None:
            length = math.inf

        op = self.peek()
        if op is None:
            return {"retain": math.inf}

        offset = self.offset
        op_length = length_of(op)
        if length >= op_length - offset:
            length = op_length - offset
            self.index += 1
            self.offset = 0
        else:
            self.offset += length

        if is_delete(op):
            if not isinstance(length, int):
                raise TypeError("delete operations require integer units")
            return {"delete": length}

        attrs = op.get("attributes")

        if is_cut(op):
            if not isinstance(length, int) or not isinstance(offset, int):
                raise TypeError("cut operations require integer units")
            # a split cut keeps its offset within the original span,
            # so stream slicing can tell which units a piece carries;
            # a whole cut passes through with no start key at all
            source = op["cut"]
            cut = CutSpec(ref=source["ref"], length=length)
            if start := source.get("start", 0) + offset:
                cut["start"] = start
            return {"cut": cut}
        if is_paste(op):
            if not isinstance(length, int) or not isinstance(offset, int):
                raise TypeError("paste operations require integer units")
            paste = op["paste"].copy()
            paste["start"] += offset
            paste["length"] = length
            return PasteOp(paste=paste, attributes=attrs) if attrs else PasteOp(paste=paste)
        if is_retain(op):
            value = op.get("retain")
            if isinstance(value, (int, float)):
                retained_value: RetainValue = length
            elif isinstance(value, dict):
                retained_value = value
            else:
                raise TypeError("retain content is missing")
            return RetainOp(retain=retained_value, attributes=attrs) if attrs else RetainOp(retain=retained_value)
        value = op.get("insert")
        if not isinstance(value, (str, dict, int, float)):
            raise TypeError("insert content is missing")
        inserted_value = str_slice(value, int(offset), int(offset + length)) if isinstance(value, str) else value
        return InsertOp(insert=inserted_value, attributes=attrs) if attrs else InsertOp(insert=inserted_value)

    __next__ = next

    def __iter__(self) -> Self:
        return self

    def peek(self) -> Op | None:
        if self.index < len(self.ops):
            return self.ops[self.index]
        return None

    def peek_length(self) -> int | float:
        op = self.peek()
        if op is None:
            return math.inf
        return length_of(op) - self.offset

    def peek_input_length(self) -> int | float:
        return 0 if self.peek_type() in ("insert", "paste") else self.peek_length()

    def peek_output_length(self) -> int | float:
        return 0 if self.peek_type() in ("delete", "cut") else self.peek_length()

    def peek_type(self) -> OpKind:
        op = self.peek()
        return type_of(op) or "retain"

    def rest(self) -> list[Op]:
        if not self.has_next():
            return []
        if self.offset == 0:
            return self.ops[self.index :]
        offset = self.offset
        index = self.index
        result = self.next()
        remaining = self.ops[self.index :]
        self.offset = offset
        self.index = index
        return [result, *remaining]


# Upstream quill-delta public names (``type`` deliberately shadows the
# builtin inside this namespace, as in the original port).
length = length_of
type = type_of


def take(it: Iterator, length: int | float) -> Op:
    """``it.next(length)`` clamped to a plain retain when exhausted (a
    bare ``next`` would yield ``{'retain': inf}``)."""
    if it.peek() is None:
        return {"retain": length}
    return it.next(length)


def sliced(ops: list[Op], start: int | float, length: int | float):
    """Yield the ops of the [start, start + length) unit slice of
    ``ops``, split through the iterator at any unit boundary."""
    it = iterator(ops)
    while start > 0 and it.has_next():
        start -= length_of(it.next(start))
    while length > 0 and it.has_next():
        piece = it.next(length)
        length -= length_of(piece)
        yield piece


def iterator(ops: Iterable[Op] | None) -> Iterator:
    return Iterator(list(ops) if ops is not None else [])
