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
from typing import Any, Literal

type Attributes = dict[str, Any]
type Op = dict[str, Any]
type OpKind = Literal["insert", "retain", "delete", "cut", "paste"]


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


def is_cut(op: Op) -> bool:
    return isinstance(op.get("cut"), dict) and isinstance(op["cut"].get("length"), int)


def is_paste(op: Op) -> bool:
    return (
        isinstance(op.get("paste"), dict)
        and isinstance(op["paste"].get("length"), int)
        and isinstance(op["paste"].get("start"), int)
    )


def length_of(op: Op) -> int | float:
    match op:
        case {"delete": int(units)}:
            return units
        case {"retain": int() | float() as units}:
            return units
        case {"retain": dict() as patch} if patch:
            return 1
        case {"insert": str(text)}:
            return str_length(text)
        case _ if is_cut(op):
            return op["cut"]["length"]
        case _ if is_paste(op):
            return op["paste"]["length"]
        case _:
            return 1


def type_of(op: Op | None) -> OpKind | None:
    if not op:
        return None
    match op:
        case {"delete": int()}:
            return "delete"
        case _ if is_cut(op):
            return "cut"
        case _ if is_paste(op):
            return "paste"
        case {"retain": int()}:
            return "retain"
        case {"retain": dict() as patch} if patch:
            return "retain"
        case _:
            return "insert"


# ── Splitting iterator ──


@dataclass(slots=True)
class Iterator:
    """
    A stateful cursor over ops that can split operations to exactly the
    length needed via ``next()``.
    """

    ops: list[Op] = field(default_factory=list)
    index: int = 0
    offset: int = 0

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

        op_type = type_of(op)
        offset = self.offset
        op_length = length_of(op)
        if length >= op_length - offset:
            length = op_length - offset
            self.index += 1
            self.offset = 0
        else:
            self.offset += length

        if op_type == "delete":
            return {"delete": length}

        result_op: Op = {}
        if op.get("attributes"):
            result_op["attributes"] = op["attributes"]

        match op:
            case _ if op_type == "cut":
                result_op["cut"] = {"ref": op["cut"]["ref"], "length": length}
            case _ if op_type == "paste":
                paste = op["paste"]
                result_op["paste"] = {**paste, "start": paste["start"] + offset, "length": length}
            case {"retain": int() | float()}:
                result_op["retain"] = length
            case {"retain": dict() as patch} if patch:
                result_op["retain"] = patch
            case {"insert": str(text)}:
                result_op["insert"] = str_slice(text, offset, offset + length)
            case _:
                assert offset == 0
                assert length == 1
                if "insert" in op:
                    result_op["insert"] = op["insert"]

        return result_op

    __next__ = next

    def __iter__(self) -> "Iterator":
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
type = type_of  # noqa: A001


def iterator(ops: Iterable[Op] | None) -> Iterator:
    return Iterator(list(ops) if ops is not None else [])
