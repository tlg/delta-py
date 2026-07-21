"""Embed handler registry and structural child-stream traversal."""

from collections.abc import Callable, Generator, Iterable
from typing import Any, Protocol, cast

from .op import Op, Ops, PathStep, Payload


type StreamPath = tuple[PathStep, ...]
type StreamTransform = Callable[[Ops], Ops | None]


class EmbedHandler[Value, Change, Context](Protocol):
    """Structural contract implemented by registered embed types."""

    def apply(self, value: Value, change: Change, context: Context) -> Value: ...

    def compose(self, first: Change, second: Change, context: Context) -> Change | None: ...

    def transform(self, first: Change, second: Change, priority: bool, context: Context) -> Change | None: ...

    def invert(self, change: Change, base: Value, context: Context) -> Change | None: ...

    def diff(self, base: Value, target: Value, context: Context) -> Change | None: ...


handlers: dict[str, EmbedHandler[Any, Any, Any]] = {}


def streams(payload: Payload) -> Generator[tuple[StreamPath, Ops]]:
    """Yield ``(path, ops)`` for child Delta streams declared by an
    embed handler. Paths are relative to the handler's payload value."""
    if len(payload) != 1:
        return
    embed_type, data = next(iter(payload.items()))
    handler = handlers.get(embed_type)
    paths = cast("Callable[[object], Iterable[Iterable[PathStep]]] | None", getattr(handler, "stream_paths", None))
    if not callable(paths):
        return
    for path in paths(data):
        path = tuple(path)
        value: object = data
        for step in path:
            if isinstance(value, list) and isinstance(step, int):
                value = cast("list[object]", value)[step]
            elif isinstance(value, dict):
                value = cast("dict[PathStep, object]", value)[step]
            else:
                raise TypeError(f"{embed_type!r} handler stream path {path!r} cannot traverse {value!r}")
        if not isinstance(value, list):
            raise TypeError(f"{embed_type!r} handler stream path {path!r} does not address a list")
        yield path, cast("Ops", value)


def _replaced(value: object, path: StreamPath, replacement: object) -> object:
    if not path:
        return replacement
    step = path[0]
    if isinstance(value, list) and isinstance(step, int):
        result = list(cast("list[object]", value))
        result[step] = _replaced(result[step], path[1:], replacement)
        return result
    if isinstance(value, dict):
        result = dict(cast("dict[PathStep, object]", value))
        result[step] = _replaced(result[step], path[1:], replacement)
        return result
    raise TypeError(f"stream path {path!r} cannot traverse {value!r}")


def map_streams(payload: Payload, transform: StreamTransform) -> Payload | None:
    """Copy-on-change map of a handler's declared child streams.

    ``transform`` returns replacement ops or ``None`` for no change.
    """
    if len(payload) != 1:
        return None
    embed_type, data = next(iter(payload.items()))
    updated: object = data
    for path, _ops in streams(payload):
        current: object = updated
        for step in path:
            if isinstance(current, list) and isinstance(step, int):
                current = cast("list[object]", current)[step]
            elif isinstance(current, dict):
                current = cast("dict[PathStep, object]", current)[step]
            else:
                raise TypeError(f"stream path {path!r} cannot traverse {current!r}")
        if not isinstance(current, list):
            raise TypeError(f"stream path {path!r} does not address a list")
        replacement = transform(cast("Ops", current))
        if replacement is not None:
            updated = _replaced(updated, path, replacement)
    return None if updated is data else {embed_type: updated}


def walk_move_ops(value: object, skip_inserts: bool = False) -> Generator[Op]:
    """Yield cut/paste operations from a root stream or handler-owned
    child streams; opaque embed data is never inspected."""
    if isinstance(value, list):
        for raw in cast("list[object]", value):
            if not isinstance(raw, dict):
                continue
            operation = cast("Op", raw)
            values = cast("dict[str, object]", raw)
            if isinstance(values.get("cut"), dict) or isinstance(values.get("paste"), dict):
                yield operation
            for carrier in ("insert", "retain"):
                payload = values.get(carrier)
                if carrier == "insert" and skip_inserts:
                    continue
                if isinstance(payload, dict):
                    yield from walk_move_ops(cast("dict[str, object]", payload), skip_inserts)
            spec = values.get("paste")
            if isinstance(spec, dict):
                change = cast("dict[str, object]", spec).get("change")
                if isinstance(change, dict):
                    yield from walk_move_ops(cast("dict[str, object]", change), skip_inserts)
        return
    if isinstance(value, dict):
        for _path, child_ops in streams(cast("Payload", value)):
            yield from walk_move_ops(child_ops, skip_inserts)
