"""
Document coordinates that transform through move-aware deltas.

A coordinate addresses a position in a nested document as a tuple:

    (5,)                     a caret at root offset 5
    (2, 'ops', 3)            a caret at offset 3 inside the child sequence
                             at payload key 'ops' of the embed at root
                             position 2
    (2, 'ops', 1, 'ops', 0)  ... and so on, one embed unit per level

``transform_coordinate(delta, coordinate)`` maps a coordinate through a
delta.  Carets shift with inserts and deletes, follow content that a move
relocates — across sequence levels, in either direction, including trash
reads — and collapse to the removal point when their span is deleted.
Embed *units* (any element followed by more coordinate) return ``None``
when the unit was deleted.

Boundary conventions match the algebra: a caret exactly at the start of a
moved region stays at the source, like a concurrent insert there; a caret
strictly inside follows the content.
"""
from . import op
from .moves import (MoveDelta, _collect_windows, _input_length,
                    _output_length, _unit_patch, _unit_position)


def transform_coordinate(delta, coordinate, priority=False):
    delta = delta if isinstance(delta, MoveDelta) else MoveDelta(list(delta.ops))
    reads = []  # trash reads carry addressed content to an output position
    position = 0
    for operation in delta.ops:
        if op.type(operation) == 'paste' and 'path' in operation['paste']:
            reads.append((operation['paste'], position))
        position += _output_length(operation)
    return _resolve(delta, tuple(coordinate),
                    _collect_windows(list(delta.ops)), reads, priority, ())


def _descend(patch, prefix, rest, windows, reads, priority):
    """Continue resolving ``rest`` through a child patch payload, or keep
    it verbatim where the patch does not reach."""
    keys = []
    for part in rest:
        if not isinstance(part, str):
            break
        keys.append(part)
    tail = rest[len(keys):]
    child_ops = patch
    for key in keys:
        child_ops = child_ops.get(key) if isinstance(child_ops, dict) else None
    if not tail or not isinstance(child_ops, list):
        return (*prefix, *rest)
    return _resolve(MoveDelta(list(child_ops)), tuple(tail), windows,
                    reads, priority, (*prefix, *keys))


def _through_read(spec, out_position, local, coordinate):
    """The coordinate's new home when a trash read carries the addressed
    content to ``out_position``, or None when it is not covered."""
    if spec.get('unit', 0) != local:
        return None
    path = tuple(spec['path'])
    tail = coordinate[1:]
    if tuple(tail[:len(path)]) != path or len(tail) <= len(path):
        return None
    rest = tail[len(path):]
    offset = rest[0]
    if not isinstance(offset, int):
        return None
    start, end = spec['start'], spec['start'] + spec['length']
    inside = (start <= offset < end) if len(rest) > 1 else (start < offset < end)
    if not inside:
        return None
    return (out_position + offset - start, *rest[1:])


def _resolve(delta, coordinate, windows, reads, priority, prefix):
    target = coordinate[0]
    is_unit = len(coordinate) > 1

    # relocation: is the addressed position inside a moved region?
    input_position = 0
    for operation in delta.ops:
        kind = op.type(operation)
        length = _input_length(operation)
        local = target - input_position
        if kind == 'cut' and 0 <= local < length:
            if is_unit or local > 0:  # boundary carets stay at the source
                ref = operation['cut']['ref']
                for window in windows.get(ref, ()):
                    start, location = window.start, window.location
                    if start <= local < start + window.length:
                        if location[0] == 'root':
                            head = (location[1] + local - start,)
                        else:
                            _, hops, child_prefix = location
                            flat = []
                            for unit, _embed_type, keys in hops:
                                flat.append(unit)
                                flat.extend(keys)
                            head = (*flat, child_prefix + local - start)
                        rest = coordinate[1:]
                        if not rest or window.change is None:
                            return (*head, *rest)
                        # the covering paste also patches the moved embed
                        return _descend(window.change[next(iter(window.change))],
                                        head, rest, windows, reads, priority)
                # a trash read may still carry the addressed content out
                for spec, out_position in reads:
                    if spec['ref'] != ref:
                        continue
                    followed = _through_read(spec, out_position, local,
                                             coordinate)
                    if followed is not None:
                        return followed
                if is_unit:
                    return None  # cut but never pasted: the unit is gone
            break
        if kind == 'delete' and 0 <= local < length and is_unit:
            return None
        input_position += length

    if not is_unit:
        return (*prefix, delta.transform_position(target, priority))

    # descend into the embed unit
    mapped = _unit_position(delta, target)
    if mapped is None:
        return None
    patch = _unit_patch(delta, target, windows)
    if patch is None:
        return (*prefix, mapped, *coordinate[1:])  # untouched inside
    return _descend(patch, (*prefix, mapped), coordinate[1:],
                    windows, reads, priority)
