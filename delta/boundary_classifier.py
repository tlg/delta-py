"""
Classify how a delta interacts with block boundaries.
"""
from .base import Delta
from .labeled_state import assert_canonical_document, flatten_document_units
from .project import block_boundaries


def _is_newline_value(value, newline):
    return isinstance(value, str) and value == newline


def _flatten_inserted_units(value, attributes=None):
    return flatten_document_units(Delta().insert(value, **(attributes or {})))


def classify_delta_boundaries(document, delta, newline='\n'):
    """
    Classify how delta affects document boundaries.
    Returns: 'block-stable', 'whole-line structural', or 'split/merge'
    """
    assert_canonical_document(document, newline)

    base_units = flatten_document_units(document)
    boundaries = set(block_boundaries(document, newline))
    cursor = 0
    structural = False
    split_merge = False
    pending_insert = []
    pending_insert_at = 0

    def flush_insert():
        nonlocal structural, split_merge, pending_insert
        if not pending_insert:
            return
        contains_newline = any(_is_newline_value(u['value'], newline) for u in pending_insert)
        if contains_newline:
            structural = True
            ends_with_newline = _is_newline_value(pending_insert[-1]['value'], newline)
            if pending_insert_at not in boundaries or not ends_with_newline:
                split_merge = True
        pending_insert = []

    for o in delta.ops:
        if o.get('insert') is not None:
            pending_insert.extend(_flatten_inserted_units(o['insert'], o.get('attributes')))
            continue

        flush_insert()

        if isinstance(o.get('delete'), int):
            start = cursor
            end = cursor + o['delete']
            deleted = base_units[start:end]
            if any(_is_newline_value(u['value'], newline) for u in deleted):
                structural = True
                if start not in boundaries or end not in boundaries:
                    split_merge = True
            cursor = end
            pending_insert_at = cursor
            continue

        if isinstance(o.get('retain'), int):
            cursor += o['retain']
            pending_insert_at = cursor
            continue

    flush_insert()

    if split_merge:
        return 'split/merge'
    if structural:
        return 'whole-line structural'
    return 'block-stable'
