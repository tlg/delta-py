"""
Labeled state: unit-level document model with gap anchors for OT resolution.
"""
import copy

from . import op as op_module


def _clone_attributes(attributes=None):
    return copy.deepcopy(attributes or {})


def _clone_value(value):
    return value if isinstance(value, str) else copy.deepcopy(value)


_ORIGIN_GAP = '_origin_gap'
_MOVE_SOURCE_BLOCK_IDS = '_move_source_block_ids'
_MOVE_BEFORE_BLOCK_ID = '_move_before_block_id'


def same_gap_anchor(a, b):
    return (a['afterUnitId'] == b['afterUnitId']
            and a['beforeUnitId'] == b['beforeUnitId']
            and a['gapId'] == b['gapId'])


def _attach_origin_gap(unit, gap=None):
    if gap is not None:
        unit[_ORIGIN_GAP] = copy.deepcopy(gap)
    return unit


def get_unit_origin_gap(unit):
    gap = unit.get(_ORIGIN_GAP)
    return copy.deepcopy(gap) if gap is not None else None


def attach_resolved_move_block_identity(move, source_block_ids, before_block_id):
    move[_MOVE_SOURCE_BLOCK_IDS] = source_block_ids[:]
    move[_MOVE_BEFORE_BLOCK_ID] = before_block_id
    return move


def get_resolved_move_source_block_ids(move):
    ids = move.get(_MOVE_SOURCE_BLOCK_IDS)
    return ids[:] if ids is not None else None


def get_resolved_move_before_block_id(move):
    return move.get(_MOVE_BEFORE_BLOCK_ID)


def _split_utf16_code_units(s):
    """Split a string into individual UTF-16 code units to match JS string semantics."""
    result = []
    for char in s:
        code = ord(char)
        if code > 0xFFFF:
            # Encode as UTF-16 surrogate pair
            code -= 0x10000
            high = chr(0xD800 + (code >> 10))
            low = chr(0xDC00 + (code & 0x3FF))
            result.append(high)
            result.append(low)
        else:
            result.append(char)
    return result


def flatten_document_units(document):
    units = []
    for o in document.ops:
        if o.get('insert') is None or o.get('delete') is not None or o.get('retain') is not None:
            raise ValueError('document delta must contain only inserts')
        attributes = _clone_attributes(o.get('attributes'))
        insert = o['insert']
        if isinstance(insert, str):
            for char in _split_utf16_code_units(insert):
                units.append({'value': char, 'attributes': _clone_attributes(attributes)})
        else:
            units.append({'value': _clone_value(insert), 'attributes': _clone_attributes(attributes)})
    return units


def is_canonical_document(document, newline='\n'):
    try:
        units = flatten_document_units(document)
        return (len(units) > 0
                and isinstance(units[-1]['value'], str)
                and units[-1]['value'] == newline)
    except Exception:
        return False


def assert_canonical_document(document, newline='\n'):
    units = flatten_document_units(document)
    if (len(units) == 0
            or not isinstance(units[-1]['value'], str)
            or units[-1]['value'] != newline):
        raise ValueError('canonical document delta must end with a final newline')


def canonicalize_document(document, newline='\n'):
    from .base import Delta
    units = flatten_document_units(document)
    canonical = Delta(copy.deepcopy(document.ops))
    if (len(units) == 0
            or not isinstance(units[-1]['value'], str)
            or units[-1]['value'] != newline):
        canonical.insert(newline)
    return canonical


def _build_gap_anchors(units):
    gaps = [{
        'gapId': 0,
        'afterUnitId': None,
        'beforeUnitId': units[0]['id'] if units else None,
    }]
    for i, unit in enumerate(units):
        gaps.append({
            'gapId': i + 1,
            'afterUnitId': unit['id'],
            'beforeUnitId': units[i + 1]['id'] if i + 1 < len(units) else None,
        })
    return gaps


def labeled_state_from_document(document, newline='\n'):
    canonical = canonicalize_document(document, newline)
    units = [
        {'id': i + 1, 'value': _clone_value(u['value']), 'attributes': _clone_attributes(u['attributes'])}
        for i, u in enumerate(flatten_document_units(canonical))
    ]
    return {'units': units, 'gaps': _build_gap_anchors(units)}


def clone_labeled_unit(unit):
    cloned = {
        'id': unit['id'],
        'value': _clone_value(unit['value']),
        'attributes': _clone_attributes(unit.get('attributes')),
    }
    origin = get_unit_origin_gap(unit)
    _attach_origin_gap(cloned, origin)
    return cloned


def clone_labeled_state(state):
    return {
        'units': [clone_labeled_unit(u) for u in state['units']],
        'gaps': [{'gapId': g['gapId'], 'afterUnitId': g['afterUnitId'], 'beforeUnitId': g['beforeUnitId']}
                 for g in state['gaps']],
    }


def _is_high_surrogate(ch):
    return isinstance(ch, str) and len(ch) == 1 and 0xD800 <= ord(ch) <= 0xDBFF


def _is_low_surrogate(ch):
    return isinstance(ch, str) and len(ch) == 1 and 0xDC00 <= ord(ch) <= 0xDFFF


def _combine_surrogates(high, low):
    code = 0x10000 + ((ord(high) - 0xD800) << 10) + (ord(low) - 0xDC00)
    return chr(code)


def labeled_state_to_delta(state):
    from .base import Delta
    d = Delta()
    units = state['units']
    i = 0
    while i < len(units):
        unit = units[i]
        value = unit['value']
        attrs = _clone_attributes(unit.get('attributes'))
        # Recombine UTF-16 surrogate pairs back into proper Unicode
        if (_is_high_surrogate(value) and i + 1 < len(units)
                and _is_low_surrogate(units[i + 1]['value'])
                and units[i + 1].get('attributes', {}) == attrs):
            value = _combine_surrogates(value, units[i + 1]['value'])
            i += 2
        else:
            i += 1
        d.insert(_clone_value(value), **attrs)
    return d


def max_unit_id(state):
    return max((u['id'] for u in state['units']), default=0)


def _flatten_inserted_content(value, attributes=None):
    if isinstance(value, str):
        return [{'value': char, 'attributes': _clone_attributes(attributes)} for char in value]
    return [{'value': _clone_value(value), 'attributes': _clone_attributes(attributes)}]


def find_gap_index(state, gap):
    units = state['units']
    if gap['afterUnitId'] is None:
        if gap['beforeUnitId'] is None:
            return 0
        for i, u in enumerate(units):
            if u['id'] == gap['beforeUnitId']:
                return i
        return 0
    for i, u in enumerate(units):
        if u['id'] == gap['afterUnitId']:
            return i + 1
    if gap['beforeUnitId'] is None:
        return len(units)
    for i, u in enumerate(units):
        if u['id'] == gap['beforeUnitId']:
            return i
    return len(units)


def _unit_was_inserted_at_gap(state, index, gap):
    if index < 0 or index >= len(state['units']):
        return False
    origin = get_unit_origin_gap(state['units'][index])
    return origin is not None and same_gap_anchor(origin, gap)


def _descendant_run_start(state, before_index, gap):
    index = before_index
    while index > 0 and _unit_was_inserted_at_gap(state, index - 1, gap):
        index -= 1
    return index


def _after_whole_line_prefix_index(state, before_index, gap, newline='\n'):
    start = _descendant_run_start(state, before_index, gap)
    insertion_index = start
    for index in range(start, before_index):
        unit = state['units'][index]
        if isinstance(unit['value'], str) and unit['value'] == newline:
            insertion_index = index + 1
    return insertion_index


def _find_gap_edge_index(state, gap, edge, newline='\n'):
    units = state['units']
    after_index = -1
    before_index = -1
    for i, u in enumerate(units):
        if u['id'] == gap.get('afterUnitId'):
            after_index = i
        if u['id'] == gap.get('beforeUnitId'):
            before_index = i

    if edge == 'before':
        if after_index >= 0:
            return after_index + 1
        if before_index >= 0:
            index = before_index
            while index > 0 and _unit_was_inserted_at_gap(state, index - 1, gap):
                index -= 1
            return index
        return find_gap_index(state, gap)

    # edge == 'after'
    if before_index >= 0:
        return _after_whole_line_prefix_index(state, before_index, gap, newline)
    if after_index >= 0:
        index = after_index + 1
        while index < len(units) and _unit_was_inserted_at_gap(state, index, gap):
            index += 1
        return index
    return find_gap_index(state, gap)


def classify_gap_descendants(state, gap, newline='\n'):
    before_index = _find_gap_edge_index(state, gap, 'before', newline)
    after_index = _find_gap_edge_index(state, gap, 'after', newline)
    descendants = state['units'][before_index:after_index]
    whole_line_prefix_units = 0
    for i, unit in enumerate(descendants):
        if isinstance(unit['value'], str) and unit['value'] == newline:
            whole_line_prefix_units = i + 1
    trailing_partial_units = len(descendants) - whole_line_prefix_units
    return {
        'wholeLinePrefixUnits': whole_line_prefix_units,
        'trailingPartialUnits': trailing_partial_units,
        'afterEdgeBlockExpressible': trailing_partial_units == 0,
    }


def rebuild_labeled_state(units):
    cloned = [clone_labeled_unit(u) for u in units]
    return {'units': cloned, 'gaps': _build_gap_anchors(cloned)}


def resolve_delta_against_state(base, delta):
    inserts_by_gap = []
    deleted_unit_ids = []
    format_patches_by_unit_id = []
    cursor = 0
    next_inserted_id = max_unit_id(base) + 1

    for o in delta.ops:
        if o.get('insert') is not None:
            units = []
            for seed in _flatten_inserted_content(o['insert'], o.get('attributes')):
                unit = {'id': next_inserted_id, 'value': seed['value'], 'attributes': seed['attributes']}
                _attach_origin_gap(unit, base['gaps'][cursor])
                units.append(unit)
                next_inserted_id += 1
            inserts_by_gap.append({
                'gap': copy.deepcopy(base['gaps'][cursor]),
                'units': units,
            })
            continue
        if isinstance(o.get('delete'), int):
            end = cursor + o['delete']
            for unit in base['units'][cursor:end]:
                deleted_unit_ids.append(unit['id'])
            cursor = end
            continue
        if isinstance(o.get('retain'), dict):
            raise ValueError('resolved delta does not support embed retain payload changes')
        if isinstance(o.get('retain'), int):
            end = cursor + o['retain']
            if o.get('attributes'):
                for unit in base['units'][cursor:end]:
                    format_patches_by_unit_id.append({
                        'unitId': unit['id'],
                        'attributes': _clone_attributes(o.get('attributes')),
                    })
            cursor = end
            continue
        raise ValueError('invalid delta op')

    return {
        'insertsByGap': inserts_by_gap,
        'deletedUnitIds': deleted_unit_ids,
        'formatPatchesByUnitId': format_patches_by_unit_id,
    }


def resolve_delta(document, delta, newline='\n'):
    assert_canonical_document(document, newline)
    return resolve_delta_against_state(labeled_state_from_document(document, newline), delta)


def replay_resolved_delta(state, resolved):
    deleted = set(resolved['deletedUnitIds'])
    format_map = {}
    for patch in resolved['formatPatchesByUnitId']:
        format_map[patch['unitId']] = _clone_attributes(patch['attributes'])

    units = []
    for unit in state['units']:
        if unit['id'] in deleted:
            continue
        patch = format_map.get(unit['id'])
        if patch is None:
            units.append(clone_labeled_unit(unit))
        else:
            composed = op_module.compose(unit.get('attributes'), patch, False) or {}
            cloned = {
                'id': unit['id'],
                'value': _clone_value(unit['value']),
                'attributes': composed,
            }
            _attach_origin_gap(cloned, get_unit_origin_gap(unit))
            units.append(cloned)

    working = {'units': units, 'gaps': _build_gap_anchors(units)}

    for insert in resolved['insertsByGap']:
        gap_index = _find_gap_edge_index(working, insert['gap'], 'after')
        cloned_units = [clone_labeled_unit(u) for u in insert['units']]
        for i, u in enumerate(cloned_units):
            working['units'].insert(gap_index + i, u)
        working['gaps'] = _build_gap_anchors(working['units'])

    return working
