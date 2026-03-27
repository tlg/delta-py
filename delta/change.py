"""
Two-stream change bridge: delta + block-delta operations resolved against labeled state.
"""
import copy
from functools import reduce

from .base import Delta
from .block import BlockDelta, diff_to_moves, normalize_move, resolve_move
from .boundary_classifier import classify_delta_boundaries
from .labeled_state import (
    assert_canonical_document,
    attach_resolved_move_block_identity,
    classify_gap_descendants,
    clone_labeled_state,
    clone_labeled_unit,
    find_gap_index,
    get_resolved_move_before_block_id,
    get_resolved_move_source_block_ids,
    get_unit_origin_gap,
    labeled_state_from_document,
    labeled_state_to_delta,
    max_unit_id,
    rebuild_labeled_state,
    replay_resolved_delta,
    resolve_delta_against_state,
    same_gap_anchor,
    _find_gap_edge_index,
    _after_whole_line_prefix_index,
    _unit_was_inserted_at_gap,
    _descendant_run_start,
)
from .project import block_boundary_gap_anchors, project_labeled_block_spans


# ── Clone helpers ──

def _clone_gap_anchor(gap):
    return copy.deepcopy(gap)


def _clone_boundary(boundary):
    return {
        'gap': _clone_gap_anchor(boundary['gap']),
        'newlineUnit': clone_labeled_unit(boundary['newlineUnit']) if boundary['newlineUnit'] else None,
    }


def _clone_resolved_move(move):
    cloned = {
        'left': _clone_boundary(move['left']),
        'right': _clone_boundary(move['right']),
        'before': _clone_boundary(move['before']),
    }
    source_block_ids = get_resolved_move_source_block_ids(move)
    if source_block_ids is not None:
        attach_resolved_move_block_identity(
            cloned, source_block_ids,
            get_resolved_move_before_block_id(move))
    return cloned


def _clone_boundary_restoration(restoration):
    return {
        'boundary': _clone_boundary(restoration['boundary']),
        'edge': restoration.get('edge'),
        'restored': restoration['restored'],
        'restoredUnitId': restoration.get('restoredUnitId'),
    }


# ── Replay helpers ──

def _is_newline_unit(value, newline='\n'):
    return isinstance(value, str) and value == newline


def _is_boundary_index(state, gap_index, newline='\n'):
    if gap_index == 0:
        return True
    return (gap_index > 0
            and gap_index <= len(state['units'])
            and _is_newline_unit(state['units'][gap_index - 1]['value'], newline))


def _boundary_from_gap(state, gap, gap_index):
    return {
        'gap': _clone_gap_anchor(gap),
        'newlineUnit': clone_labeled_unit(state['units'][gap_index - 1]) if gap_index > 0 else None,
    }


def _block_identity(state, from_idx, to_idx, newline='\n'):
    if to_idx <= from_idx or to_idx > len(state['units']):
        return None
    unit = state['units'][to_idx - 1]
    return unit['id'] if _is_newline_unit(unit['value'], newline) else None


def _split_inserted_fragment(units, newline='\n'):
    prefix_units = 0
    for i, unit in enumerate(units):
        if isinstance(unit['value'], str) and unit['value'] == newline:
            prefix_units = i + 1
    return {
        'wholeLinePrefix': [clone_labeled_unit(u) for u in units[:prefix_units]],
        'trailingPartial': [clone_labeled_unit(u) for u in units[prefix_units:]],
    }


def _insert_units_at_index(state, index, units):
    if not units:
        return clone_labeled_state(state)
    next_units = state['units'][:]
    for i, u in enumerate(units):
        next_units.insert(index + i, clone_labeled_unit(u))
    return rebuild_labeled_state(next_units)


def _restore_boundary(state, boundary, edge, newline='\n', preferred_unit_id=None):
    gap_index = _find_gap_edge_index(state, boundary['gap'], edge, newline)
    if _is_boundary_index(state, gap_index, newline) or boundary['newlineUnit'] is None:
        return {'state': state, 'restored': False, 'restoredUnitId': None}
    restored_unit = clone_labeled_unit(boundary['newlineUnit'])
    restored_unit['id'] = max_unit_id(state) + 1 if preferred_unit_id is None else preferred_unit_id
    units = state['units'][:]
    units.insert(gap_index, restored_unit)
    return {'state': rebuild_labeled_state(units), 'restored': True, 'restoredUnitId': restored_unit['id']}


def _materialize_boundary_gap_index(state, boundary, edge, restored_unit_id):
    if restored_unit_id is None:
        return _find_gap_edge_index(state, boundary['gap'], edge, '\n')
    for i, u in enumerate(state['units']):
        if u['id'] == restored_unit_id:
            return i + 1
    raise ValueError('restored boundary unit not found in current state')


def _resolve_block_move_on_state(state, index, move):
    blocks = project_labeled_block_spans(state)
    resolved = normalize_move(resolve_move(index, move['count'], move['before']), len(blocks))
    if resolved is None:
        return None
    boundary_anchors = block_boundary_gap_anchors(state)
    left_gap = blocks[resolved['index']]['leftGap']
    right_gap = blocks[resolved['index'] + resolved['count'] - 1]['rightGap']
    before_gap = boundary_anchors[resolved['before']]
    resolved_move = {
        'left': _boundary_from_gap(state, left_gap,
                                   0 if resolved['index'] == 0 else blocks[resolved['index']]['from']),
        'right': _boundary_from_gap(state, right_gap,
                                    blocks[resolved['index'] + resolved['count'] - 1]['to']),
        'before': _boundary_from_gap(state, before_gap,
                                     0 if resolved['before'] == 0 else blocks[resolved['before'] - 1]['to']),
    }
    source_block_ids = [
        _block_identity(state, b['from'], b['to'])
        for b in blocks[resolved['index']:resolved['index'] + resolved['count']]
    ]
    source_block_ids = [bid for bid in source_block_ids if bid is not None]
    before_block_id = (
        _block_identity(state, blocks[resolved['before']]['from'], blocks[resolved['before']]['to'])
        if resolved['before'] < len(blocks) else None
    )
    return attach_resolved_move_block_identity(resolved_move, source_block_ids, before_block_id)


def _materialize_move_by_block_identity(state, move, destination_edge, newline='\n'):
    if destination_edge == 'after' and get_resolved_move_before_block_id(move) is not None:
        return None
    source_block_ids = get_resolved_move_source_block_ids(move)
    if not source_block_ids:
        return None
    blocks = project_labeled_block_spans(state, newline)
    block_ids = [_block_identity(state, b['from'], b['to'], newline) for b in blocks]
    try:
        start_index = block_ids.index(source_block_ids[0])
    except ValueError:
        return None
    if start_index + len(source_block_ids) > len(blocks):
        return None
    for offset in range(len(source_block_ids)):
        if block_ids[start_index + offset] != source_block_ids[offset]:
            return None
    before_block_id = get_resolved_move_before_block_id(move)
    if before_block_id is None:
        before_block_index = len(blocks)
    else:
        try:
            before_block_index = block_ids.index(before_block_id)
        except ValueError:
            return None
    start = blocks[start_index]['from']
    end = blocks[start_index + len(source_block_ids) - 1]['to']
    before = len(state['units']) if before_block_index == len(blocks) else blocks[before_block_index]['from']
    return {
        'start': start, 'end': end, 'before': before,
        'noop': start_index <= before_block_index <= start_index + len(source_block_ids),
    }


def _materialize_resolved_block_move(state, move, newline='\n', options=None):
    options = options or {}
    destination_edge = options.get('destinationEdge', 'before')
    restore = options.get('restoreBoundaries', True)

    working = clone_labeled_state(state)
    restorations = []

    left = _restore_boundary(working, move['left'], 'after', newline) if restore else {'state': working, 'restored': False, 'restoredUnitId': None}
    working = left['state']
    restorations.append({'boundary': _clone_boundary(move['left']), 'edge': 'after', 'restored': left['restored'], 'restoredUnitId': left['restoredUnitId']})

    right = _restore_boundary(working, move['right'], 'before', newline) if restore else {'state': working, 'restored': False, 'restoredUnitId': None}
    working = right['state']
    restorations.append({'boundary': _clone_boundary(move['right']), 'edge': 'before', 'restored': right['restored'], 'restoredUnitId': right['restoredUnitId']})

    before = _restore_boundary(working, move['before'], destination_edge, newline) if restore else {'state': working, 'restored': False, 'restoredUnitId': None}
    working = before['state']
    restorations.append({'boundary': _clone_boundary(move['before']), 'edge': destination_edge, 'restored': before['restored'], 'restoredUnitId': before['restoredUnitId']})

    identity_exact = _materialize_move_by_block_identity(working, move, destination_edge, newline)
    if identity_exact is not None:
        return {'state': working, 'restorations': restorations, 'exact': identity_exact}

    left_gap_index = _materialize_boundary_gap_index(working, move['left'], 'after', left['restoredUnitId'])
    right_gap_index = _materialize_boundary_gap_index(working, move['right'], 'before', right['restoredUnitId'])
    before_gap_index = _materialize_boundary_gap_index(working, move['before'], destination_edge, before['restoredUnitId'])

    return {
        'state': working, 'restorations': restorations,
        'exact': {
            'start': left_gap_index, 'end': right_gap_index, 'before': before_gap_index,
            'noop': left_gap_index <= before_gap_index <= right_gap_index,
        },
    }


def _replay_boundary_restoration(state, restoration, newline='\n'):
    if not restoration['restored']:
        return state
    return _restore_boundary(
        state, restoration['boundary'],
        restoration.get('edge', 'before'), newline,
        restoration.get('restoredUnitId'))['state']


def _same_labeled_state(left, right):
    import json
    return json.dumps([u for u in left['units'] if not u.get('_origin_gap')],
                      default=str) == json.dumps([u for u in right['units'] if not u.get('_origin_gap')], default=str)


def _same_labeled_state_units(left, right):
    if len(left['units']) != len(right['units']):
        return False
    for l, r in zip(left['units'], right['units']):
        if l['id'] != r['id'] or l['value'] != r['value'] or l.get('attributes') != r.get('attributes'):
            return False
    return True


# ── Public replay API ──

def replay_resolved_block_move(state, move, newline='\n', options=None):
    materialized = _materialize_resolved_block_move(state, move, newline, options)
    exact = materialized['exact']
    if exact['noop']:
        return {'state': materialized['state'], 'restorations': materialized['restorations']}
    moved = [clone_labeled_unit(u) for u in materialized['state']['units'][exact['start']:exact['end']]]
    remaining = [clone_labeled_unit(u) for u in
                 materialized['state']['units'][:exact['start']] + materialized['state']['units'][exact['end']:]]
    insertion_index = exact['before'] if exact['before'] < exact['start'] else exact['before'] - (exact['end'] - exact['start'])
    reordered = remaining[:insertion_index] + moved + remaining[insertion_index:]
    return {'state': rebuild_labeled_state(reordered), 'restorations': materialized['restorations']}


def replay_resolved_block_moves(state, moves, newline='\n', options=None):
    result = {'state': clone_labeled_state(state), 'restorations': []}
    for move in moves:
        next_result = replay_resolved_block_move(result['state'], move, newline, options)
        result = {'state': next_result['state'], 'restorations': result['restorations'] + next_result['restorations']}
    return result


def ensure_resolved_block_moves(state, moves, newline='\n', destination_edge='before'):
    result = {'state': clone_labeled_state(state), 'restorations': []}
    for move in moves:
        materialized = _materialize_resolved_block_move(
            result['state'], move, newline,
            {'destinationEdge': destination_edge, 'restoreBoundaries': True})
        result = {'state': materialized['state'], 'restorations': result['restorations'] + materialized['restorations']}
    return result


def resolve_block_delta(state, block_delta, newline='\n'):
    resolved_moves = []
    working = clone_labeled_state(state)
    for move in block_delta.resolve(len(project_labeled_block_spans(state, newline))):
        resolved = _resolve_block_move_on_state(working, move['index'], {'count': move['count'], 'before': move['before']})
        if resolved is None:
            continue
        resolved_moves.append(_clone_resolved_move(resolved))
        working = replay_resolved_block_move(working, resolved, newline)['state']
    return resolved_moves


def _resolve_change_against_state(state, change, newline='\n'):
    resolved_delta = resolve_delta_against_state(state, change['delta'])
    post_delta_state = replay_resolved_delta(state, resolved_delta)
    resolved_moves = resolve_block_delta(post_delta_state, change['blockDelta'], newline)
    return {'resolvedDelta': resolved_delta, 'postDeltaState': post_delta_state, 'resolvedMoves': resolved_moves}


def _final_state_of(resolved, newline='\n'):
    return replay_resolved_block_moves(resolved['postDeltaState'], resolved['resolvedMoves'], newline)['state']


def _block_count(state, newline='\n'):
    return len(project_labeled_block_spans(state, newline))


def _delta_only_change(from_state, to_state):
    return {'delta': labeled_state_to_delta(from_state).diff(labeled_state_to_delta(to_state)), 'blockDelta': BlockDelta()}


# ── Replay with insert edge ──

def _replay_resolved_delta_with_insert_edge(state, resolved, insert_edge, newline='\n'):
    working = replay_resolved_delta(state, {
        'insertsByGap': [],
        'deletedUnitIds': resolved['deletedUnitIds'],
        'formatPatchesByUnitId': resolved['formatPatchesByUnitId'],
    })
    for insert in resolved['insertsByGap']:
        split = _split_inserted_fragment(insert['units'], newline)
        if split['wholeLinePrefix']:
            if insert_edge == 'before':
                idx = _find_gap_edge_index(working, insert['gap'], 'before', newline)
            else:
                before_idx = find_gap_index(working, insert['gap'])
                idx = _after_whole_line_prefix_index(working, before_idx, insert['gap'], newline)
            working = _insert_units_at_index(working, idx, split['wholeLinePrefix'])
        if split['trailingPartial']:
            before_idx = find_gap_index(working, insert['gap'])
            partial_before = _after_whole_line_prefix_index(working, before_idx, insert['gap'], newline)
            partial_after = _find_gap_edge_index(working, insert['gap'], 'after', newline)
            idx = partial_before if insert_edge == 'before' else partial_after
            working = _insert_units_at_index(working, idx, split['trailingPartial'])
    return working


# ── Rekey ──

def _rekey_resolved_change_against_state(resolved, state):
    next_unit_id = max_unit_id(state) + 1
    unit_id_map = {}
    for insert in resolved['resolvedDelta']['insertsByGap']:
        for unit in insert['units']:
            if unit['id'] not in unit_id_map:
                unit_id_map[unit['id']] = next_unit_id
                next_unit_id += 1

    def remap_unit(unit):
        cloned = clone_labeled_unit(unit)
        cloned['id'] = unit_id_map.get(cloned['id'], cloned['id'])
        return cloned

    def remap_gap(gap):
        return {
            'gapId': gap['gapId'],
            'afterUnitId': unit_id_map.get(gap['afterUnitId'], gap['afterUnitId']) if gap['afterUnitId'] is not None else None,
            'beforeUnitId': unit_id_map.get(gap['beforeUnitId'], gap['beforeUnitId']) if gap['beforeUnitId'] is not None else None,
        }

    def remap_boundary(boundary):
        nl = None
        if boundary['newlineUnit']:
            nl = clone_labeled_unit(boundary['newlineUnit'])
            nl['id'] = unit_id_map.get(nl['id'], nl['id'])
        return {'gap': remap_gap(boundary['gap']), 'newlineUnit': nl}

    return {
        'resolvedDelta': {
            'insertsByGap': [{'gap': copy.deepcopy(ins['gap']), 'units': [remap_unit(u) for u in ins['units']]} for ins in resolved['resolvedDelta']['insertsByGap']],
            'deletedUnitIds': resolved['resolvedDelta']['deletedUnitIds'][:],
            'formatPatchesByUnitId': [{'unitId': p['unitId'], 'attributes': copy.deepcopy(p['attributes'])} for p in resolved['resolvedDelta']['formatPatchesByUnitId']],
        },
        'postDeltaState': rebuild_labeled_state([remap_unit(u) for u in resolved['postDeltaState']['units']]),
        'resolvedMoves': [
            (lambda m: (
                attach_resolved_move_block_identity(m,
                    [unit_id_map.get(bid, bid) for bid in (get_resolved_move_source_block_ids(mv) or [])],
                    (lambda bbid: unit_id_map.get(bbid, bbid) if bbid is not None else None)(get_resolved_move_before_block_id(mv)))
                if get_resolved_move_source_block_ids(mv) is not None else m
            ))(
                {'left': remap_boundary(mv['left']), 'right': remap_boundary(mv['right']), 'before': remap_boundary(mv['before'])}
            )
            for mv in resolved['resolvedMoves']
        ],
    }


# ── Lowering ──

def _block_order_keys(state, newline='\n'):
    return [
        ','.join(str(u['id']) for u in state['units'][b['from']:b['to']])
        for b in project_labeled_block_spans(state, newline)
    ]


def _try_lower_block_delta(from_state, to_state, newline='\n'):
    try:
        return BlockDelta.from_moves(
            diff_to_moves(_block_order_keys(from_state, newline), _block_order_keys(to_state, newline)))
    except Exception:
        return None


def _lower_change(from_state, post_delta_state, final_state, newline='\n'):
    from_doc = labeled_state_to_delta(from_state)
    post_delta_doc = labeled_state_to_delta(post_delta_state)
    final_doc = labeled_state_to_delta(final_state)
    block_delta = _try_lower_block_delta(post_delta_state, final_state, newline)
    if block_delta is None:
        return {'delta': from_doc.diff(final_doc), 'blockDelta': BlockDelta()}
    return {'delta': from_doc.diff(post_delta_doc), 'blockDelta': block_delta}


# ── Prepare ──

def _replay_prepared_prefix(state, steps, newline='\n'):
    result = clone_labeled_state(state)
    for step in steps:
        if step['isNoOp']:
            result = clone_labeled_state(result)
        else:
            result = replay_resolved_block_move(result, step['move'], newline,
                                                {'destinationEdge': step['destinationEdge'], 'restoreBoundaries': True})['state']
    return result


def _prepare_resolved_block_move(before_state, move, newline='\n', destination_edge='before'):
    materialized = _materialize_resolved_block_move(before_state, move, newline,
                                                    {'destinationEdge': destination_edge, 'restoreBoundaries': True})
    if materialized['exact']['noop']:
        return {
            'move': _clone_resolved_move(move), 'beforeState': clone_labeled_state(before_state),
            'preparedState': clone_labeled_state(before_state), 'finalState': clone_labeled_state(before_state),
            'isNoOp': True, 'destinationEdge': destination_edge, 'restored': [],
        }
    final_state = replay_resolved_block_move(materialized['state'], move, newline,
                                             {'destinationEdge': destination_edge, 'restoreBoundaries': False})['state']
    return {
        'move': _clone_resolved_move(move), 'beforeState': clone_labeled_state(before_state),
        'preparedState': clone_labeled_state(materialized['state']), 'finalState': clone_labeled_state(final_state),
        'isNoOp': False, 'destinationEdge': destination_edge,
        'restored': [_clone_boundary_restoration(r) for r in materialized['restorations']],
    }


def _is_boundary_edge_block_expressible(state, boundary, edge, newline='\n'):
    return edge == 'before' or classify_gap_descendants(state, boundary['gap'], newline)['afterEdgeBlockExpressible']


def _insert_boundary_at_gap(state, restoration, gap_index):
    if not restoration['restored'] or restoration['boundary']['newlineUnit'] is None:
        return clone_labeled_state(state)
    restored_unit = clone_labeled_unit(restoration['boundary']['newlineUnit'])
    restored_unit['id'] = restoration.get('restoredUnitId') or max_unit_id(state) + 1
    units = state['units'][:]
    units.insert(gap_index, restored_unit)
    return rebuild_labeled_state(units)


def _pull_back_required_boundary(pre_move_state, required, prior_steps, current_step_state, target_state, newline='\n'):
    if not required['restored'] or required['boundary']['newlineUnit'] is None:
        return {'state': clone_labeled_state(pre_move_state), 'found': True}
    if (required.get('edge') == 'after'
            and not classify_gap_descendants(current_step_state, required['boundary']['gap'], newline)['afterEdgeBlockExpressible']):
        return {'state': clone_labeled_state(pre_move_state), 'found': False}
    for gap_index in range(len(pre_move_state['units']) + 1):
        candidate = _insert_boundary_at_gap(pre_move_state, required, gap_index)
        if _same_labeled_state_units(_replay_prepared_prefix(candidate, prior_steps, newline), target_state):
            return {'state': candidate, 'found': True}
    return {'state': clone_labeled_state(pre_move_state), 'found': False}


def _prepare_resolved_block_moves(initial_state, move_initial_state, moves, newline='\n', choose_destination_edge=None):
    if choose_destination_edge is None:
        choose_destination_edge = lambda _s, _m, _i: 'before'

    current = clone_labeled_state(move_initial_state)
    pre_move_state = clone_labeled_state(move_initial_state)
    expressible = True
    steps = []

    for step_index, move in enumerate(moves):
        destination_edge = choose_destination_edge(current, move, step_index)
        step = _prepare_resolved_block_move(current, move, newline, destination_edge)

        if expressible and not step['isNoOp']:
            if (not _is_boundary_edge_block_expressible(step['beforeState'], step['move']['left'], 'after', newline)
                    or not _is_boundary_edge_block_expressible(step['beforeState'], step['move']['before'], destination_edge, newline)):
                expressible = False

        local_state = clone_labeled_state(step['beforeState'])
        for restoration in step['restored']:
            if not restoration['restored']:
                continue
            target = _replay_boundary_restoration(local_state, restoration, newline)
            if expressible:
                pulled = _pull_back_required_boundary(pre_move_state, restoration, steps, local_state, target, newline)
                if pulled['found']:
                    pre_move_state = pulled['state']
                else:
                    expressible = False
            local_state = target

        steps.append(step)
        current = clone_labeled_state(step['finalState'])

    if expressible and not _same_labeled_state_units(_replay_prepared_prefix(pre_move_state, steps, newline), current):
        expressible = False

    return {
        'initialState': clone_labeled_state(initial_state),
        'moveInitialState': clone_labeled_state(move_initial_state),
        'preMoveState': pre_move_state,
        'steps': steps,
        'finalState': current,
        'expressibleAsBlockDelta': expressible,
    }


def _lower_prepared_move_program(program, newline='\n'):
    if not program['expressibleAsBlockDelta']:
        return {
            'delta': labeled_state_to_delta(program['initialState']).diff(labeled_state_to_delta(program['finalState'])),
            'blockDelta': BlockDelta(),
        }
    block_delta = _try_lower_block_delta(program['preMoveState'], program['finalState'], newline)
    if block_delta is None:
        return {
            'delta': labeled_state_to_delta(program['initialState']).diff(labeled_state_to_delta(program['finalState'])),
            'blockDelta': BlockDelta(),
        }
    return {
        'delta': labeled_state_to_delta(program['initialState']).diff(labeled_state_to_delta(program['preMoveState'])),
        'blockDelta': block_delta,
    }


def _prepare_and_lower(initial_state, move_initial_state, moves, exact_final_state, newline='\n', choose_destination_edge=None):
    try:
        prepared = _prepare_resolved_block_moves(initial_state, move_initial_state, moves, newline, choose_destination_edge)
        if _same_document_state(prepared['finalState'], exact_final_state):
            return _lower_prepared_move_program(prepared, newline)
    except Exception:
        pass
    return _delta_only_change(initial_state, exact_final_state)


def _same_document_state(left, right):
    return labeled_state_to_delta(left).ops == labeled_state_to_delta(right).ops


# ── Public operations ──

def resolve_change(document, change, newline='\n'):
    assert_canonical_document(document, newline)
    return _resolve_change_against_state(labeled_state_from_document(document, newline), change, newline)


def apply_change(document, change, newline='\n'):
    assert_canonical_document(document, newline)
    if not change['blockDelta'].ops:
        return document.compose(change['delta'])
    base_state = labeled_state_from_document(document, newline)
    return labeled_state_to_delta(_final_state_of(_resolve_change_against_state(base_state, change, newline), newline))


def compose_change(document, first, second, newline='\n'):
    assert_canonical_document(document, newline)
    base_state = labeled_state_from_document(document, newline)

    if not first['delta'].ops and not second['delta'].ops:
        return {'delta': Delta(), 'blockDelta': first['blockDelta'].compose(second['blockDelta'], _block_count(base_state, newline))}
    if not first['blockDelta'].ops and not second['blockDelta'].ops:
        return {'delta': first['delta'].compose(second['delta']), 'blockDelta': BlockDelta()}

    first_resolved = _resolve_change_against_state(base_state, first, newline)
    first_final = _final_state_of(first_resolved, newline)
    second_resolved = _resolve_change_against_state(first_final, second, newline)
    sequential_final = _final_state_of(second_resolved, newline)

    if first['blockDelta'].ops and not second['blockDelta'].ops:
        return _delta_only_change(base_state, sequential_final)

    combined_post_delta = replay_resolved_delta(first_resolved['postDeltaState'], second_resolved['resolvedDelta'])
    combined_moves = first_resolved['resolvedMoves'] + second_resolved['resolvedMoves']
    if not combined_moves:
        return _lower_change(base_state, combined_post_delta, sequential_final, newline)

    return _prepare_and_lower(base_state, combined_post_delta, combined_moves, sequential_final, newline)


def transform_change(left, right, document, priority=False, newline='\n'):
    assert_canonical_document(document, newline)
    base_state = labeled_state_from_document(document, newline)

    if not left['blockDelta'].ops and not right['blockDelta'].ops:
        return {'delta': left['delta'].transform(right['delta'], priority), 'blockDelta': BlockDelta()}
    if not left['delta'].ops and not right['delta'].ops:
        return {'delta': Delta(), 'blockDelta': left['blockDelta'].transform(right['blockDelta'], _block_count(base_state, newline), priority)}

    if (left['blockDelta'].ops and right['blockDelta'].ops
            and classify_delta_boundaries(document, left['delta'], newline) == 'block-stable'
            and classify_delta_boundaries(document, right['delta'], newline) == 'block-stable'):
        return {
            'delta': transform_change(left, {'delta': right['delta'], 'blockDelta': BlockDelta()}, document, priority, newline)['delta'],
            'blockDelta': left['blockDelta'].transform(right['blockDelta'], _block_count(base_state, newline), priority),
        }

    if left['blockDelta'].ops and not right['blockDelta'].ops:
        right_final_doc = apply_change(document, right, newline)
        left_prime = transform_change(right, left, document, not priority, newline)
        return {
            'delta': apply_change(document, left, newline).diff(apply_change(right_final_doc, left_prime, newline)),
            'blockDelta': BlockDelta(),
        }

    left_resolved = _resolve_change_against_state(base_state, left, newline)
    left_final = _final_state_of(left_resolved, newline)
    right_resolved = _rekey_resolved_change_against_state(
        _resolve_change_against_state(base_state, right, newline), left_final)
    transformed_post_delta = _replay_resolved_delta_with_insert_edge(
        left_final, right_resolved['resolvedDelta'], 'after' if priority else 'before')

    if not right_resolved['resolvedMoves']:
        return _lower_change(left_final, transformed_post_delta, transformed_post_delta, newline)

    prepared = _prepare_resolved_block_moves(
        left_final, transformed_post_delta, right_resolved['resolvedMoves'], newline,
        lambda _s, _m, _i: 'after' if priority else 'before')
    return _lower_prepared_move_program(prepared, newline)


def invert_change(document, change, newline='\n'):
    assert_canonical_document(document, newline)
    base_state = labeled_state_from_document(document, newline)

    if not change['delta'].ops:
        return {'delta': Delta(), 'blockDelta': change['blockDelta'].invert(_block_count(base_state, newline))}
    if not change['blockDelta'].ops:
        return {'delta': change['delta'].invert(document), 'blockDelta': BlockDelta()}

    resolved = _resolve_change_against_state(base_state, change, newline)
    final_state = _final_state_of(resolved, newline)
    inverse_resolved_delta = resolve_delta_against_state(resolved['postDeltaState'], change['delta'].invert(document))
    inverse_post_delta = replay_resolved_delta(final_state, inverse_resolved_delta)
    inverse_block_delta = _try_lower_block_delta(inverse_post_delta, base_state, newline)
    if inverse_block_delta is None:
        return _delta_only_change(final_state, base_state)

    return _prepare_and_lower(
        final_state, inverse_post_delta,
        resolve_block_delta(inverse_post_delta, inverse_block_delta, newline),
        base_state, newline)
