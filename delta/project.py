"""
Block span projection from documents and labeled states.
"""
from .labeled_state import labeled_state_from_document


def _is_newline_unit(value, newline):
    return isinstance(value, str) and value == newline


def project_labeled_block_spans(state, newline='\n'):
    spans = []
    from_idx = 0
    for i, unit in enumerate(state['units']):
        if not _is_newline_unit(unit['value'], newline):
            continue
        spans.append({
            'from': from_idx,
            'to': i + 1,
            'leftGap': state['gaps'][from_idx],
            'rightGap': state['gaps'][i + 1],
        })
        from_idx = i + 1
    if from_idx != len(state['units']):
        raise ValueError('labeled state must end with a final newline boundary')
    return spans


def project_block_spans(document, newline='\n'):
    spans = project_labeled_block_spans(
        labeled_state_from_document(document, newline), newline)
    return [{'from': s['from'], 'to': s['to']} for s in spans]


def block_boundaries(document, newline='\n'):
    spans = project_block_spans(document, newline)
    if not spans:
        return [0]
    return [spans[0]['from']] + [s['to'] for s in spans]


def block_boundary_gap_anchors(state, newline='\n'):
    spans = project_labeled_block_spans(state, newline)
    if not spans:
        return state['gaps'][:1]
    return [spans[0]['leftGap']] + [s['rightGap'] for s in spans]
