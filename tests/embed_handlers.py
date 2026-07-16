"""
Test-only embed handler examples, shared with the TypeScript fixtures
(mirrors elium-delta's test/embedHandlers.ts).

Provides the 'delta' handler (nested documents) and the 'table-embed'
handler (rows/columns as whole-block moves plus per-cell nested changes).
"""
import copy
from contextlib import contextmanager

from delta import Delta
from delta import op

# ─────────────────────────────────────────────────────────────────────────────
# Minimal block-move kernel (test-only).
#
# The table embed handler below expresses row/column reorders as whole-block
# moves over a fixed block set, serialized as `blockDelta` ops of the form
# [{"retain": n}, {"move": {"count": c, "before": b}}].  This is example code
# for the fixtures — the library itself stays agnostic of the shape handlers
# choose.
# ─────────────────────────────────────────────────────────────────────────────


def _is_move_op(block_op):
    return isinstance(block_op, dict) and block_op.get('move') is not None


def _validate_block_count(block_count):
    if not isinstance(block_count, int) or block_count < 0:
        raise ValueError(f'invalid block count: {block_count}')


def _validate_move(move, block_count):
    _validate_block_count(block_count)
    for key in ('index', 'count', 'before'):
        if not isinstance(move[key], int) or move[key] < 0:
            raise ValueError(f'invalid move {key}: {move[key]}')
    if move['index'] + move['count'] > block_count:
        raise ValueError(
            f"move source out of range: {move['index']} + {move['count']}"
            f' > {block_count}')
    if move['before'] > block_count:
        raise ValueError(
            f"move destination out of range: {move['before']} > {block_count}")


def _to_ordinals(block_count):
    _validate_block_count(block_count)
    return list(range(block_count))


def _find_anchor_index(items, anchor):
    if anchor not in items:
        raise ValueError('anchor not found in current block order')
    return items.index(anchor)


def _assert_same_items(from_items, to_items):
    if len(from_items) != len(to_items):
        raise ValueError('block orders must have the same length')
    remaining = set(from_items)
    if len(remaining) != len(from_items) or len(set(to_items)) != len(to_items):
        raise ValueError('block orders must contain unique items')
    for item in to_items:
        if item not in remaining:
            raise ValueError('block orders must contain the same items')


def resolve_move(index, count, before):
    return {'index': index, 'count': count, 'before': before}


def normalize_move(move, block_count):
    _validate_move(move, block_count)
    if move['count'] == 0:
        return None
    if move['index'] <= move['before'] <= move['index'] + move['count']:
        return None
    return dict(move)


def apply_move(blocks, move):
    normalized = normalize_move(move, len(blocks))
    if normalized is None:
        return list(blocks)
    index, count, before = (
        normalized['index'], normalized['count'], normalized['before'])
    moved = blocks[index:index + count]
    remaining = blocks[:index] + blocks[index + count:]
    insert_at = before if before < index else before - count
    return remaining[:insert_at] + moved + remaining[insert_at:]


def apply_moves(blocks, moves):
    current = list(blocks)
    for move in moves:
        current = apply_move(current, move)
    return current


def _next_move_cursor(move):
    insert_at = (move['before'] if move['before'] < move['index']
                 else move['before'] - move['count'])
    return insert_at + move['count']


def _resolve_move_intent(base, move):
    """Semantic form of a move: the blocks it captures plus the pre-op
    block it lands before (None appends at the end)."""
    normalized = normalize_move(move, len(base))
    if normalized is None:
        return None
    index, count, before = (
        normalized['index'], normalized['count'], normalized['before'])
    return {
        'moved': base[index:index + count],
        'before_block': None if before == len(base) else base[before],
    }


def _apply_move_intent(current, intent):
    moved = intent['moved']
    remaining = [block for block in current if block not in moved]
    if intent['before_block'] is None:
        return remaining + list(moved)
    insert_at = _find_anchor_index(remaining, intent['before_block'])
    return remaining[:insert_at] + list(moved) + remaining[insert_at:]


def diff_to_moves(from_items, to_items):
    """Deterministic canonicalizer: scan the target order left to right and,
    at each mismatch, extract the next maximal contiguous run from the
    current order that matches the target."""
    _assert_same_items(from_items, to_items)
    working = list(from_items)
    moves = []
    for target_index in range(len(to_items)):
        if working[target_index] == to_items[target_index]:
            continue
        try:
            source_index = working.index(to_items[target_index], target_index)
        except ValueError:
            raise ValueError('target block not found in current block order')
        count = 1
        while (target_index + count < len(to_items)
               and source_index + count < len(working)
               and working[source_index + count] == to_items[target_index + count]):
            count += 1
        move = normalize_move(
            resolve_move(source_index, count, target_index), len(working))
        if move is None:
            continue
        moves.append(move)
        working = apply_move(working, move)
    return moves


def _resolve_block_ops(ops, block_count):
    _validate_block_count(block_count)
    entries = []
    cursor = 0
    current = _to_ordinals(block_count)
    for block_op in ops:
        if _is_move_op(block_op):
            resolved = normalize_move(
                resolve_move(cursor, block_op['move']['count'],
                             block_op['move']['before']),
                len(current))
            if resolved is None:
                continue
            intent = _resolve_move_intent(current, resolved)
            if intent is None:
                continue
            entries.append({'resolved': resolved, 'intent': intent})
            current = _apply_move_intent(current, intent)
            cursor = _next_move_cursor(resolved)
            continue
        retain = block_op.get('retain')
        if not isinstance(retain, int) or retain < 0:
            raise ValueError(f'invalid block retain: {retain}')
        cursor += retain
        if cursor > len(current):
            raise ValueError(
                f'block cursor out of range: {cursor} > {len(current)}')
    return entries


def _replay_move_entries(current, entries):
    working = list(current)
    for entry in entries:
        working = _apply_move_intent(working, entry['intent'])
    return working


class BlockDelta:

    def __init__(self, ops=None):
        if hasattr(ops, 'ops'):
            ops = ops.ops
        self.ops = ops if ops is not None else []

    @classmethod
    def from_moves(cls, moves):
        delta = cls()
        cursor = 0
        for move in moves:
            if move['index'] < cursor:
                raise ValueError(
                    'move sequence is not representable from the current'
                    f" cursor: {move['index']} < {cursor}")
            delta.retain(move['index'] - cursor)
            delta.move(move['count'], move['before'])
            cursor = _next_move_cursor(move)
        return delta.chop()

    def retain(self, length):
        if length <= 0:
            return self
        return self.push({'retain': length})

    def move(self, count, before):
        if count <= 0:
            return self
        return self.push({'move': {'count': count, 'before': before}})

    def push(self, new_op):
        new_op = copy.deepcopy(new_op)
        last_op = self.ops[-1] if self.ops else None
        if (last_op is not None and not _is_move_op(last_op)
                and not _is_move_op(new_op)
                and isinstance(last_op.get('retain'), int)
                and isinstance(new_op.get('retain'), int)):
            last_op['retain'] += new_op['retain']
            return self
        self.ops.append(new_op)
        return self

    def chop(self):
        last_op = self.ops[-1] if self.ops else None
        if (last_op is not None and not _is_move_op(last_op)
                and last_op.get('retain', 0) > 0):
            self.ops.pop()
        return self

    def resolve(self, block_count):
        return [entry['resolved']
                for entry in _resolve_block_ops(self.ops, block_count)]

    def apply(self, blocks):
        return apply_moves(blocks, self.resolve(len(blocks)))

    def transform(self, other, block_count, priority=False):
        base = _to_ordinals(block_count)
        this_applied = self.apply(base)
        other_applied = other.apply(base)
        this_entries = _resolve_block_ops(self.ops, block_count)
        other_entries = _resolve_block_ops(other.ops, block_count)
        if priority:
            final = _replay_move_entries(this_applied, other_entries)
        else:
            final = _replay_move_entries(other_applied, this_entries)
        return BlockDelta.from_moves(diff_to_moves(this_applied, final))


class Change:
    """A change pairs an inline delta with whole-block moves.  The delta
    applies first; block moves address the post-delta block order."""

    def __init__(self, delta, block_delta):
        self.delta = delta
        self.block_delta = block_delta


def _split_line_blocks(document, newline='\n'):
    """Split a canonical document (every line newline-terminated) into one
    Delta per line, each keeping its newline and line attributes."""
    blocks = []
    for line, attributes, _index in document.iter_lines(newline):
        block = Delta(copy.deepcopy(line.ops))
        block.insert(newline, **(attributes or {}))
        blocks.append(block)
    return blocks


def apply_change(document, change, newline='\n'):
    after_delta = document.compose(change.delta)
    if not change.block_delta.ops:
        return after_delta
    result = Delta()
    for block in change.block_delta.apply(_split_line_blocks(after_delta,
                                                             newline)):
        result = result.concat(block)
    return result


def transform_change(left, right, document, priority=False, newline='\n'):
    if not left.block_delta.ops and not right.block_delta.ops:
        return Change(left.delta.transform(right.delta, priority),
                      BlockDelta())
    if not left.delta.ops and not right.delta.ops:
        block_count = len(_split_line_blocks(document, newline))
        return Change(
            Delta(),
            left.block_delta.transform(right.block_delta, block_count,
                                       priority))
    raise ValueError('transform_change: mixed delta/blockDelta transforms'
                     ' are not supported by the test kernel')


# ─────────────────────────────────────────────────────────────────────────────
# Table embed handler.
# ─────────────────────────────────────────────────────────────────────────────

EMPTY_TABLE_LINE = '\n'


def _is_table_patch(value):
    return isinstance(value, dict) and 'base' in value


def _clone_table_data(value):
    return copy.deepcopy(value)


def _parse_cell_identity(identity):
    separator = identity.find(':')
    if separator < 0:
        raise ValueError(f'invalid table cell identity: {identity}')
    return identity[:separator], identity[separator + 1:]


def _axis_items_from_ops(ops):
    return [{'id': axis_op['insert']['id'], 'op': axis_op}
            for axis_op in ops
            if isinstance(axis_op.get('insert'), dict)]


def _axis_doc_from_ops(ops=None):
    items = _axis_items_from_ops(ops or [])
    if not items:
        return Delta().insert(EMPTY_TABLE_LINE)
    doc = Delta()
    for item in items:
        doc.insert(item['id'])
        doc.insert(EMPTY_TABLE_LINE, **(item['op'].get('attributes') or {}))
    return doc


def _axis_ops_from_doc(doc):
    ops = []
    for line, attributes, _index in doc.iter_lines():
        line_id = ''.join(line_op['insert'] for line_op in line.ops
                          if isinstance(line_op.get('insert'), str))
        if not line_id:
            continue
        axis_op = {'insert': {'id': line_id}}
        if attributes:
            axis_op['attributes'] = attributes
        ops.append(axis_op)
    return ops


def _canonical_cell_doc(ops=None):
    doc = Delta(copy.deepcopy(ops or []))
    last = doc.ops[-1] if doc.ops else None
    if (last is None or not isinstance(last.get('insert'), str)
            or not last['insert'].endswith(EMPTY_TABLE_LINE)):
        doc.insert(EMPTY_TABLE_LINE)
    return doc


def _cell_doc_to_ops(doc):
    length = doc.length()
    if length == 0:
        return []
    return doc[0:max(0, length - 1)].ops


def _compact_cell_data(content, attributes):
    data = {}
    ops = _cell_doc_to_ops(content)
    if ops:
        data['content'] = ops
    if attributes:
        data['attributes'] = attributes
    return data or None


def _compact_table_data(rows, columns, cells):
    data = {}
    if rows:
        data['rows'] = rows
    if columns:
        data['columns'] = columns
    if cells:
        data['cells'] = cells
    return data


def _change_from_spec(spec=None):
    spec = spec or {}
    return Change(Delta(copy.deepcopy(spec.get('delta') or [])),
                  BlockDelta(copy.deepcopy(spec.get('blockDelta') or [])))


def _change_spec_from_change(change):
    if not change.delta.ops and not change.block_delta.ops:
        return None
    spec = {}
    if change.delta.ops:
        spec['delta'] = change.delta.ops
    if change.block_delta.ops:
        spec['blockDelta'] = change.block_delta.ops
    return spec


def _apply_axis_change(base_ops, spec=None):
    if not spec:
        return [dict(item['op']) for item in _axis_items_from_ops(base_ops)]
    doc = apply_change(_axis_doc_from_ops(base_ops), _change_from_spec(spec))
    return _axis_ops_from_doc(doc)


def _apply_cell_patch(base_cell, patch):
    base_cell = base_cell or {}
    patch = patch or {}
    content = _canonical_cell_doc(base_cell.get('content') or [])
    if patch.get('change'):
        content = apply_change(content, _change_from_spec(patch['change']))
    attributes = op.compose(base_cell.get('attributes'),
                            patch.get('attributes'), False)
    return _compact_cell_data(content, attributes)


def _diff_cell_patch(base_cell, target_cell):
    base_cell = base_cell or {}
    target_cell = target_cell or {}
    content_change = _change_spec_from_change(Change(
        _canonical_cell_doc(base_cell.get('content') or []).diff(
            _canonical_cell_doc(target_cell.get('content') or [])),
        BlockDelta()))
    attributes = op.diff(base_cell.get('attributes'),
                         target_cell.get('attributes'))
    if not content_change and not attributes:
        return None
    patch = {}
    if content_change:
        patch['change'] = content_change
    if attributes:
        patch['attributes'] = attributes
    return patch


def _has_unique_axis_ids(ops):
    ids = [item['id'] for item in _axis_items_from_ops(ops)]
    return len(set(ids)) == len(ids)


def _same_axis_shape(base_ops, target_ops):
    base = _axis_items_from_ops(base_ops)
    target = _axis_items_from_ops(target_ops)
    if len(base) != len(target):
        return False
    base_ids = [item['id'] for item in base]
    target_ids = [item['id'] for item in target]
    if (len(set(base_ids)) != len(base_ids)
            or len(set(target_ids)) != len(target_ids)):
        return False
    if sorted(base_ids) != sorted(target_ids):
        return False
    target_by_id = {item['id']: item['op'] for item in target}
    return all(
        item['op'].get('attributes') ==
        target_by_id[item['id']].get('attributes')
        for item in base)


def _build_post_delta_axis_ops(base_ops, target_ops):
    if not _has_unique_axis_ids(base_ops) or not _has_unique_axis_ids(target_ops):
        return None
    base = _axis_items_from_ops(base_ops)
    target = _axis_items_from_ops(target_ops)
    target_by_id = {item['id']: item['op'] for item in target}
    shared_ids = {item['id'] for item in base if item['id'] in target_by_id}
    inserted_before = {}
    pending_inserted = []

    for item in target:
        if item['id'] in shared_ids:
            inserted_before[item['id']] = (
                inserted_before.get(item['id'], []) + pending_inserted)
            pending_inserted = []
            continue
        pending_inserted.append(item['op'])
    inserted_before[None] = pending_inserted

    post_delta = []
    for item in base:
        if item['id'] not in shared_ids:
            continue
        post_delta.extend(inserted_before.get(item['id'], []))
        post_delta.append(target_by_id[item['id']])
    post_delta.extend(inserted_before.get(None, []))
    return post_delta


def _diff_axis_change(base_ops, target_ops):
    base = _axis_items_from_ops(base_ops)
    target = _axis_items_from_ops(target_ops)
    if _same_axis_shape(base_ops, target_ops):
        moves = diff_to_moves([item['id'] for item in base],
                              [item['id'] for item in target])
        return _change_spec_from_change(
            Change(Delta(), BlockDelta.from_moves(moves)))
    post_delta_ops = _build_post_delta_axis_ops(base_ops, target_ops)
    if post_delta_ops is not None:
        return _change_spec_from_change(Change(
            _axis_doc_from_ops(base_ops).diff(
                _axis_doc_from_ops(post_delta_ops)),
            BlockDelta.from_moves(diff_to_moves(
                [item['id'] for item in _axis_items_from_ops(post_delta_ops)],
                [item['id'] for item in target]))))
    return _change_spec_from_change(Change(
        _axis_doc_from_ops(base_ops).diff(_axis_doc_from_ops(target_ops)),
        BlockDelta()))


def _filter_cells_by_axes(cells, rows, columns):
    row_ids = {item['id'] for item in _axis_items_from_ops(rows)}
    column_ids = {item['id'] for item in _axis_items_from_ops(columns)}
    filtered = {}
    for identity, cell in cells.items():
        row_id, column_id = _parse_cell_identity(identity)
        if row_id in row_ids and column_id in column_ids:
            filtered[identity] = cell
    return filtered


def _compact_table_patch(patch):
    compact = {'base': _clone_table_data(patch['base'])}
    if patch.get('rows'):
        compact['rows'] = patch['rows']
    if patch.get('columns'):
        compact['columns'] = patch['columns']
    if patch.get('cells'):
        compact['cells'] = patch['cells']
    return compact


def _apply_table_patch(base, patch):
    rows = _apply_axis_change(base.get('rows') or [], patch.get('rows'))
    columns = _apply_axis_change(base.get('columns') or [],
                                 patch.get('columns'))
    cells = _filter_cells_by_axes(dict(base.get('cells') or {}), rows, columns)
    for identity, cell_patch in (patch.get('cells') or {}).items():
        row_id, column_id = _parse_cell_identity(identity)
        valid_rows = {item['id'] for item in _axis_items_from_ops(rows)}
        valid_columns = {item['id'] for item in _axis_items_from_ops(columns)}
        if row_id not in valid_rows or column_id not in valid_columns:
            cells.pop(identity, None)
            continue
        cell = _apply_cell_patch((base.get('cells') or {}).get(identity),
                                 cell_patch)
        if cell:
            cells[identity] = cell
        else:
            cells.pop(identity, None)
    return _compact_table_data(rows, columns, cells)


def _diff_table(base, target):
    rows = _diff_axis_change(base.get('rows') or [], target.get('rows') or [])
    columns = _diff_axis_change(base.get('columns') or [],
                                target.get('columns') or [])
    cells = {}
    identities = dict.fromkeys(
        list(base.get('cells') or {}) + list(target.get('cells') or {}))
    for identity in identities:
        patch = _diff_cell_patch((base.get('cells') or {}).get(identity),
                                 (target.get('cells') or {}).get(identity))
        if patch:
            cells[identity] = patch
    return _compact_table_patch({
        'base': base,
        'rows': rows,
        'columns': columns,
        'cells': cells,
    })


def _transform_nested_change(left, right, document, priority):
    if not right:
        return None
    if not left:
        return right
    return _change_spec_from_change(transform_change(
        _change_from_spec(left), _change_from_spec(right), document,
        priority))


def _transform_cell_attributes(left, right, priority):
    return op.transform(left, right, priority)


def _compose_table_patch(left, right):
    base = _clone_table_data(left['base'])
    middle = _apply_table_patch(base, left)
    final = _apply_table_patch(middle, right)
    return _diff_table(base, final)


def _transform_table_patch(applied, other, priority):
    base = _clone_table_data(applied['base'])
    base_after_applied = _apply_table_patch(base, applied)

    rows_prime = _transform_nested_change(
        applied.get('rows'), other.get('rows'),
        _axis_doc_from_ops(base.get('rows') or []), priority)
    columns_prime = _transform_nested_change(
        applied.get('columns'), other.get('columns'),
        _axis_doc_from_ops(base.get('columns') or []), priority)
    cells = {}
    identities = dict.fromkeys(
        list(base.get('cells') or {})
        + list(applied.get('cells') or {})
        + list(other.get('cells') or {}))

    for identity in identities:
        base_cell = (base.get('cells') or {}).get(identity) or {}
        applied_patch = (applied.get('cells') or {}).get(identity) or {}
        other_patch = (other.get('cells') or {}).get(identity) or {}
        change = _transform_nested_change(
            applied_patch.get('change'), other_patch.get('change'),
            _canonical_cell_doc(base_cell.get('content') or []), priority)
        attributes = _transform_cell_attributes(
            applied_patch.get('attributes'), other_patch.get('attributes'),
            priority)
        if change or attributes:
            cells[identity] = {}
            if change:
                cells[identity]['change'] = change
            if attributes:
                cells[identity]['attributes'] = attributes

    final = _apply_table_patch(base_after_applied, _compact_table_patch({
        'base': base_after_applied,
        'rows': rows_prime,
        'columns': columns_prime,
        'cells': cells,
    }))
    return _diff_table(base_after_applied, final)


def _invert_table_patch(change, base):
    final = _apply_table_patch(base, change)
    return _diff_table(final, base)


class TableHandler:

    @staticmethod
    def compose(a, b, _keep_null=False):
        if not _is_table_patch(b):
            raise ValueError('table-embed compose expects a table patch as'
                             ' the right operand')
        if _is_table_patch(a):
            return _compose_table_patch(a, b)
        return _apply_table_patch(a, b)

    @staticmethod
    def diff(a, b):
        return _diff_table(a, b)

    @staticmethod
    def transform(a, b, priority=False):
        if not _is_table_patch(a) or not _is_table_patch(b):
            raise ValueError('table-embed transform expects self-contained'
                             ' table patches')
        return _transform_table_patch(a, b, priority)

    @staticmethod
    def invert(change, base):
        if not _is_table_patch(change) or _is_table_patch(base):
            raise ValueError('table-embed invert expects change patch over'
                             ' a table document')
        return _invert_table_patch(change, base)


class DeltaHandler:

    @staticmethod
    def compose(a, b, *_args):
        return Delta(a).compose(Delta(b)).ops

    @staticmethod
    def diff(a, b):
        return Delta(a).diff(Delta(b)).ops

    @staticmethod
    def transform(a, b, priority=False):
        return Delta(a).transform(Delta(b), priority).ops

    @staticmethod
    def invert(a, b):
        return Delta(a).invert(Delta(b)).ops


def register_delta_embed():
    Delta.register_embed('delta', DeltaHandler)


def unregister_delta_embed():
    Delta.unregister_embed('delta')


def register_table_embed():
    Delta.register_embed('table-embed', TableHandler)


def unregister_table_embed():
    Delta.unregister_embed('table-embed')


_HANDLERS = {
    'delta': (register_delta_embed, unregister_delta_embed),
    'table-embed': (register_table_embed, unregister_table_embed),
}


def register_embed_handler(name):
    try:
        register = _HANDLERS[name][0]
    except KeyError:
        raise ValueError(f'unknown test embed handler: {name}')
    register()


def unregister_embed_handler(name):
    try:
        unregister = _HANDLERS[name][1]
    except KeyError:
        raise ValueError(f'unknown test embed handler: {name}')
    unregister()


@contextmanager
def registered_embed_handler(name):
    register_embed_handler(name)
    try:
        yield
    finally:
        unregister_embed_handler(name)
