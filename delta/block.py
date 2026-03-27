"""
Block-level move operations over a fixed set of blocks.

Models whole-block reorders only. Structural block semantics (splits, merges,
boundary restoration) live in the change layer, not this move-only kernel.
"""
import copy


def _validate_block_count(block_count):
    if not isinstance(block_count, int) or block_count < 0:
        raise ValueError(f'invalid block count: {block_count}')


def _validate_move(op, block_count):
    _validate_block_count(block_count)
    if not isinstance(op['index'], int) or op['index'] < 0:
        raise ValueError(f'invalid move index: {op["index"]}')
    if not isinstance(op['count'], int) or op['count'] < 0:
        raise ValueError(f'invalid move count: {op["count"]}')
    if not isinstance(op['before'], int) or op['before'] < 0:
        raise ValueError(f'invalid move destination: {op["before"]}')
    if op['index'] + op['count'] > block_count:
        raise ValueError(
            f'move source out of range: {op["index"]} + {op["count"]} > {block_count}')
    if op['before'] > block_count:
        raise ValueError(
            f'move destination out of range: {op["before"]} > {block_count}')


def _to_ordinals(block_count):
    _validate_block_count(block_count)
    return list(range(block_count))


def _find_anchor_index(items, anchor):
    try:
        return items.index(anchor)
    except ValueError:
        raise ValueError('anchor not found in current block order')


def _assert_same_items(from_list, to_list):
    if len(from_list) != len(to_list):
        raise ValueError('block orders must have the same length')
    if len(set(from_list)) != len(from_list) or len(set(to_list)) != len(to_list):
        raise ValueError('block orders must contain unique items')
    if set(from_list) != set(to_list):
        raise ValueError('block orders must contain the same items')


def _is_move_op(op):
    return 'move' in op


def _next_move_cursor(op):
    insert_at = op['before'] if op['before'] < op['index'] else op['before'] - op['count']
    return insert_at + op['count']


def resolve_move(index, count, before):
    """Create a resolved move descriptor."""
    return {'index': index, 'count': count, 'before': before}


def normalize_move(op, block_count):
    """Normalize a move, returning None if it's a no-op."""
    _validate_move(op, block_count)
    if op['count'] == 0:
        return None
    if op['before'] >= op['index'] and op['before'] <= op['index'] + op['count']:
        return None
    return {'index': op['index'], 'count': op['count'], 'before': op['before']}


def apply_move(blocks, op):
    """Apply a single move to a block list, returning a new list."""
    normalized = normalize_move(op, len(blocks))
    if normalized is None:
        return blocks[:]
    idx = normalized['index']
    count = normalized['count']
    before = normalized['before']
    moved = blocks[idx:idx + count]
    remaining = blocks[:idx] + blocks[idx + count:]
    insert_at = before if before < idx else before - count
    return remaining[:insert_at] + moved + remaining[insert_at:]


def apply_moves(blocks, ops):
    """Apply a sequence of moves to a block list."""
    current = blocks[:]
    for op in ops:
        current = apply_move(current, op)
    return current


def _resolve_move_intent(base, op):
    normalized = normalize_move(op, len(base))
    if normalized is None:
        return None
    idx = normalized['index']
    count = normalized['count']
    before = normalized['before']
    return {
        'moved': base[idx:idx + count],
        'before_block': None if before == len(base) else base[before],
    }


def _apply_move_intent(current, intent):
    source_set = set(intent['moved'])
    remaining = [b for b in current if b not in source_set]
    if intent['before_block'] is None:
        return remaining + intent['moved']
    insert_at = _find_anchor_index(remaining, intent['before_block'])
    return remaining[:insert_at] + intent['moved'] + remaining[insert_at:]


def diff_to_moves(from_list, to_list):
    """Compute the canonical move sequence to transform from_list into to_list."""
    _assert_same_items(from_list, to_list)
    working = from_list[:]
    ops = []
    for target_index in range(len(to_list)):
        if working[target_index] == to_list[target_index]:
            continue
        source_index = working.index(to_list[target_index])
        count = 1
        while (target_index + count < len(to_list)
               and source_index + count < len(working)
               and working[source_index + count] == to_list[target_index + count]):
            count += 1
        op = normalize_move(
            resolve_move(source_index, count, target_index),
            len(working))
        if op is None:
            continue
        ops.append(op)
        working[:] = apply_move(working, op)
    return ops


def invert_move(op, block_count):
    """Compute the inverse move(s) that undo op."""
    base = _to_ordinals(block_count)
    intent = _resolve_move_intent(base, op)
    final = base if intent is None else _apply_move_intent(base, intent)
    return diff_to_moves(final, base)


def compose_moves(a, b, block_count):
    """Compose two moves into a minimal move sequence."""
    base = _to_ordinals(block_count)
    a_intent = _resolve_move_intent(base, a)
    after_a = base if a_intent is None else _apply_move_intent(base, a_intent)
    b_intent = _resolve_move_intent(after_a, b)
    final = after_a if b_intent is None else _apply_move_intent(after_a, b_intent)
    return diff_to_moves(base, final)


def transform_move(a, b, priority, block_count):
    """Transform move b against move a with the given priority."""
    base = _to_ordinals(block_count)
    a_intent = _resolve_move_intent(base, a)
    b_intent = _resolve_move_intent(base, b)
    after_a = base if a_intent is None else _apply_move_intent(base, a_intent)
    after_b = base if b_intent is None else _apply_move_intent(base, b_intent)
    if priority or a_intent is None:
        final = after_a if b_intent is None else _apply_move_intent(after_a, b_intent)
    else:
        final = _apply_move_intent(after_b, a_intent)
    return diff_to_moves(after_a, final)


def project_blocks(document, newline='\n'):
    """Project a document Delta into a list of {delta, attributes} blocks."""
    from .base import Delta
    blocks = []
    for line, attributes, _index in document.iter_lines(newline):
        blocks.append({
            'delta': Delta(copy.deepcopy(line.ops)),
            'attributes': copy.deepcopy(attributes or {}),
        })
    return blocks


def _resolve_block_ops(ops, block_count):
    _validate_block_count(block_count)
    entries = []
    cursor = 0
    current = _to_ordinals(block_count)
    for op in ops:
        if _is_move_op(op):
            move = op['move']
            resolved = normalize_move(
                resolve_move(cursor, move['count'], move['before']),
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
        if not isinstance(op.get('retain'), int) or op['retain'] < 0:
            raise ValueError(f'invalid block retain: {op.get("retain")}')
        cursor += op['retain']
        if cursor > len(current):
            raise ValueError(
                f'block cursor out of range: {cursor} > {len(current)}')
    return entries


def _replay_move_entries(current, entries):
    working = current[:]
    for entry in entries:
        working = _apply_move_intent(working, entry['intent'])
    return working


class BlockDelta:
    """A delta over block ordering (retains and moves only)."""

    def __init__(self, ops=None):
        if isinstance(ops, list):
            self.ops = ops
        elif ops is not None and hasattr(ops, 'ops'):
            self.ops = ops.ops
        else:
            self.ops = []

    def __eq__(self, other):
        if not isinstance(other, BlockDelta):
            return NotImplemented
        return self.ops == other.ops

    def __repr__(self):
        return f'BlockDelta({self.ops})'

    @staticmethod
    def from_moves(moves):
        delta = BlockDelta()
        cursor = 0
        for move in moves:
            if move['index'] < cursor:
                raise ValueError(
                    f'move sequence is not representable from the current cursor: '
                    f'{move["index"]} < {cursor}')
            delta.retain(move['index'] - cursor).move(move['count'], move['before'])
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
        op = copy.deepcopy(new_op)
        if (self.ops
                and not _is_move_op(self.ops[-1])
                and not _is_move_op(op)
                and isinstance(self.ops[-1].get('retain'), int)
                and isinstance(op.get('retain'), int)):
            self.ops[-1]['retain'] += op['retain']
            return self
        self.ops.append(op)
        return self

    def chop(self):
        if self.ops and not _is_move_op(self.ops[-1]) and self.ops[-1].get('retain', 0) > 0:
            self.ops.pop()
        return self

    def resolve(self, block_count):
        return [e['resolved'] for e in _resolve_block_ops(self.ops, block_count)]

    def apply(self, blocks):
        return apply_moves(blocks, self.resolve(len(blocks)))

    def compose(self, other, block_count):
        base = _to_ordinals(block_count)
        final = other.apply(self.apply(base))
        return BlockDelta.from_moves(diff_to_moves(base, final))

    def invert(self, block_count):
        base = _to_ordinals(block_count)
        return BlockDelta.from_moves(diff_to_moves(self.apply(base), base))

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
