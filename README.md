# elium-delta-py

Python port of [elium-delta](https://github.com/whatever-company/elium-delta), a fork of [quill-delta](https://github.com/quilljs/delta) extended with block-level move operations and a two-stream change bridge.

Implements the same OT primitives as the TypeScript version — `compose`, `transform`, `invert`, `diff` — with full support for custom embed handlers, block-level move operations, and the two-stream change bridge. Both implementations share the same [JSON test fixtures](tests/fixtures/) to guarantee identical behavior, including UTF-16 code unit length semantics.

## Install

```
pip install diff-match-patch
```

## Delta operations

Delta represents a document or a change to a document as a sequence of operations: `insert`, `retain`, and `delete`.

```python
from delta import Delta

# Build a document
doc = Delta().insert('Hello ', bold=True).insert('World\n')

# Apply a change
change = Delta().retain(6).insert('Beautiful ')
result = doc.compose(change)

# Transform concurrent edits
a = Delta().insert('A')
b = Delta().insert('B')
b_prime = a.transform(b, priority=True)

# Invert a change
inverse = change.invert(doc)
assert doc.compose(change).compose(inverse) == doc

# Diff two documents
a = Delta().insert('Hello')
b = Delta().insert('Hello World')
assert a.diff(b) == Delta().retain(5).insert(' World')
```

### UTF-16 length semantics

All string lengths and positions use UTF-16 code units to match JavaScript. Characters above U+FFFF (emoji, etc.) count as 2 units:

```python
from delta.op import utf16_len

assert utf16_len('abc') == 3
assert utf16_len('😀') == 2   # surrogate pair

# retain(2) covers one emoji, same as in JS
doc = Delta().insert('😀\n')
bold = Delta().retain(2).retain(1, bold=True)
result = doc.compose(bold)
# [{'insert': '😀'}, {'insert': '\n', 'attributes': {'bold': True}}]
```

## Block moves

`BlockDelta` models whole-block reordering over a document's line structure. It supports the same OT operations as `Delta` — compose, transform, and invert — but operates on block indices rather than character positions.

```python
from delta import BlockDelta

# Move block at index 1 to position 0 (swap first two blocks)
bd = BlockDelta().retain(1).move(1, 0)
assert bd.apply(['A', 'B', 'C']) == ['B', 'A', 'C']

# Compose two block reorders
a = BlockDelta().retain(1).move(2, 5)
b = BlockDelta().retain(3).move(2, 0)
composed = a.compose(b, block_count=5)

# Transform concurrent block moves
a = BlockDelta().retain(1).move(1, 0)
b = BlockDelta().retain(3).move(1, 0)
b_prime = a.transform(b, block_count=5, priority=True)

# Invert a block reorder
inverted = bd.invert(block_count=3)
assert inverted.apply(bd.apply(['A', 'B', 'C'])) == ['A', 'B', 'C']
```

Lower-level move functions are also available:

```python
from delta.block import (
    apply_move, apply_moves, compose_moves, diff_to_moves,
    invert_move, resolve_move, transform_move, project_blocks,
)

# Compute the move sequence between two block orderings
moves = diff_to_moves(['A', 'B', 'C'], ['C', 'A', 'B'])
assert apply_moves(['A', 'B', 'C'], moves) == ['C', 'A', 'B']

# Project a document into blocks
blocks = project_blocks(Delta().insert('Hello\nWorld\n'))
# [{'delta': Delta([{'insert': 'Hello'}]), 'attributes': {}},
#  {'delta': Delta([{'insert': 'World'}]), 'attributes': {}}]
```

## Changes (delta + block moves)

A Change combines a text delta with a block reorder, applied together atomically. The change bridge resolves both streams against a labeled unit-level state model with gap anchors, ensuring block moves and text edits compose correctly even when they interact with the same boundaries.

```python
from delta import Delta, BlockDelta
from delta.change import apply_change, compose_change, transform_change, invert_change

base = Delta().insert('A\nB\nC\n')

# Apply: delta runs first, then blockDelta reorders the resulting blocks
change = {
    'delta': Delta().retain(1).insert('x'),
    'blockDelta': BlockDelta().retain(1).move(1, 3),
}
result = apply_change(base, change)
assert result == Delta().insert('Ax\nC\nB\n')

# Compose sequential changes
first = {
    'delta': Delta().retain(1).insert('x'),
    'blockDelta': BlockDelta().retain(1).move(1, 3),
}
second = {
    'delta': Delta().retain(6).insert('z'),
    'blockDelta': BlockDelta().retain(1).move(1, 0),
}
composed = compose_change(base, first, second)
assert apply_change(base, composed) == Delta().insert('C\nAx\nBz\n')

# Transform concurrent changes
left = {'delta': Delta(), 'blockDelta': BlockDelta().retain(1).move(1, 0)}
right = {'delta': Delta().retain(3).insert('x'), 'blockDelta': BlockDelta()}
right_prime = transform_change(left, right, base, priority=True)
assert apply_change(apply_change(base, left), right_prime) == Delta().insert('Bx\nA\nC\n')

# Invert a change
inverse = invert_change(base, change)
assert apply_change(apply_change(base, change), inverse) == base
```

## Supporting modules

### Labeled state

The labeled state model assigns unique IDs to every character-level unit and tracks gap anchors between them. This is the foundation for resolving concurrent edits at the unit level.

```python
from delta.labeled_state import (
    labeled_state_from_document,
    labeled_state_to_delta,
    resolve_delta_against_state,
    replay_resolved_delta,
)

state = labeled_state_from_document(Delta().insert('AB\n'))
resolved = resolve_delta_against_state(state, Delta().retain(1).insert('X'))
new_state = replay_resolved_delta(state, resolved)
assert labeled_state_to_delta(new_state) == Delta().insert('AXB\n')
```

### Boundary classifier

Classifies how a delta interacts with block boundaries, used by the change transform to decide whether moves can be preserved or must fall back to delta-only.

```python
from delta.boundary_classifier import classify_delta_boundaries

base = Delta().insert('A\nB\n')
assert classify_delta_boundaries(base, Delta().retain(1).insert('x')) == 'block-stable'
assert classify_delta_boundaries(base, Delta().retain(2).insert('X\n')) == 'whole-line structural'
assert classify_delta_boundaries(base, Delta().retain(1).insert('\n')) == 'split/merge'
```

### Project

Projects documents and labeled states into block spans with boundary information.

```python
from delta.project import project_block_spans, block_boundaries

doc = Delta().insert('Hello\nWorld\n')
assert project_block_spans(doc) == [{'from': 0, 'to': 6}, {'from': 6, 'to': 12}]
assert block_boundaries(doc) == [0, 6, 12]
```

## Tests

```
pytest tests/
```

Test cases in `tests/fixtures/` are shared with the TypeScript [elium-delta](https://github.com/whatever-company/elium-delta) implementation. Non-fixture tests cover immutability, callbacks, and Python-specific behavior.
