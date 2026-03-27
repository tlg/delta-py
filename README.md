# Delta (Python Port)

Python port of the [quill-delta](https://github.com/quilljs/delta) library for rich text operational transformation.

Implements the same OT primitives as the TypeScript version: `compose`, `transform`, `invert`, `diff`, with full support for custom embed handlers and block-level move operations. Both implementations share the same [JSON test fixtures](tests/fixtures/) to guarantee identical behavior.

## Usage

### Delta operations

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

### Block moves

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
from delta import Delta
blocks = project_blocks(Delta().insert('Hello\nWorld\n'))
# [{'delta': Delta([{'insert': 'Hello'}]), 'attributes': {}},
#  {'delta': Delta([{'insert': 'World'}]), 'attributes': {}}]
```

## Tests

```
pytest tests/
```

Test cases in `tests/fixtures/` are shared with the TypeScript implementation. Non-fixture tests cover immutability, callbacks, and Python-specific behavior.
