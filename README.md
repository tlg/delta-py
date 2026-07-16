# elium-delta

Rich text OT deltas with **semantic cut/paste moves**, in the
[quill-delta](https://github.com/quilljs/delta) lineage.

The package implements the full delta algebra — `compose`, `transform`,
`invert`, `diff`, custom embed handlers — plus two operation types that
express *moving* content without buffering it:

```python
{'cut':   {'ref': 'r', 'length': 10}}
{'paste': {'ref': 'r', 'start': 0, 'length': 10}}
```

A paste addresses its cut span positionally, so deltas stay closed under
composition: an insert into a moved region splits the window, a delete
shrinks it, a format becomes the paste's attribute patch, and the inverse
of a move is the opposite move. Moves cross embed nesting levels
recursively (table cells in table cells), may target freshly inserted
embeds, and deleting an embed that still sources a move degrades to a
trash-bin read (Davis, Sun & Lu 2002). See the `delta/moves.py` module
docstring for the complete semantics and the documented boundaries.

All offsets count UTF-16 code units, matching JavaScript string
semantics — astral characters (most emoji) count 2, and boundaries may
split surrogate pairs exactly as JavaScript allows.

Requires Python 3.12+.

## Usage

```python
from delta import Delta

# Build a document
doc = Delta().insert('Hello ', bold=True).insert('World\n')

# Apply a change
change = Delta().retain(6).insert('Beautiful ')
result = doc.compose(change)

# Move: cut "Hello " to the end of the line
move = Delta().cut('m', 6).retain(5).paste('m', 0, 6)

# Transform concurrent edits (moves rebase; deletes follow content)
a = Delta().insert('A')
b = Delta().insert('B')
b_prime = a.transform(b, priority=True)

# Invert a change — the inverse of a move is a move
inverse = move.invert(doc)
assert doc.compose(move).compose(inverse) == doc
```

Nested-document coordinates transform through moves, across levels:

```python
from delta import transform_coordinate

transform_coordinate(delta, (2, 'ops', 3))  # caret inside an embed
```

## Tests

```
python -m pytest
```

The suite combines the shared [JSON fixtures](tests/fixtures/) from the
TypeScript upstream with seeded fuzzing of the compose/transform/invert
laws, including closure over compose outputs.
