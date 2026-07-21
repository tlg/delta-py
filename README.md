# elium-delta

Rich text OT deltas with **semantic cut/paste moves**, in the
[quill-delta](https://github.com/quilljs/delta) lineage.

The package implements the full delta algebra — `compose`, `transform`,
`invert`, `diff`, custom embed handlers — plus two operation types that
express *moving* content without buffering it:

```python
{'cut':   {'ref': 'r', 'length': 10}}
{'paste': {'ref': 'r', 'start': 0, 'length': 10}}
{'paste': {'ref': 'e', 'start': 0, 'length': 1,
           'change': {'cell': {'ops': [{'delete': 2}]}}}}
```

The corresponding builders are `cut(ref, length)` and
`paste(ref, start, length, change=None, **attributes)`. `start` is an
offset in the captured transfer interval, not a document position. A
`change` is valid only for a length-one paste: it patches the anonymous
embed after the moved value reaches its destination. Paste attributes
remain ordinary top-level Delta attributes.

A paste addresses its cut span positionally, so deltas stay closed under
composition: an insert into a moved region splits the window, a delete
shrinks it, a format becomes the paste's attribute patch, and the inverse
of a move is the opposite move. Moves cross embed nesting levels
recursively (table cells in table cells), may target freshly inserted
embeds, and deleting an embed that still sources a move degrades to a
trash-bin read (Davis, Sun & Lu 2002). The compact implementation lives
in `delta/base.py`.

Handlers that own nested Delta streams expose their paths structurally
and recurse with the same operation context:

```python
class CellHandler:
    @staticmethod
    def stream_paths(value):
        return [("ops",)] if isinstance(value.get("ops"), list) else []

    @staticmethod
    def apply(value, change, context):
        return {
            "ops": Delta(value["ops"]).compose(
                Delta(change["ops"]), _context=context
            ).ops
        }

    compose = apply

    @staticmethod
    def transform(first, second, priority, context):
        return {
            "ops": Delta(first["ops"]).transform(
                Delta(second["ops"]), priority, _context=context
            ).ops
        }

    @staticmethod
    def invert(change, base, context):
        return {
            "ops": Delta(change["ops"]).invert(
                Delta(base["ops"]), _context=context
            ).ops
        }

    @staticmethod
    def diff(base, target, context):
        return {
            "ops": Delta(base["ops"]).diff(
                Delta(target["ops"]), _context=context
            ).ops
        }


Delta.register_embed("cell", CellHandler)
```

Paths are relative to the handler payload; `()` denotes a payload that
is itself an operation list. Only declared streams participate in move
validation, reference renaming and cross-level routing. Other embed data
remains opaque even when it contains cut/paste-shaped dictionaries.

The embed handler contract keeps values and changes distinct:

```python
class EmbedHandler:
    def apply(value, change, context): ...
    def compose(first, second, context): ...
    def transform(first, second, priority, context): ...
    def invert(change, base, context): ...
    def diff(base, target, context): ...
```

`stream_paths` is optional when a handler owns no nested operation
streams. The five algebra methods are the handler contract. Transaction
contexts are opaque and operation-specific. A handler recursing into a
child Delta forwards the same context by keyword and must not pass it to
a different operation.

All offsets count UTF-16 code units, matching JavaScript string
semantics — astral characters (most emoji) count 2, and boundaries may
split surrogate pairs exactly as JavaScript allows.

`Delta(ops)`, `Delta(other_delta)`, `push()` and `extend()` take safe
ownership by cloning every operation dictionary and deep-copying nested
attributes and embed payloads. Callers may therefore mutate their source
values afterwards without changing the Delta. Internally,
`_from_owned_ops()` is the explicit no-copy path for a freshly allocated
list whose ownership is being transferred; it is not a public
construction API.

Requires Python 3.14+.

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

# Move one anonymous cell embed and edit its nested stream on arrival
move_and_edit = Delta([
    {'retain': 2},
    {'cut': {'ref': 'e', 'length': 1}},
    {'retain': 1},
    {'paste': {'ref': 'e', 'start': 0, 'length': 1,
               'change': {'cell': {'ops': [{'delete': 2}]}}}},
])

# Transform concurrent edits (moves rebase; deletes follow content)
a = Delta().insert('A')
b = Delta().insert('B')
b_prime = a.transform(b, priority=True)

# Invert a change — the inverse of a move is a move
inverse = move.invert(doc)
assert doc.compose(move).compose(inverse) == doc
```

The same reference table is shared through recursive handlers, so one
cell can cut content that another cell pastes—even when structural
traversal encounters the paste first:

```python
def cell(text):
    return {'cell': {'ops': [{'insert': text}]}}


base = Delta().insert(cell('one')).insert(cell('two'))
cross_cell = Delta([
    {'retain': {'cell': {'ops': [
        {'retain': 3},
        {'paste': {'ref': 'x', 'start': 0, 'length': 3}},
    ]}}},
    {'retain': {'cell': {'ops': [
        {'cut': {'ref': 'x', 'length': 3}},
    ]}}},
])

assert base.compose(cross_cell) == (
    Delta().insert(cell('onetwo')).insert({'cell': {'ops': []}})
)
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
