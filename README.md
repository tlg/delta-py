# Delta (Python Port)

Python port of the [quill-delta](https://github.com/quilljs/delta) library for rich text operational transformation.

Implements the same OT primitives as the TypeScript version: `compose`, `transform`, `invert`, `diff`, with full support for custom embed handlers. Both implementations share the same [JSON test fixtures](tests/fixtures/) to guarantee identical behavior.

## Usage

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
```

## Tests

```
pytest tests/
```

Test cases in `tests/fixtures/` are shared with the TypeScript implementation. Non-fixture tests cover immutability and callback behavior.
