# Data-driven op tests are now in fixtures/op-length.json, attributes-*.json, op-iterator.json
# Run via test_fixtures.py
#
# The tests below cover Python-specific iterator behavior (reset, __iter__) not in fixtures.

from delta import op
import math


def test_iterator_reset():
    ops = [
        {'insert': 'Hello', 'attributes': {'bold': True}},
        {'retain': 3},
        {'insert': 2, 'attributes': {'src': 'http://quilljs.com/'}},
        {'delete': 4},
    ]

    iterator = op.iterator(ops)
    iterator.next()
    iterator.next()
    iterator.reset()
    assert iterator.index == 0
    assert iterator.offset == 0
    assert iterator.peek() == ops[0]


def test_iterator_for_loop():
    ops = [
        {'insert': 'Hello', 'attributes': {'bold': True}},
        {'retain': 3},
        {'insert': 2, 'attributes': {'src': 'http://quilljs.com/'}},
        {'delete': 4},
    ]

    iterator = op.iterator(ops)
    for operator, next_op in zip(ops, iterator):
        assert operator == next_op


def test_empty_iterator():
    iterator = op.iterator([])
    assert iterator.offset == 0
    assert iterator.index == 0
    assert iterator.ops == []
    assert iterator.has_next() is False
    assert iterator.peek() is None
    assert iterator.peek_length() is math.inf
    assert iterator.peek_type() is 'retain'


def test_type_of_empty():
    assert op.type({}) is None
