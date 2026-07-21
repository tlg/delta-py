# Data-driven op tests are now in fixtures/op-length.json, attributes-*.json, op-iterator.json
# Run via test_fixtures.py
#
# The tests below cover Python-specific iterator behavior (reset, __iter__) not in fixtures.

from delta import op
import math


def test_iterator_reset():
    ops = [
        {"insert": "Hello", "attributes": {"bold": True}},
        {"retain": 3},
        {"insert": 2, "attributes": {"src": "http://quilljs.com/"}},
        {"delete": 4},
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
        {"insert": "Hello", "attributes": {"bold": True}},
        {"retain": 3},
        {"insert": 2, "attributes": {"src": "http://quilljs.com/"}},
        {"delete": 4},
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
    assert iterator.peek_type() == "retain"


def test_type_of_empty():
    assert op.type({}) is None
    assert op.type({"retain": 1.5}) == "retain"


def test_input_and_output_lengths():
    cases = [
        ({"insert": "abc"}, 0, 3),
        ({"retain": 3}, 3, 3),
        ({"delete": 3}, 3, 0),
        ({"cut": {"ref": "r", "length": 3}}, 3, 0),
        ({"paste": {"ref": "r", "start": 0, "length": 3}}, 0, 3),
        ({"insert": {"cell": {}}}, 0, 1),
        ({"retain": {"cell": {"ops": []}}}, 1, 1),
        ({"paste": {"ref": "malformed", "start": 0}}, 0, 1),
    ]
    for operation, consumed, emitted in cases:
        assert op.input_length(operation) == consumed
        assert op.output_length(operation) == emitted


def test_iterator_peek_input_and_output_lengths():
    iterator = op.iterator(
        [
            {"paste": {"ref": "r", "start": 0, "length": 3}},
            {"cut": {"ref": "r", "length": 2}},
        ]
    )
    assert (iterator.peek_input_length(), iterator.peek_output_length()) == (0, 3)
    iterator.next(1)
    assert (iterator.peek_input_length(), iterator.peek_output_length()) == (0, 2)
    iterator.next()
    assert (iterator.peek_input_length(), iterator.peek_output_length()) == (2, 0)
    iterator.next(1)
    assert (iterator.peek_input_length(), iterator.peek_output_length()) == (1, 0)
