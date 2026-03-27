# Data-driven diff tests are now in fixtures/delta-diff.json
# Run via test_fixtures.py
#
# The test below checks immutability which cannot be expressed as a JSON fixture.

from delta import Delta


def test_immutability():
    attr1 = {'color': 'red'}
    attr2 = {'color': 'red'}
    a1 = Delta().insert('A', **attr1)
    a2 = Delta().insert('A', **attr1)
    b1 = Delta().insert('A', **{'bold': True}).insert('B')
    b2 = Delta().insert('A', **{'bold': True}).insert('B')
    expected = Delta().retain(1, **{'bold': True, 'color': None}).insert('B')

    assert a1.diff(b1) == expected
    assert a1 == a2
    assert b2 == b2
    assert attr1 == attr2
