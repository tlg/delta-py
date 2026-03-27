# Data-driven transform tests are now in fixtures/delta-transform.json,
# delta-transform-embed.json, and delta-transform-position.json
# Run via test_fixtures.py
#
# The test below checks immutability which cannot be expressed as a JSON fixture.

from delta import Delta


def test_immutability():
    a1 = Delta().insert('A')
    a2 = Delta().insert('A')
    b1 = Delta().insert('B')
    b2 = Delta().insert('B')
    expected = Delta().retain(1).insert('B')

    assert a1.transform(b1, True) == expected
    assert a1 == a2
    assert b1 == b2
