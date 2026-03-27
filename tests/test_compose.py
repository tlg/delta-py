# Data-driven compose tests are now in fixtures/delta-compose.json and delta-compose-embed.json
# Run via test_fixtures.py
#
# The test below checks immutability which cannot be expressed as a JSON fixture.

from delta import Delta


def test_immutability():
    attr1 = {'bold': True}
    attr2 = {'bold': True}
    a1 = Delta().insert('Test', **attr1)
    a2 = Delta().insert('Test', **attr1)
    b1 = Delta().retain(1, **{'color': 'red'}).delete(2)
    b2 = Delta().retain(1, **{'color': 'red'}).delete(2)
    expected = Delta().insert(
        'T', **{'color': 'red', 'bold': True}).insert('t', **attr1)

    assert a1.compose(b1) == expected
    assert a1 == a2
    assert b1 == b2
    assert attr1 == attr2
