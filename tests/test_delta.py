# Data-driven builder tests are now in fixtures/delta-builder.json
# Run via test_fixtures.py
#
# The test below checks basic construction which is Python-specific.

from delta import Delta


def test_creation():
    d = Delta()
    assert d.ops == []
    d = Delta([])
    assert d.ops == []
    d2 = Delta(d)
    assert d2.ops == []
