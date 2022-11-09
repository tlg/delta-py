import pytest
from delta import Delta


def test_insert():
    delta = Delta().retain(2).insert('A')
    base = Delta().insert('123456')
    expected = Delta().retain(2).delete(1)

    inverted = delta.invert(base)
    assert inverted == expected
    assert base.compose(delta).compose(inverted) == base


def test_delete():
    delta = Delta().retain(2).delete(3)
    base = Delta().insert('123456')
    expected = Delta().retain(2).insert('345')

    inverted = delta.invert(base)
    assert inverted == expected
    assert base.compose(delta).compose(inverted) == base


def test_retain():
    delta = Delta().retain(2).retain(3,  **{"bold": True})
    base = Delta().insert('123456')
    expected = Delta().retain(2).retain(3, **{"bold": None})

    inverted = delta.invert(base)
    assert inverted == expected
    assert base.compose(delta).compose(inverted) == base


def test_retai_on_delta_different_attributes():
    base = Delta().insert('123').insert(4, **{"bold": True})
    delta = Delta().retain(4, **{"italic": True})
    expected = Delta().retain(4, **{"italic": None})

    inverted = delta.invert(base)
    assert inverted == expected
    assert base.compose(delta).compose(inverted) == base


def test_combined():
    print(Delta().retain(2).insert("123").delete(2))
    delta = Delta().retain(2).delete(2).insert("AB", **{"italic": True}).retain(
        2, **{"italic": None, "bold": True}).retain(2, **{"color": "red"}).delete(1)
    base = Delta().insert("123", **{"bold": True}).insert(
        '456', **{"italic": True}).insert("789", **{"color": 'red', 'bold': True})
    expected = Delta().retain(2).insert('3', **{"bold": True}).insert('4', **{"italic": True}).delete(
        2).retain(2, **{"italic": True, "bold": None}).retain(2).insert('9', **{"color": 'red', "bold": True})

    inverted = delta.invert(base)
    assert inverted == expected
    assert base.compose(delta).compose(inverted) == base
