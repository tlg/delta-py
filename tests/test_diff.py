import pytest
from delta import Delta


def test_insert():
    a = Delta().insert('A')
    b = Delta().insert('AB')
    expected = Delta().retain(1).insert('B')

    assert (a.diff(b) == expected)


def test_delete():
    a = Delta().insert('AB')
    b = Delta().insert('A')
    expected = Delta().retain(1).delete(1)

    assert (a.diff(b) == expected)


def test_retain():
    a = Delta().insert('A')
    b = Delta().insert('A')
    expected = Delta()

    assert (a.diff(b) == expected)


def test_format():
    a = Delta().insert('A')
    b = Delta().insert('A', **{'bold': True})
    expected = Delta().retain(1, **{'bold': True})

    assert (a.diff(b) == expected)


def test_object_attributes():
    a = Delta().insert(
        'A', **{'font': {'family': 'Helvetica', 'size': '15px'}})
    b = Delta().insert(
        'A', **{'font': {'family': 'Helvetica', 'size': '15px'}})
    expected = Delta()

    assert (a.diff(b) == expected)


def test_embed_integer_match():
    a = Delta().insert({'embed': 1})
    b = Delta().insert({'embed': 1})
    expected = Delta()

    assert (a.diff(b) == expected)


def test_embed_integer_mismatch():
    a = Delta().insert({'embed': 1})
    b = Delta().insert({'embed': 2})
    expected = Delta().delete(1).insert({'embed': 2})

    assert (a.diff(b) == expected)


def test_embed_object_match():
    a = Delta().insert({'image': 'http://quilljs.com'})
    b = Delta().insert({'image': 'http://quilljs.com'})
    expected = Delta()

    assert (a.diff(b) == expected)


def test_embed_object_mismatch():
    a = Delta().insert({'image': 'http://quilljs.com', 'alt': 'Overwrite'})
    b = Delta().insert({'image': 'http://quilljs.com'})
    expected = Delta().insert({'image': 'http://quilljs.com'}).delete(1)

    assert (a.diff(b) == expected)


def test_embed_object_change():
    embed = {'image': 'http://quilljs.com'}
    a = Delta().insert(embed)
    embed['image'] = 'http://github.com'
    b = Delta().insert(embed)
    expected = Delta().insert({'image': 'http://github.com'}).delete(1)

    assert (a.diff(b) == expected)


def test_embed_false_positive():
    a = Delta().insert({'embed': 1})
    b = Delta().insert('\0')
    expected = Delta().insert('\0').delete(1)

    assert (a.diff(b) == expected)


def test_error_on_non_documents():
    a = Delta().insert('A')
    b = Delta().retain(1).insert('B')

    with pytest.raises(Exception):
        a.diff(b)
    with pytest.raises(Exception):
        b.diff(a)


def test_inconvenient_indexes():
    a = Delta().insert('12', **{'bold': True}).insert('34', **{'italic': True})
    b = Delta().insert('123', **{'color': 'red'})
    expected = Delta().retain(2, **{'bold': None, 'color': 'red'}).retain(
        1, **{'italic': None, 'color': 'red'}).delete(1)

    assert (a.diff(b) == expected)


def test_combination():
    a = Delta().insert('Bad', **{'color': 'red'}
                       ).insert('cat', **{'color': 'blue'})
    b = Delta().insert('Good', **{'bold': True}
                       ).insert('dog', **{'italic': True})
    expected = Delta().insert('Good', **{'bold': True}).delete(2).retain(
        1, **{'italic': True, 'color': None}).delete(3).insert('og', **{'italic': True})

    assert (a.diff(b) == expected)


def test_same_document():
    a = Delta().insert('A').insert('B', **{'bold': True})
    expected = Delta()

    assert (a.diff(a) == expected)


def test_immutability():
    attr1 = {'color': 'red'}
    attr2 = {'color': 'red'}
    a1 = Delta().insert('A', **attr1)
    a2 = Delta().insert('A', **attr1)
    b1 = Delta().insert('A', **{'bold': True}).insert('B')
    b2 = Delta().insert('A', **{'bold': True}).insert('B')
    expected = Delta().retain(1, **{'bold': True, 'color': None}).insert('B')

    assert (a1.diff(b1) == expected)

    assert (a1 == a2)

    assert (b2 == b2)

    assert (attr1 == attr2)


def test_non_document():
    a = Delta().insert('Test')
    b = Delta().delete(4)

    with pytest.raises(Exception):
        a.diff(b)
