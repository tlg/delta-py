import pytest
from delta import Delta


def test_insert_insert():
    a = Delta().insert('A')
    b = Delta().insert('B')
    expected = Delta().insert('B').insert('A')

    assert (a.compose(b) == expected)


def test_insert_retain():
    a = Delta().insert('A')
    b = Delta().retain(1, **{'bold': True, 'color': 'red', 'font': None})
    expected = Delta().insert('A', **{'bold': True, 'color': 'red'})

    assert (a.compose(b) == expected)


def test_insert_delete():
    a = Delta().insert('A')
    b = Delta().delete(1)
    expected = Delta()

    assert (a.compose(b) == expected)


def test_delete_insert():
    a = Delta().delete(1)
    b = Delta().insert('B')
    expected = Delta().insert('B').delete(1)

    assert (a.compose(b) == expected)


def test_delete_retain():
    a = Delta().delete(1)
    b = Delta().retain(1, **{'bold': True, 'color': 'red'})
    expected = Delta().delete(1).retain(1, **{'bold': True, 'color': 'red'})

    assert (a.compose(b) == expected)


def test_delete_delete():
    a = Delta().delete(1)
    b = Delta().delete(1)
    expected = Delta().delete(2)

    assert (a.compose(b) == expected)


def test_retain_insert():
    a = Delta().retain(1, **{'color': 'blue'})
    b = Delta().insert('B')
    expected = Delta().insert('B').retain(1, **{'color': 'blue'})

    assert (a.compose(b) == expected)


def test_retain_retain():
    a = Delta().retain(1, **{'color': 'blue'})
    b = Delta().retain(1, **{'bold': True, 'color': 'red', 'font': None})
    expected = Delta().retain(
        1, **{'bold': True, 'color': 'red', 'font': None})

    assert (a.compose(b) == expected)


def test_retain_delete():
    a = Delta().retain(1, **{'color': 'blue'})
    b = Delta().delete(1)
    expected = Delta().delete(1)

    assert (a.compose(b) == expected)


def test_insert_in_middle_of_text():
    a = Delta().insert('Hello')
    b = Delta().retain(3).insert('X')
    expected = Delta().insert('HelXlo')

    assert (a.compose(b) == expected)


def test_insert_and_delete_ordering():
    a = Delta().insert('Hello')
    b = Delta().insert('Hello')
    insertFirst = Delta().retain(3).insert('X').delete(1)
    deleteFirst = Delta().retain(3).delete(1).insert('X')
    expected = Delta().insert('HelXo')

    assert (a.compose(insertFirst) == expected)

    assert (b.compose(deleteFirst) == expected)


def test_insert_embed():
    a = Delta().insert({'embed': 1}, **{'src': 'http://quilljs.com/image.png'})
    b = Delta().retain(1, **{'alt': 'logo'})
    expected = Delta().insert(
        {'embed': 1}, **{'src': 'http://quilljs.com/image.png', 'alt': 'logo'})

    assert (a.compose(b) == expected)


def test_retain_embed():
    a = Delta().retain({'figure': True}, **{
        'src': 'http://quilljs.com/image.png'})
    b = Delta().retain(1, **{'alt': 'logo'})
    expected = Delta().retain({'figure': True}, **{
        'src': 'http://quilljs.com/image.png', 'alt': 'logo'})

    assert (a.compose(b) == expected)


def test_delete_entire_text():
    a = Delta().retain(4).insert('Hello')
    b = Delta().delete(9)
    expected = Delta().delete(4)

    assert (a.compose(b) == expected)


def test_retain_more_than_length_of_text():
    a = Delta().insert('Hello')
    b = Delta().retain(10)
    expected = Delta().insert('Hello')

    assert (a.compose(b) == expected)


def test_retain_empty_embed():
    a = Delta().insert({'embed': 1})
    b = Delta().retain(1)
    expected = Delta().insert({'embed': 1})

    assert (a.compose(b) == expected)


def test_remove_all_attributes():
    a = Delta().insert('A', **{'bold': True})
    b = Delta().retain(1, **{'bold': None})
    expected = Delta().insert('A')

    assert (a.compose(b) == expected)


def test_remove_all_embed_attributes():
    a = Delta().insert({'embed': 2}, **{'bold': True})
    b = Delta().retain(1, **{'bold': None})
    expected = Delta().insert({'embed': 2})

    assert (a.compose(b) == expected)


def test_immutability():
    attr1 = {'bold': True}
    attr2 = {'bold': True}
    a1 = Delta().insert('Test', **attr1)
    a2 = Delta().insert('Test', **attr1)
    b1 = Delta().retain(1, **{'color': 'red'}).delete(2)
    b2 = Delta().retain(1, **{'color': 'red'}).delete(2)
    expected = Delta().insert(
        'T', **{'color': 'red', 'bold': True}).insert('t', **attr1)

    assert (a1.compose(b1) == expected)

    assert (a1 == a2)

    assert (b1 == b2)

    assert (attr1 == attr2)


def test_retain_start_optimization():
    a = Delta().insert('A', **{'bold': True}).insert(
        'B').insert('C', **{'bold': True}).delete(1)
    b = Delta().retain(3).insert('D')
    expected = Delta().insert('A', **{'bold': True}).insert(
        'B').insert('C', **{'bold': True}).insert('D').delete(1)

    assert (a.compose(b) == expected)


def test_retain_start_optimization_split():
    a = Delta().insert('A', **{'bold': True}).insert(
        'B').insert('C', **{'bold': True}).retain(5).delete(1)
    b = Delta().retain(4).insert('D')
    expected = Delta().insert('A', **{'bold': True}).insert('B').insert(
        'C', **{'bold': True}).retain(1).insert('D').retain(4).delete(1)

    assert (a.compose(b) == expected)


def test_retain_end_optimization():
    a = Delta().insert('A', **{'bold': True}).insert(
        'B').insert('C', **{'bold': True})
    b = Delta().delete(1)
    expected = Delta().insert('B').insert('C', **{'bold': True})

    assert (a.compose(b) == expected)


def test_retain_end_optimization_join():
    a = Delta().insert('A', **{'bold': True}).insert('B').insert(
        'C', **{'bold': True}).insert('D').insert('E', **{'bold': True}).insert('F')
    b = Delta().retain(1).delete(1)
    expected = Delta().insert('AC', **{'bold': True}).insert(
        'D').insert('E', **{'bold': True}).insert('F')

    assert (a.compose(b) == expected)


@pytest.fixture
def embed_handler():

    class DeltaHandler:
        @staticmethod
        def compose(a, b, keep_null=False):
            return Delta(a).compose(Delta(b)).ops

        @staticmethod
        def invert(a, b):
            return Delta(a).invert(Delta(b)).ops

    Delta.register_embed('delta', DeltaHandler)
    yield
    Delta.unregister_embed('delta')


def test_retain_an_embed_with_a_number(embed_handler):
    a = Delta().insert({"delta": [{"insert": "a"}]})
    b = Delta().retain(1, bold=True)

    expected = Delta().insert({"delta": [{"insert": "a"}]}, bold=True)
    assert a.compose(b) == expected


def test_retain_an_embed_with_an_embed(embed_handler):
    a = Delta().insert({"delta": [{"insert": "a"}]})
    b = Delta().retain({"delta": [{"insert": "b"}]})

    expected = Delta().insert({"delta": [{"insert": "ba"}]})
    assert a.compose(b) == expected


def test_keeps_other_delete_when_this_op_is_a_retain(embed_handler):
    a = Delta().retain({"delta": [{"insert": "a"}]})
    b = Delta().insert('\n').delete(1)

    expected = Delta().insert('\n').delete(1)
    assert a.compose(b) == expected


def test_retain_an_embed_with_another_type_of_embed(embed_handler):
    with pytest.raises(Exception):
        a = Delta().insert({"delta": [{"insert": "a"}]})
        b = Delta().retain({"otherdelta": [{"insert": "b"}]})
        a.compose(b)


def test_retain_a_string_with_an_embed(embed_handler):
    with pytest.raises(Exception):
        a = Delta().insert({"insert": "a"})
        b = Delta().retain({"delta": [{"insert": "b"}]})
        a.compose(b)


def test_retain_embeds_without_a_handler(embed_handler):
    with pytest.raises(Exception):
        a = Delta().insert({"mydelta": [{"insert": "a"}]})
        b = Delta().retain({"mydelta": [{"insert": "b"}]})
        a.compose(b)
