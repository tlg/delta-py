"""
Delta offsets count UTF-16 code units, matching the upstream JavaScript
library: astral characters (most emoji) count 2, and an offset may fall
inside a surrogate pair exactly as JavaScript allows.
"""
from delta import Delta, op
from delta.moves import MoveDelta


def test_astral_length():
    assert op.length({'insert': '😀'}) == 2
    assert len(Delta().insert('a😀b')) == 4
    assert Delta().insert('a😀b').change_length() == 4


def test_retain_lands_after_an_emoji():
    doc = Delta().insert('a😀b')
    edit = Delta().retain(3).insert('X')
    assert doc.compose(edit) == Delta().insert('a😀Xb')


def test_delete_covers_an_emoji():
    doc = Delta().insert('a😀b')
    assert doc.compose(Delta().retain(1).delete(2)) == Delta().insert('ab')


def test_split_inside_a_surrogate_pair_reassembles():
    doc = Delta().insert('😀')
    # formatting half an emoji splits the pair, exactly as JS allows
    bolded = doc.compose(Delta().retain(1, bold=True))
    assert bolded.ops == [
        {'insert': '\ud83d', 'attributes': {'bold': True}},
        {'insert': '\ude00'}]
    # removing the format merges the halves back into the real character
    assert bolded.compose(Delta().retain(1, bold=None)) == doc


def test_transform_position_counts_units():
    delta = Delta().insert('😀')
    assert delta.transform_position(0) == 2
    assert Delta().delete(2).transform_position(3) == 1


def test_diff_measures_units():
    a = Delta().insert('a😀b')
    b = Delta().insert('ab')
    d = a.diff(b)
    assert d == Delta().retain(1).delete(2)
    assert a.compose(d) == b


def test_invert_restores_an_emoji():
    doc = Delta().insert('a😀b')
    edit = Delta().retain(1).delete(2)
    inverse = edit.invert(doc)
    assert doc.compose(edit).compose(inverse) == doc


def test_iter_lines_after_an_emoji():
    doc = Delta().insert('😀ab\ncd')
    lines = list(doc.iter_lines())
    assert lines[0][0] == Delta().insert('😀ab')
    assert lines[1][0] == Delta().insert('cd')


def test_move_of_an_emoji_span():
    doc = MoveDelta().insert('a😀b')
    move = MoveDelta().cut('m', 3).paste('m', 0, 3)  # 'a😀' shifted... no-op move
    assert doc.compose(move) == doc
    swap = MoveDelta().cut('m', 1).retain(2).paste('m', 0, 1)
    assert doc.compose(swap) == MoveDelta().insert('😀ab')


def test_concurrent_edits_around_an_emoji_converge():
    doc = MoveDelta().insert('a😀b')
    left = MoveDelta().retain(3).insert('X')
    right = MoveDelta().retain(1).delete(2)
    for priority in (True, False):
        right_prime = left.transform(right, priority)
        left_prime = right.transform(left, not priority)
        assert (doc.compose(left).compose(right_prime)
                == doc.compose(right).compose(left_prime))
