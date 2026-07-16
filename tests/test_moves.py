import random

import pytest

from delta import Delta, MoveDelta, op
from delta.moves import check, has_moves


def apply(doc, delta):
    return MoveDelta(doc.ops).compose(delta)


def doc(text):
    return MoveDelta().insert(text)


# --- builders and normalization ------------------------------------------

def test_builders():
    delta = MoveDelta().cut('r', 3).retain(2).paste('r', 0, 3, bold=True)
    assert delta.ops == [
        {'cut': {'ref': 'r', 'length': 3}},
        {'retain': 2},
        {'paste': {'ref': 'r', 'start': 0, 'length': 3},
         'attributes': {'bold': True}},
    ]


def test_push_merges_adjacent_paste_windows():
    delta = MoveDelta().paste('r', 0, 2).paste('r', 2, 8)
    assert delta.ops == [{'paste': {'ref': 'r', 'start': 0, 'length': 10}}]


def test_push_keeps_non_adjacent_windows():
    delta = MoveDelta().paste('r', 0, 2).paste('r', 5, 3)
    assert len(delta.ops) == 2


def test_change_length():
    delta = MoveDelta().cut('r', 10).retain(3).paste('r', 0, 4)
    assert delta.change_length() == -6
    assert has_moves(delta)


def test_check_rejects_invalid_moves():
    with pytest.raises(ValueError):
        check(MoveDelta().cut('r', 3).retain(1).cut('r', 2).paste('r', 0, 3))
    with pytest.raises(ValueError):
        check(MoveDelta().paste('r', 0, 3))
    with pytest.raises(ValueError):
        check(MoveDelta().cut('r', 3).paste('r', 0, 2).paste('r', 1, 2))
    with pytest.raises(ValueError):
        check(MoveDelta().cut('r', 3).paste('r', 2, 2))
    # a cut without a paste is a verbose delete, but legal
    check(MoveDelta().cut('r', 3))


# --- apply (compose with a document) --------------------------------------

def test_apply_move_right():
    move = MoveDelta().cut('r', 10).retain(3).paste('r', 0, 10)
    assert apply(doc('ABCDEFGHIJxyz'), move).document() == 'xyzABCDEFGHIJ'


def test_apply_move_left():
    move = MoveDelta().retain(3).paste('r', 0, 3).retain(4).cut('r', 3)
    assert apply(doc('xxxyyyyABC'), move).document() == 'xxxABCyyyy'


def test_apply_partial_windows_drop_the_gap():
    move = MoveDelta().cut('r', 4).retain(2).paste('r', 0, 1).paste('r', 3, 1)
    assert apply(doc('ABCDxx'), move).document() == 'xxAD'


def test_apply_paste_attributes_patch_content():
    base = MoveDelta().insert('AB', bold=True).insert('xx')
    move = MoveDelta().cut('r', 2).retain(2).paste('r', 0, 2, bold=None, i=True)
    assert apply(base, move).ops == [
        {'insert': 'xx'},
        {'insert': 'AB', 'attributes': {'i': True}},
    ]


def test_apply_moves_embeds():
    base = MoveDelta().insert({'image': 'a.png'}).insert('xy')
    move = MoveDelta().cut('r', 1).retain(2).paste('r', 0, 1)
    assert apply(base, move).ops == [
        {'insert': 'xy'}, {'insert': {'image': 'a.png'}}]


# --- compose: later ordinary edits over a pasted span ---------------------

def test_compose_insert_splits_paste_window():
    move = MoveDelta().cut('r', 10).retain(3).paste('r', 0, 10)
    edit = MoveDelta().retain(5).insert('test')
    composed = move.compose(edit)
    assert composed.ops == [
        {'cut': {'ref': 'r', 'length': 10}},
        {'retain': 3},
        {'paste': {'ref': 'r', 'start': 0, 'length': 2}},
        {'insert': 'test'},
        {'paste': {'ref': 'r', 'start': 2, 'length': 8}},
    ]
    base = doc('ABCDEFGHIJxyz')
    assert apply(base, composed) == apply(base, move).compose(edit)


def test_compose_delete_splits_paste_window():
    move = MoveDelta().cut('r', 10).retain(3).paste('r', 0, 10)
    edit = MoveDelta().retain(4).delete(3)
    composed = move.compose(edit)
    assert composed.ops == [
        {'cut': {'ref': 'r', 'length': 10}},
        {'retain': 3},
        {'paste': {'ref': 'r', 'start': 0, 'length': 1}},
        {'paste': {'ref': 'r', 'start': 4, 'length': 6}},
    ]
    base = doc('ABCDEFGHIJxyz')
    assert apply(base, composed) == apply(base, move).compose(edit)


def test_compose_format_lands_on_paste_attributes():
    move = MoveDelta().cut('r', 10).retain(3).paste('r', 0, 10)
    edit = MoveDelta().retain(3).retain(2, bold=True)
    composed = move.compose(edit)
    assert composed.ops == [
        {'cut': {'ref': 'r', 'length': 10}},
        {'retain': 3},
        {'paste': {'ref': 'r', 'start': 0, 'length': 2},
         'attributes': {'bold': True}},
        {'paste': {'ref': 'r', 'start': 2, 'length': 8}},
    ]
    base = doc('ABCDEFGHIJxyz')
    assert apply(base, composed) == apply(base, move).compose(edit)


def test_compose_delete_of_every_window_degrades_cut_to_delete():
    move = MoveDelta().cut('r', 3).retain(3).paste('r', 0, 3)
    edit = MoveDelta().retain(3).delete(3)
    assert move.compose(edit).ops == [{'delete': 3}]


# --- compose: a later cut over an edited region ----------------------------

def test_compose_cut_carries_earlier_insert_to_the_paste_site():
    first = MoveDelta().retain(2).insert('X')
    second = MoveDelta().retain(1).cut('m', 4).retain(2).paste('m', 0, 4)
    composed = first.compose(second)
    assert composed.ops == [
        {'retain': 1},
        {'cut': {'ref': 'm', 'length': 3}},
        {'retain': 2},
        {'paste': {'ref': 'm', 'start': 0, 'length': 1}},
        {'insert': 'X'},
        {'paste': {'ref': 'm', 'start': 1, 'length': 2}},
    ]
    base = doc('ABCDEF')
    assert apply(base, composed) == apply(base, first).compose(second)
    assert apply(base, composed).document() == 'AEFBXCD'


def test_compose_cut_carries_earlier_format_to_the_paste_site():
    first = MoveDelta().retain(1).retain(2, bold=True)
    second = MoveDelta().cut('m', 4).retain(2).paste('m', 0, 4)
    composed = first.compose(second)
    base = doc('ABCDEF')
    assert apply(base, composed) == apply(base, first).compose(second)


def test_compose_cut_absorbs_earlier_delete():
    first = MoveDelta().retain(1).delete(2)
    second = MoveDelta().cut('m', 2).retain(2).paste('m', 0, 2)
    composed = first.compose(second)
    # the cut spans the deleted base characters, the windows skip them
    assert composed.ops == [
        {'cut': {'ref': 'm', 'length': 4}},
        {'retain': 2},
        {'paste': {'ref': 'm', 'start': 0, 'length': 1}},
        {'paste': {'ref': 'm', 'start': 3, 'length': 1}},
    ]
    base = doc('ABCDEF')
    assert apply(base, composed) == apply(base, first).compose(second)


def test_compose_cut_over_a_paste_chains_the_reference():
    first = MoveDelta().cut('r', 3).retain(3).paste('r', 0, 3)
    second = (MoveDelta().paste('s', 0, 3).retain(2)
              .cut('s', 3).retain(1))
    composed = first.compose(second)
    base = doc('ABCDEF')
    assert apply(base, first).document() == 'DEFABC'
    assert apply(base, composed).document() == 'FABDEC'
    assert apply(base, composed) == apply(base, first).compose(second)


def test_compose_move_of_a_pure_insert_needs_no_cut():
    first = MoveDelta().insert('XY')
    second = MoveDelta().retain(1).cut('m', 1).paste('m', 0, 1)
    composed = first.compose(second)
    assert not has_moves(composed)
    assert composed.ops == [{'insert': 'XY'}]


def test_compose_renames_colliding_refs():
    first = MoveDelta().cut('r', 1).retain(1).paste('r', 0, 1)
    second = MoveDelta().cut('r', 1).retain(1).paste('r', 0, 1)
    composed = first.compose(second)
    base = doc('AB')
    assert apply(base, composed) == apply(base, first).compose(second)


def test_compose_cut_over_cut_splits_the_later_cut():
    first = MoveDelta().retain(1).cut('a', 2).retain(1).paste('a', 0, 2)
    second = MoveDelta().cut('b', 4).retain(2).paste('b', 0, 4)
    composed = first.compose(second)
    assert composed.ops == [
        {'cut': {'ref': 'b', 'length': 1}},
        {'cut': {'ref': 'a', 'length': 2}},
        {'cut': {'ref': 'b:1', 'length': 1}},
        {'retain': 2},
        {'paste': {'ref': 'b', 'start': 0, 'length': 1}},
        {'paste': {'ref': 'b:1', 'start': 0, 'length': 1}},
        {'paste': {'ref': 'a', 'start': 0, 'length': 2}},
    ]
    base = doc('ABCDEF')
    assert apply(base, first).document() == 'ADBCEF'
    assert apply(base, composed).document() == 'EFADBC'
    assert apply(base, composed) == apply(base, first).compose(second)


@pytest.fixture
def fig_handler():
    class FigHandler:
        @staticmethod
        def compose(a, b, keep_null):
            merged = {**(a or {}), **(b or {})}
            if not keep_null:
                merged = {k: v for k, v in merged.items() if v is not None}
            return merged

        @staticmethod
        def transform(a, b, priority):
            if not priority:
                return b
            return {k: v for k, v in (b or {}).items() if k not in (a or {})}

        @staticmethod
        def invert(change, base):
            return {k: base.get(k) for k in (change or {})
                    if base.get(k) != change[k]}

    Delta.register_embed('fig', FigHandler)
    yield
    Delta.unregister_embed('fig')


def test_compose_moving_an_embed_change(fig_handler):
    # first patches the embed, second moves it: the patch rides the paste
    first = MoveDelta().retain(1).retain({'fig': {'w': 2}})
    second = MoveDelta().retain(1).cut('m', 1).retain(1).paste('m', 0, 1)
    composed = first.compose(second)
    assert composed.ops == [
        {'retain': 1},
        {'cut': {'ref': 'm', 'length': 1}},
        {'retain': 1},
        {'paste': {'ref': 'm', 'start': 0, 'length': 1,
                   'change': {'fig': {'w': 2}}}},
    ]
    base = MoveDelta().insert('a').insert({'fig': {'v': 1}}).insert('b')
    assert apply(base, composed) == apply(base, first).compose(second)
    assert apply(base, composed).ops == [
        {'insert': 'ab'}, {'insert': {'fig': {'v': 1, 'w': 2}}}]


def test_compose_embed_change_over_a_pasted_embed(fig_handler):
    # first moves the embed, second patches it at the paste site
    first = MoveDelta().cut('m', 1).retain(2).paste('m', 0, 1)
    second = MoveDelta().retain(2).retain({'fig': {'w': 2}})
    composed = first.compose(second)
    assert composed.ops == [
        {'cut': {'ref': 'm', 'length': 1}},
        {'retain': 2},
        {'paste': {'ref': 'm', 'start': 0, 'length': 1,
                   'change': {'fig': {'w': 2}}}},
    ]
    base = MoveDelta().insert({'fig': {'v': 1}}).insert('ab')
    assert apply(base, composed) == apply(base, first).compose(second)


def test_invert_reverts_a_paste_change(fig_handler):
    base = MoveDelta().insert({'fig': {'v': 1}}).insert('ab')
    move = MoveDelta([
        {'cut': {'ref': 'm', 'length': 1}},
        {'retain': 2},
        {'paste': {'ref': 'm', 'start': 0, 'length': 1,
                   'change': {'fig': {'v': 9, 'w': 2}}}},
    ])
    inverted = move.invert(base)
    assert apply(base, move).compose(inverted) == base


def test_transform_concurrent_embed_change_and_move_converge(fig_handler):
    base = MoveDelta().insert({'fig': {'v': 1}}).insert('ab')
    move = MoveDelta().cut('m', 1).retain(2).paste('m', 0, 1)
    edit = MoveDelta().retain({'fig': {'v': 5}})
    for move_wins in (True, False):
        edit_prime = move.transform(edit, move_wins)
        move_prime = edit.transform(move, not move_wins)
        assert (apply(base, move).compose(edit_prime)
                == apply(base, edit).compose(move_prime)), move_wins


# --- lower and invert ------------------------------------------------------

def test_lower_matches_move_application():
    base = doc('ABCDEFGHIJxyz')
    move = MoveDelta().cut('r', 10).retain(3).paste('r', 0, 10, bold=True)
    lowered = move.lower(base)
    assert not has_moves(MoveDelta(lowered.ops))
    assert apply(base, move) == apply(base, MoveDelta(lowered.ops))


def test_invert_is_semantic():
    base = doc('ABCDEF')
    move = MoveDelta().cut('r', 3).retain(3).paste('r', 0, 3)
    inverted = move.invert(base)
    # the inverse is the opposite move, not a materialized delete/insert
    assert inverted.ops == [
        {'paste': {'ref': 'r', 'start': 0, 'length': 3}},
        {'retain': 3},
        {'cut': {'ref': 'r', 'length': 3}},
    ]
    assert apply(base, move).compose(inverted) == base


def test_invert_restores_dropped_gaps_and_splits_refs():
    base = doc('ABCDEFxx')
    move = MoveDelta().cut('r', 4).retain(2).paste('r', 0, 1).retain(2).paste('r', 3, 1)
    inverted = move.invert(base)
    assert has_moves(inverted)
    # the never-pasted middle 'BC' comes back as a literal insert
    assert {'insert': 'BC'} in inverted.ops
    assert apply(base, move).compose(inverted) == base


def test_invert_reverts_paste_attribute_patches():
    base = MoveDelta().insert('AB', bold=True).insert('xx')
    move = MoveDelta().cut('r', 2).retain(2).paste('r', 0, 2, bold=None, i=True)
    inverted = move.invert(base)
    assert inverted.ops == [
        {'paste': {'ref': 'r', 'start': 0, 'length': 2},
         'attributes': {'bold': True, 'i': None}},
        {'retain': 2},
        {'cut': {'ref': 'r', 'length': 2}},
    ]
    assert apply(base, move).compose(inverted) == base


# --- transform -------------------------------------------------------------

def test_transform_routes_format_to_the_paste_site():
    move = MoveDelta().cut('r', 3).retain(3).paste('r', 0, 3)
    edit = MoveDelta().retain(1).retain(1, bold=True)
    routed = move.transform(edit, True)
    assert routed.ops == [{'retain': 4}, {'retain': 1, 'attributes': {'bold': True}}]


def test_transform_routes_delete_to_the_paste_site():
    move = MoveDelta().cut('r', 3).retain(3).paste('r', 0, 3)
    edit = MoveDelta().retain(1).delete(1)
    routed = move.transform(edit, True)
    assert routed.ops == [{'retain': 4}, {'delete': 1}]


def test_transform_insert_in_source_stays_at_source():
    move = MoveDelta().cut('r', 3).retain(3).paste('r', 0, 3)
    edit = MoveDelta().retain(1).insert('x')
    routed = move.transform(edit, True)
    assert routed.ops == [{'insert': 'x'}]


def test_transform_shrinks_windows_when_source_is_deleted():
    move = MoveDelta().cut('r', 3).retain(3).paste('r', 0, 3)
    edit = MoveDelta().retain(1).delete(1)
    transformed = edit.transform(move, False)
    assert transformed.ops == [
        {'cut': {'ref': 'r', 'length': 2}},
        {'retain': 3},
        {'paste': {'ref': 'r', 'start': 0, 'length': 2}},
    ]


def test_transform_splits_cut_around_a_concurrent_insert():
    move = MoveDelta().cut('r', 3).retain(3).paste('r', 0, 3)
    edit = MoveDelta().retain(1).insert('x')
    transformed = edit.transform(move, False)
    assert transformed.ops == [
        {'cut': {'ref': 'r', 'length': 1}},
        {'retain': 1},
        {'cut': {'ref': 'r:1', 'length': 2}},
        {'retain': 3},
        {'paste': {'ref': 'r', 'start': 0, 'length': 1}},
        {'paste': {'ref': 'r:1', 'start': 0, 'length': 2}},
    ]


def test_transform_convergence_simple():
    base = doc('ABCDEF')
    move = MoveDelta().cut('r', 3).retain(3).paste('r', 0, 3)
    for edit in (
        MoveDelta().retain(1).retain(1, bold=True),
        MoveDelta().retain(1).delete(1),
        MoveDelta().retain(1).insert('x'),
        MoveDelta().delete(4),
        MoveDelta().retain(2).delete(3),
    ):
        for move_wins in (True, False):
            edit_prime = move.transform(edit, move_wins)
            move_prime = edit.transform(move, not move_wins)
            assert (apply(base, move).compose(edit_prime)
                    == apply(base, edit).compose(move_prime)), (edit, move_wins)


def test_transform_concurrent_same_source_moves_rebase():
    base = doc('ABCDEF')
    # both move BC: winner to the end, loser to the front
    winner = MoveDelta().retain(1).cut('a', 2).retain(3).paste('a', 0, 2)
    loser = MoveDelta().retain(1).paste('b', 0, 2).cut('b', 2)

    loser_prime = winner.transform(loser, True)
    winner_prime = loser.transform(winner, False)
    # the loser's claim drops entirely
    assert loser_prime.ops == []
    # the winner is rebased: it re-cuts BC out of the loser's paste site
    assert winner_prime.ops == [
        {'retain': 1},
        {'cut': {'ref': 'a', 'length': 2}},
        {'retain': 3},
        {'paste': {'ref': 'a', 'start': 0, 'length': 2}},
    ]
    assert (apply(base, winner).compose(loser_prime)
            == apply(base, loser).compose(winner_prime)
            == doc('ADEFBC'))


def test_transform_concurrent_disjoint_moves_both_apply():
    base = doc('ABCDEF')
    a = MoveDelta().retain(1).cut('a', 1).retain(4).paste('a', 0, 1)
    b = MoveDelta().paste('b', 0, 1).retain(4).cut('b', 1)
    for a_wins in (True, False):
        b_prime = a.transform(b, a_wins)
        a_prime = b.transform(a, not a_wins)
        one = apply(base, a).compose(b_prime)
        two = apply(base, b).compose(a_prime)
        assert one == two == doc('EACDFB')


def test_transform_concurrent_overlapping_moves_converge():
    base = doc('ABCDEF')
    a = MoveDelta().cut('a', 4).retain(2).paste('a', 0, 4)      # ABCD -> end
    b = MoveDelta().retain(2).cut('b', 4).paste('b', 0, 4)      # CDEF -> end
    for a_wins in (True, False):
        b_prime = a.transform(b, a_wins)
        a_prime = b.transform(a, not a_wins)
        assert (apply(base, a).compose(b_prime)
                == apply(base, b).compose(a_prime)), a_wins


def test_transform_position_follows_moved_content():
    move = MoveDelta().cut('r', 3).retain(3).paste('r', 0, 3)   # ABCDEF -> DEFABC
    assert move.transform_position(1) == 4   # inside the moved span: follows
    assert move.transform_position(0) == 0   # at the region start: stays
    assert move.transform_position(4) == 1   # after the cut: shifts left


def test_transform_position_dropped_content_collapses():
    move = MoveDelta().cut('r', 3).retain(3).paste('r', 0, 1).paste('r', 2, 1)
    assert move.transform_position(1) == 0   # in the dropped gap
    assert move.transform_position(2) == 4   # in the second window


# --- fuzzing the algebraic laws ---------------------------------------------

# the emoji is astral: it exercises UTF-16 offsets through every law
LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', '😀']
FUZZ_ATTRS = [None, None, {'bold': True}, {'bold': None}, {'italic': True}]


def random_doc(rng):
    result = MoveDelta()
    for index in range(rng.randint(2, 5)):
        if rng.random() < 0.2:
            result.insert({'image': f'{index}.png'})
            continue
        text = ''.join(rng.choice(LETTERS) for _ in range(rng.randint(1, 4)))
        attrs = {'bold': True} if rng.random() < 0.3 else {}
        result.insert(text, **attrs)
    return result


def random_ordinary(rng, length):
    delta = MoveDelta()
    position = 0
    while position < length:
        choice = rng.random()
        span = rng.randint(1, min(3, length - position))
        if choice < 0.3:
            delta.retain(span)
        elif choice < 0.55:
            delta.retain(span, **rng.choice(FUZZ_ATTRS[2:]))
        elif choice < 0.8:
            delta.delete(span)
        else:
            delta.insert('X' * rng.randint(1, 2))
            continue
        position += span
    if rng.random() < 0.4:
        delta.insert('Z')
    return delta


def random_move(rng, length, ref='r'):
    """A move with a possibly dropped middle, windows possibly scattered
    to different destinations, and conflicting attribute patches."""
    size = rng.randint(1, min(4, length - 1))
    source = rng.randint(0, length - size)
    windows = [(0, size)]
    if size >= 2 and rng.random() < 0.5:
        left = rng.randint(1, size - 1)
        right = rng.randint(left, size - 1)
        windows = [w for w in [(0, left), (right, size - right)] if w[1] > 0]
    events = [(source, 0, 'cut', None, None)]
    for order, window in enumerate(windows, 1):
        target = rng.choice([q for q in range(length + 1)
                             if q <= source or q >= source + size])
        events.append((target, order, 'paste', window, rng.choice(FUZZ_ATTRS)))
    events.sort(key=lambda event: (event[0], rng.random()))
    delta = MoveDelta()
    position = 0
    for target, _, kind, window, attrs in events:
        delta.retain(max(0, target - position))
        position = max(position, target)
        if kind == 'cut':
            delta.cut(ref, size)
            position += size
        else:
            delta.paste(ref, window[0], window[1], **(attrs or {}))
    return delta


def test_fuzz_compose_move_then_ordinary():
    rng = random.Random(1)
    for _ in range(400):
        base = random_doc(rng)
        if len(base) < 2:
            continue
        move = random_move(rng, len(base))
        after = apply(base, move)
        edit = random_ordinary(rng, len(after))
        assert after.compose(edit) == apply(base, move.compose(edit))


def test_fuzz_compose_ordinary_then_move():
    rng = random.Random(2)
    for _ in range(400):
        base = random_doc(rng)
        edit = random_ordinary(rng, len(base))
        after = apply(base, edit)
        if len(after) < 2:
            continue
        move = random_move(rng, len(after))
        assert after.compose(move) == apply(base, edit.compose(move))


def test_fuzz_compose_move_then_move():
    rng = random.Random(3)
    for _ in range(400):
        base = random_doc(rng)
        if len(base) < 2:
            continue
        first = random_move(rng, len(base), ref='a')
        after = apply(base, first)
        if len(after) < 2:
            continue
        second = random_move(rng, len(after), ref='b')
        composed = first.compose(second)
        assert after.compose(second) == apply(base, composed)


def test_fuzz_compose_lower_equivalence():
    rng = random.Random(4)
    for _ in range(200):
        base = random_doc(rng)
        if len(base) < 2:
            continue
        move = random_move(rng, len(base))
        lowered = MoveDelta(move.lower(base).ops)
        assert apply(base, move) == apply(base, lowered)
        assert apply(base, move).compose(move.invert(base)) == base


def test_fuzz_transform_convergence():
    rng = random.Random(5)
    for _ in range(400):
        base = random_doc(rng)
        if len(base) < 2:
            continue
        move = random_move(rng, len(base))
        edit = random_ordinary(rng, len(base))
        for move_wins in (True, False):
            edit_prime = move.transform(edit, move_wins)
            move_prime = edit.transform(move, not move_wins)
            assert (apply(base, move).compose(edit_prime)
                    == apply(base, edit).compose(move_prime)), \
                (base.ops, move.ops, edit.ops, move_wins)


def test_fuzz_transform_move_vs_move_convergence():
    rng = random.Random(6)
    for _ in range(400):
        base = random_doc(rng)
        if len(base) < 2:
            continue
        first = random_move(rng, len(base), ref='a')
        second = random_move(rng, len(base), ref='b')
        for first_wins in (True, False):
            second_prime = first.transform(second, first_wins)
            first_prime = second.transform(first, not first_wins)
            assert (apply(base, first).compose(second_prime)
                    == apply(base, second).compose(first_prime)), \
                (base.ops, first.ops, second.ops, first_wins)


def doc_units(document):
    units = []
    for operation in document.ops:
        insert = operation['insert']
        units.extend('t' * op.str_length(insert)
                     if isinstance(insert, str) else 'e')
    return units


def random_fig_doc(rng):
    result = MoveDelta()
    for index in range(rng.randint(2, 5)):
        if rng.random() < 0.35:
            result.insert({'fig': {'v': index}})
        else:
            result.insert(
                ''.join(rng.choice(LETTERS) for _ in range(rng.randint(1, 3))))
    return result


def random_fig_edit(rng, units):
    delta = MoveDelta()
    position = 0
    while position < len(units):
        if units[position] == 'e' and rng.random() < 0.5:
            delta.retain({'fig': {rng.choice('vw'): rng.randint(0, 3)}})
            position += 1
            continue
        span = rng.randint(1, min(3, len(units) - position))
        choice = rng.random()
        if choice < 0.4:
            delta.retain(span)
        elif choice < 0.6:
            delta.retain(span, bold=True)
        elif choice < 0.8:
            delta.delete(span)
        else:
            delta.insert('X')
            continue
        position += span
    return delta


def random_fig_move(rng, units, ref='r'):
    move = random_move(rng, len(units), ref)
    source = 0
    for operation in move.ops:
        if 'cut' in operation:
            break
        if 'retain' in operation:
            source += operation['retain']
        elif 'delete' in operation:
            source += operation['delete']
    for operation in move.ops:
        spec = operation.get('paste')
        if (spec and spec['length'] == 1 and rng.random() < 0.6
                and units[source + spec['start']] == 'e'):
            spec['change'] = {'fig': {rng.choice('vw'): rng.randint(0, 3)}}
    return move


def test_fuzz_embed_changes_with_moves(fig_handler):
    rng = random.Random(8)
    for _ in range(300):
        base = random_fig_doc(rng)
        units = doc_units(base)
        if len(units) < 2:
            continue
        move = random_fig_move(rng, units, ref='a')
        second = random_fig_move(rng, units, ref='b')
        edit = random_fig_edit(rng, units)
        # compose laws in both orders, and move over move
        after = apply(base, move)
        later = random_fig_edit(rng, doc_units(after))
        assert after.compose(later) == apply(base, move.compose(later))
        after_edit = apply(base, edit)
        if len(after_edit) >= 2:
            chased = random_move(rng, len(after_edit), ref='c')
            assert (after_edit.compose(chased)
                    == apply(base, MoveDelta(edit.ops).compose(chased)))
        # invert roundtrip with paste changes
        assert after.compose(move.invert(base)) == base
        # transform convergence against edits and against concurrent moves
        for pair in ((move, edit), (move, second)):
            for wins in (True, False):
                right_prime = pair[0].transform(pair[1], wins)
                left_prime = pair[1].transform(pair[0], not wins)
                assert (apply(base, pair[0]).compose(right_prime)
                        == apply(base, pair[1]).compose(left_prime)), \
                    (base.ops, pair[0].ops, pair[1].ops, wins)


@pytest.fixture
def cell_handler():
    class CellHandler:
        @staticmethod
        def compose(a, b, keep_null):
            return {'ops': MoveDelta(list(a['ops'])).compose(
                MoveDelta(list(b['ops']))).ops}

        @staticmethod
        def transform(a, b, priority):
            return {'ops': MoveDelta(list(a['ops'])).transform(
                MoveDelta(list(b['ops'])), priority).ops}

        @staticmethod
        def invert(change, base):
            return {'ops': MoveDelta(list(change['ops'])).invert(
                MoveDelta(list(base['ops']))).ops}

        @staticmethod
        def diff(a, b):
            return {'ops': MoveDelta(list(a['ops'])).diff(
                MoveDelta(list(b['ops']))).ops}

    Delta.register_embed('cell', CellHandler)
    yield
    Delta.unregister_embed('cell')


def cell(text):
    return {'cell': {'ops': [{'insert': text}]}}


def cell_patch(*ops):
    return {'retain': {'cell': {'ops': list(ops)}}}


def test_apply_cell_to_root_move(cell_handler):
    base = MoveDelta().insert('AB').insert(cell('Hello')).insert('CD')
    move = MoveDelta([
        {'retain': 2},
        cell_patch({'cut': {'ref': 'm', 'length': 5}}),
        {'retain': 1},
        {'paste': {'ref': 'm', 'start': 0, 'length': 5}},
    ])
    assert apply(base, move).ops == [
        {'insert': 'AB'}, {'insert': {'cell': {'ops': []}}},
        {'insert': 'CHelloD'}]
    # same move with the paste ahead of the cut exercises the rerun
    leftward = MoveDelta([
        {'paste': {'ref': 'm', 'start': 0, 'length': 5}},
        {'retain': 2},
        cell_patch({'cut': {'ref': 'm', 'length': 5}}),
    ])
    assert apply(base, leftward).ops == [
        {'insert': 'HelloAB'}, {'insert': {'cell': {'ops': []}}},
        {'insert': 'CD'}]


def test_apply_root_to_cell_move(cell_handler):
    base = MoveDelta().insert('AB').insert(cell('Hello')).insert('CD')
    move = MoveDelta([
        {'cut': {'ref': 'm', 'length': 2}},
        cell_patch({'retain': 5}, {'paste': {'ref': 'm', 'start': 0, 'length': 2}}),
    ])
    assert apply(base, move).ops == [
        {'insert': {'cell': {'ops': [{'insert': 'HelloAB'}]}}},
        {'insert': 'CD'}]


def test_apply_cell_to_cell_move(cell_handler):
    base = MoveDelta().insert(cell('one')).insert(cell('two'))
    move = MoveDelta([
        cell_patch({'cut': {'ref': 'x', 'length': 3}}),
        cell_patch({'retain': 3}, {'paste': {'ref': 'x', 'start': 0, 'length': 3}}),
    ])
    assert apply(base, move).ops == [
        {'insert': {'cell': {'ops': []}}},
        {'insert': {'cell': {'ops': [{'insert': 'twoone'}]}}}]


def test_compose_edit_splits_cross_level_paste(cell_handler):
    base = MoveDelta().insert('AB').insert(cell('Hello')).insert('CD')
    move = MoveDelta([
        {'retain': 2},
        cell_patch({'cut': {'ref': 'm', 'length': 5}}),
        {'retain': 1},
        {'paste': {'ref': 'm', 'start': 0, 'length': 5}},
    ])
    later = MoveDelta().retain(6).insert('X')
    composed = move.compose(later)
    assert composed.ops == [
        {'retain': 2},
        cell_patch({'cut': {'ref': 'm', 'length': 5}}),
        {'retain': 1},
        {'paste': {'ref': 'm', 'start': 0, 'length': 2}},
        {'insert': 'X'},
        {'paste': {'ref': 'm', 'start': 2, 'length': 3}},
    ]
    assert apply(base, composed) == apply(base, move).compose(later)


def test_invert_cross_level_moves(cell_handler):
    base = MoveDelta().insert('AB').insert(cell('Hello')).insert('CD')
    for move in (
        MoveDelta([{'retain': 2},
                   cell_patch({'cut': {'ref': 'm', 'length': 5}}),
                   {'retain': 1},
                   {'paste': {'ref': 'm', 'start': 0, 'length': 5}}]),
        MoveDelta([{'cut': {'ref': 'm', 'length': 2}},
                   cell_patch({'retain': 5},
                              {'paste': {'ref': 'm', 'start': 0, 'length': 2}})]),
        MoveDelta([{'retain': 2},
                   cell_patch({'retain': 1}, {'cut': {'ref': 'm', 'length': 4}}),
                   {'paste': {'ref': 'm', 'start': 0, 'length': 1}},
                   {'retain': 2},
                   {'paste': {'ref': 'm', 'start': 2, 'length': 2}}]),
    ):
        after = apply(base, move)
        assert after.compose(move.invert(base)) == base, move.ops


def test_compose_deleting_a_sourcing_embed_uses_the_trash(cell_handler):
    base = MoveDelta().insert('AB').insert(cell('Hello')).insert('CD')
    move = MoveDelta([
        {'retain': 2},
        cell_patch({'retain': 1}, {'cut': {'ref': 'm', 'length': 3}}),
        {'retain': 1},
        {'paste': {'ref': 'm', 'start': 0, 'length': 3}},
    ])
    later = MoveDelta().retain(2).delete(1)  # delete the emptied cell
    composed = move.compose(later)
    # the deletion became a trash cut; the paste reads through it by path
    assert composed.ops == [
        {'retain': 2},
        {'cut': {'ref': 'trash', 'length': 1}},
        {'retain': 1},
        {'paste': {'ref': 'trash', 'unit': 0, 'path': ['ops'],
                   'start': 1, 'length': 3}},
    ]
    after = apply(base, move).compose(later)
    assert after == apply(base, composed) == doc('ABCellD')
    # and the trashed composition still inverts
    assert apply(base, composed).compose(composed.invert(base)) == base


def test_transform_cell_edit_routes_to_root_paste_site(cell_handler):
    base = MoveDelta().insert('AB').insert(cell('Hello')).insert('CD')
    move = MoveDelta([
        {'retain': 2},
        cell_patch({'cut': {'ref': 'm', 'length': 5}}),
        {'retain': 1},
        {'paste': {'ref': 'm', 'start': 0, 'length': 5}},
    ])
    edit = MoveDelta([{'retain': 2},
                      cell_patch({'retain': 1},
                                 {'retain': 2, 'attributes': {'bold': True}})])
    routed = move.transform(edit, True)
    # the format lands at the root paste site (position 5 after the move),
    # not inside the emptied cell; the cell patch itself becomes a no-op
    assert routed.ops == [
        {'retain': 2}, {'retain': {'cell': {'ops': []}}},
        {'retain': 2}, {'retain': 2, 'attributes': {'bold': True}}]


def test_transform_cross_level_convergence(cell_handler):
    base = MoveDelta().insert('AB').insert(cell('Hello')).insert('CD')
    moves = (
        MoveDelta([{'retain': 2},
                   cell_patch({'cut': {'ref': 'm', 'length': 5}}),
                   {'retain': 1},
                   {'paste': {'ref': 'm', 'start': 0, 'length': 5}}]),
        MoveDelta([{'cut': {'ref': 'm', 'length': 2}},
                   cell_patch({'retain': 5},
                              {'paste': {'ref': 'm', 'start': 0, 'length': 2}})]),
    )
    edits = (
        MoveDelta().insert('X'),
        MoveDelta().retain(1, bold=True),
        MoveDelta().retain(2).delete(1),         # deletes the embed
        MoveDelta().retain(4).delete(1),
        MoveDelta([{'retain': 2},
                   cell_patch({'retain': 1}, {'delete': 2})]),
        MoveDelta([{'retain': 2},
                   cell_patch({'retain': 2}, {'insert': 'zz'})]),
        MoveDelta().retain(3).cut('z', 1).paste('z', 0, 1),
    )
    for move in moves:
        for edit in edits:
            for move_wins in (True, False):
                edit_prime = move.transform(edit, move_wins)
                move_prime = edit.transform(move, not move_wins)
                assert (apply(base, move).compose(edit_prime)
                        == apply(base, edit).compose(move_prime)), \
                    (move.ops, edit.ops, move_wins)


def cell_length(document):
    for operation in document.ops:
        insert = operation.get('insert')
        if isinstance(insert, dict) and 'cell' in insert:
            return sum(
                op.str_length(o['insert'])
                if isinstance(o['insert'], str) else 1
                for o in insert['cell']['ops'])
    return None


def random_cross_move(rng, document, ref='r'):
    """A move whose halves may sit at root or inside the cell embed."""
    units = doc_units(document)
    root_length = len(units)
    embed_position = units.index('e') if 'e' in units else None
    inner = cell_length(document)
    kinds = ['root']
    if embed_position is not None and inner and inner >= 2:
        kinds += ['cell-to-root', 'root-to-cell', 'inside-cell']
    kind = rng.choice(kinds)
    if kind == 'root':
        return random_move(rng, root_length, ref)
    if kind == 'inside-cell':
        child = random_move(rng, inner, ref)
        return (MoveDelta().retain(embed_position)
                .push({'retain': {'cell': {'ops': child.ops}}}))
    if kind == 'cell-to-root':
        size = rng.randint(1, min(3, inner))
        source = rng.randint(0, inner - size)
        target = rng.randint(0, root_length)
        patch = cell_patch({'retain': source} if source else {'retain': 0},
                           {'cut': {'ref': ref, 'length': size}})
        patch['retain']['cell']['ops'] = [o for o in patch['retain']['cell']['ops']
                                          if o != {'retain': 0}]
        events = [(embed_position, 0, patch), (target, 1, {
            'paste': {'ref': ref, 'start': 0, 'length': size}})]
        events.sort(key=lambda event: (event[0], rng.random()))
        delta = MoveDelta()
        position = 0
        for target_position, _, operation in events:
            delta.retain(max(0, target_position - position))
            position = max(position, target_position)
            delta.push(operation)
            if 'retain' in operation:  # the embed consumes one unit
                position += 1
        return delta
    # root-to-cell: cut root text (avoiding the embed), paste inside
    spans = []
    run_start = None
    for index, unit in enumerate(units + ['e']):
        if unit == 't' and run_start is None:
            run_start = index
        elif unit != 't' and run_start is not None:
            spans.append((run_start, index))
            run_start = None
    spans = [s for s in spans if s[1] - s[0] >= 1]
    if not spans:
        return random_move(rng, root_length, ref)
    low, high = rng.choice(spans)
    size = rng.randint(1, min(3, high - low))
    source = rng.randint(low, high - size)
    inner_target = rng.randint(0, inner)
    child_ops = []
    if inner_target:
        child_ops.append({'retain': inner_target})
    child_ops.append({'paste': {'ref': ref, 'start': 0, 'length': size}})
    events = [(source, 0, {'cut': {'ref': ref, 'length': size}}),
              (embed_position, 1, {'retain': {'cell': {'ops': child_ops}}})]
    events.sort(key=lambda event: event[0])
    delta = MoveDelta()
    position = 0
    for target_position, _, operation in events:
        delta.retain(max(0, target_position - position))
        position = max(position, target_position)
        delta.push(operation)
        position += operation['cut']['length'] if 'cut' in operation else 1
    return delta


def random_two_level_edit(rng, document):
    units = doc_units(document)
    inner = cell_length(document)
    delta = MoveDelta()
    position = 0
    while position < len(units):
        if units[position] == 'e':
            roll = rng.random()
            if roll < 0.4 and inner:
                child = random_ordinary(rng, inner)
                delta.push({'retain': {'cell': {'ops': child.ops}}})
            elif roll < 0.55:
                delta.delete(1)
            else:
                delta.retain(1)
            position += 1
            continue
        span = rng.randint(1, min(3, len(units) - position))
        if 't' * span != ''.join(units[position:position + span]):
            span = 1
        roll = rng.random()
        if roll < 0.35:
            delta.retain(span)
        elif roll < 0.55:
            delta.retain(span, italic=True)
        elif roll < 0.8:
            delta.delete(span)
        else:
            delta.insert('Y')
            continue
        position += span
    return delta


def test_fuzz_cross_level_moves(cell_handler):
    rng = random.Random(9)
    for _ in range(250):
        text = ''.join(rng.choice(LETTERS) for _ in range(rng.randint(2, 5)))
        base = (MoveDelta()
                .insert(''.join(rng.choice(LETTERS) for _ in range(rng.randint(1, 3))))
                .insert(cell(text))
                .insert(''.join(rng.choice(LETTERS) for _ in range(rng.randint(1, 3)))))
        move = random_cross_move(rng, base, ref='a')
        after = apply(base, move)
        # invert roundtrip
        assert after.compose(move.invert(base)) == base, move.ops
        # compose law with a later two-level edit; deleting a move-sourcing
        # embed now composes via a trash cut instead of raising
        later = random_two_level_edit(rng, after) if cell_length(after) is not None \
            else random_ordinary(rng, len(doc_units(after)))
        assert after.compose(later) == apply(base, move.compose(later)), \
            (move.ops, later.ops)
        # transform convergence against edits and against another move
        edit = random_two_level_edit(rng, base)
        second = random_cross_move(rng, base, ref='b')
        for pair in ((move, edit), (move, second)):
            for wins in (True, False):
                right_prime = pair[0].transform(pair[1], wins)
                left_prime = pair[1].transform(pair[0], not wins)
                assert (apply(base, pair[0]).compose(right_prime)
                        == apply(base, pair[1]).compose(left_prime)), \
                    (base.ops, pair[0].ops, pair[1].ops, wins)


def test_fuzz_multi_ref_deltas_and_transform_output_inverts():
    rng = random.Random(7)
    for _ in range(300):
        base = random_doc(rng)
        if len(base) < 2:
            continue
        first = random_move(rng, len(base), ref='a')
        after = apply(base, first)
        if len(after) < 2:
            continue
        # a composed delta carries several refs, split parts and chains
        combined = first.compose(random_move(rng, len(after), ref='c'))
        assert apply(base, combined).compose(combined.invert(base)) == base
        other = random_move(rng, len(base), ref='b')
        for combined_wins in (True, False):
            other_prime = combined.transform(other, combined_wins)
            combined_prime = other.transform(combined, not combined_wins)
            one = apply(base, combined).compose(other_prime)
            two = apply(base, other).compose(combined_prime)
            assert one == two, (base.ops, combined.ops, other.ops)
            # transform outputs invert against their own base
            assert (one.compose(other_prime.invert(apply(base, combined)))
                    == apply(base, combined))


# --- coordinates -------------------------------------------------------------

from delta import transform_coordinate  # noqa: E402


def test_coordinate_root_caret_matches_transform_position():
    move = MoveDelta().cut('r', 3).retain(3).paste('r', 0, 3)
    for index in range(7):
        assert transform_coordinate(move, (index,)) == \
            (move.transform_position(index),)


def test_coordinate_caret_follows_cell_to_root_move(cell_handler):
    # base: AB [cell 'Hello'] CD ; move 'Hello' after C
    move = MoveDelta([
        {'retain': 2},
        cell_patch({'cut': {'ref': 'm', 'length': 5}}),
        {'retain': 1},
        {'paste': {'ref': 'm', 'start': 0, 'length': 5}},
    ])
    # caret between 'e' and 'l' in the cell -> root, inside the pasted
    # span, which starts at root offset 4 after the move
    assert transform_coordinate(move, (2, 'ops', 2)) == (6,)
    # caret at the region start stays at the (emptied) source
    assert transform_coordinate(move, (2, 'ops', 0)) == (2, 'ops', 0)


def test_coordinate_caret_follows_root_to_cell_move(cell_handler):
    # move 'AB' into the cell after 'Hello'
    move = MoveDelta([
        {'cut': {'ref': 'm', 'length': 2}},
        cell_patch({'retain': 5}, {'paste': {'ref': 'm', 'start': 0, 'length': 2}}),
    ])
    assert transform_coordinate(move, (1,)) == (0, 'ops', 6)


def test_coordinate_caret_shifts_with_cell_edits(cell_handler):
    edit = MoveDelta([{'retain': 2}, cell_patch({'insert': 'xx'})])
    assert transform_coordinate(edit, (2, 'ops', 3)) == (2, 'ops', 5)
    assert transform_coordinate(edit, (1,)) == (1,)


def test_coordinate_embed_unit_follows_root_move():
    # embed at position 2 moves to the front
    move = MoveDelta().paste('r', 0, 1).retain(2).cut('r', 1)
    assert transform_coordinate(move, (2, 'ops', 1)) == (0, 'ops', 1)


def test_coordinate_unit_dies_with_its_embed():
    assert transform_coordinate(MoveDelta().retain(2).delete(1),
                                (2, 'ops', 1)) is None
    dropped = MoveDelta().retain(2).cut('r', 2).paste('r', 1, 1)
    assert transform_coordinate(dropped, (2, 'ops', 1)) is None
    # while a caret merely collapses
    assert transform_coordinate(MoveDelta().retain(2).delete(3), (4,)) == (2,)


def test_coordinate_follows_cell_to_cell_move(cell_handler):
    move = MoveDelta([
        cell_patch({'cut': {'ref': 'x', 'length': 3}}),
        cell_patch({'retain': 3}, {'paste': {'ref': 'x', 'start': 0, 'length': 3}}),
    ])
    assert transform_coordinate(move, (0, 'ops', 1)) == (1, 'ops', 4)


# --- three-level documents: recursion across nested cells -------------------

def deep_base():
    inner = {'cell': {'ops': [{'insert': 'wxyz'}]}}
    middle = {'cell': {'ops': [
        {'insert': 'mno'}, {'insert': inner}, {'insert': 'pq'}]}}
    return MoveDelta().insert('AB').insert(middle).insert('CD')


def at_level(level, ops):
    """Wrap ops to address root (0), the middle cell (1), or the inner
    cell (2) of the deep_base document shape."""
    if level == 0:
        return MoveDelta(list(ops))
    if level == 2:
        ops = [{'retain': 3}, {'retain': {'cell': {'ops': list(ops)}}}]
    return MoveDelta([{'retain': 2}, {'retain': {'cell': {'ops': list(ops)}}}])


def deep_moves(ref):
    return [
        # inner cell -> root
        MoveDelta([{'retain': 2},
                   {'retain': {'cell': {'ops': [{'retain': 3},
                       {'retain': {'cell': {'ops': [
                           {'cut': {'ref': ref, 'length': 4}}]}}}]}}},
                   {'retain': 1},
                   {'paste': {'ref': ref, 'start': 0, 'length': 4}}]),
        # root -> inner cell
        MoveDelta([{'cut': {'ref': ref, 'length': 2}},
                   {'retain': {'cell': {'ops': [{'retain': 3},
                       {'retain': {'cell': {'ops': [{'retain': 4},
                           {'paste': {'ref': ref, 'start': 0, 'length': 2}}]}}}]}}}]),
        # middle cell -> inner cell (one subtree, two levels)
        MoveDelta([{'retain': 2},
                   {'retain': {'cell': {'ops': [
                       {'cut': {'ref': ref, 'length': 3}},
                       {'retain': {'cell': {'ops': [{'retain': 4},
                           {'paste': {'ref': ref, 'start': 0, 'length': 3}}]}}}]}}}]),
        # inner cell -> middle cell
        MoveDelta([{'retain': 2},
                   {'retain': {'cell': {'ops': [{'retain': 1},
                       {'paste': {'ref': ref, 'start': 1, 'length': 2}},
                       {'retain': 2},
                       {'retain': {'cell': {'ops': [
                           {'cut': {'ref': ref, 'length': 4}}]}}}]}}}]),
    ]


def test_deep_moves_apply_compose_invert(cell_handler):
    base = deep_base()
    for move in deep_moves('m'):
        after = apply(base, move)
        assert after.compose(move.invert(base)) == base, move.ops
        later = MoveDelta().retain(1).insert('X')
        assert after.compose(later) == apply(base, move.compose(later)), move.ops


def test_fuzz_deep_transform_convergence(cell_handler):
    base = deep_base()
    lengths = {0: 5, 1: 6, 2: 4}
    rng = random.Random(10)
    moves = deep_moves('a')
    for _ in range(150):
        move = rng.choice(moves)
        level = rng.randint(0, 2)
        edit = at_level(level, random_ordinary(rng, lengths[level]).ops)
        second = rng.choice(deep_moves('b'))
        for pair in ((move, edit), (move, second)):
            for wins in (True, False):
                right_prime = pair[0].transform(pair[1], wins)
                left_prime = pair[1].transform(pair[0], not wins)
                assert (apply(base, pair[0]).compose(right_prime)
                        == apply(base, pair[1]).compose(left_prime)), \
                    (pair[0].ops, pair[1].ops, wins)


def test_deep_coordinates_follow_moves(cell_handler):
    move_out, move_in, move_down, move_up = deep_moves('m')
    # caret in the inner cell follows its text to root ('wxyz' lands at 4)
    assert transform_coordinate(move_out, (2, 'ops', 3, 'ops', 2)) == (6,)
    # root caret follows into the inner cell
    assert transform_coordinate(move_in, (1,)) == (0, 'ops', 3, 'ops', 5)
    # middle-cell caret follows one level deeper
    assert transform_coordinate(move_down, (2, 'ops', 1)) == (2, 'ops', 0, 'ops', 5)


# --- review regressions: compose outputs are first-class citizens -----------

def trash_read_delta():
    """The shape compose emits when a move-sourcing embed is deleted."""
    return MoveDelta([
        {'retain': 2}, {'cut': {'ref': 'trash', 'length': 1}}, {'retain': 1},
        {'paste': {'ref': 'trash', 'unit': 0, 'path': ['ops'],
                   'start': 1, 'length': 3}}])


def test_transform_is_total_over_trash_reads(cell_handler):
    base = MoveDelta().insert('AB').insert(cell('Hello')).insert('CD')
    composed = trash_read_delta()
    edits = (
        MoveDelta().insert('X'),
        MoveDelta().retain(4).delete(1),
        cell_edit := MoveDelta([{'retain': 2},
                                cell_patch({'retain': 1}, {'delete': 1})]),
        MoveDelta([{'retain': 2},
                   cell_patch({'retain': 2},
                              {'retain': 2, 'attributes': {'bold': True}})]),
        MoveDelta([{'retain': 2}, cell_patch({'retain': 2}, {'insert': 'Z'})]),
    )
    for edit in edits:
        for wins in (True, False):
            right_prime = composed.transform(edit, wins)
            left_prime = edit.transform(composed, not wins)
            assert (apply(base, composed).compose(right_prime)
                    == apply(base, edit).compose(left_prime)), (edit.ops, wins)
    # the read survives renumbering rather than being silently dropped
    kept = cell_edit.transform(composed, False)
    assert any('path' in o.get('paste', {}) for o in kept.ops), kept.ops


def test_compose_chain_preserves_trash_read_keys(cell_handler):
    base = MoveDelta().insert('AB').insert(cell('Hello')).insert('CD')
    composed = trash_read_delta()
    later = MoveDelta().retain(3).cut('z', 3).retain(1).paste('z', 0, 3)
    chained = composed.compose(later)
    spec = next(o['paste'] for o in chained.ops if 'paste' in o)
    assert spec.get('path') == ['ops'] and 'unit' in spec
    assert apply(base, composed).compose(later) == apply(base, chained)


def test_compose_retargets_base_backed_trash_reads(cell_handler):
    base = MoveDelta().insert('AB').insert(cell('Hello')).insert('CD')
    first = MoveDelta().retain(2).insert('PQ')
    read = MoveDelta([
        {'retain': 2},
        {'cut': {'ref': 't', 'length': 3}},  # P, Q and the cell
        {'retain': 1},
        {'paste': {'ref': 't', 'unit': 2, 'path': ['ops'],
                   'start': 0, 'length': 5}}])
    composed = first.compose(read)
    spec = next(o['paste'] for o in composed.ops if 'paste' in o)
    assert spec['unit'] == 0  # re-targeted after the inserts were absorbed
    assert apply(base, first).compose(read) == apply(base, composed)


def test_transform_handles_moves_riding_paste_changes(cell_handler):
    base = MoveDelta().insert('AB').insert(cell('Hello')).insert('CD')
    outer = MoveDelta().retain(2).cut('c', 1).retain(2).paste('c', 0, 1)
    inner = MoveDelta([{'retain': 4},
                       cell_patch({'cut': {'ref': 'm', 'length': 2}},
                                  {'retain': 3},
                                  {'paste': {'ref': 'm', 'start': 0,
                                             'length': 2}})])
    combo = outer.compose(inner)  # the inner move rides the paste change
    edit = MoveDelta([{'retain': 2}, cell_patch({'insert': 'Z'})])
    for wins in (True, False):
        right_prime = combo.transform(edit, wins)
        left_prime = edit.transform(combo, not wins)
        assert (apply(base, combo).compose(right_prime)
                == apply(base, edit).compose(left_prime)), wins


def test_package_delta_is_move_aware():
    import delta as package
    assert package.Delta is MoveDelta
    # slicing keeps the move-aware class
    assert isinstance(MoveDelta().insert('abc')[0:2], MoveDelta)


def test_transform_follows_destination_into_another_embed(cell_handler):
    base = MoveDelta().insert('ab').insert(cell('F')).insert(cell('G'))
    mover = MoveDelta([
        {'cut': {'ref': 'x', 'length': 2}},
        cell_patch({'retain': 1}, {'paste': {'ref': 'x', 'start': 0,
                                             'length': 2}})])
    other = MoveDelta([
        {'retain': 2, 'attributes': {'bold': True}},
        {'cut': {'ref': 'f', 'length': 1}},
        cell_patch({'retain': 1}, {'paste': {'ref': 'f', 'start': 0,
                                             'length': 1}})])
    for wins in (True, False):
        right_prime = mover.transform(other, wins)
        left_prime = other.transform(mover, not wins)
        assert (apply(base, mover).compose(right_prime)
                == apply(base, other).compose(left_prime)), wins


def test_coordinate_maps_through_a_paste_change(cell_handler):
    base = MoveDelta().insert('xy').insert(cell('Hello')).insert('z')
    move = MoveDelta([
        {'retain': 2},
        {'cut': {'ref': 'e', 'length': 1}},
        {'retain': 1},
        {'paste': {'ref': 'e', 'start': 0, 'length': 1,
                   'change': {'cell': {'ops': [{'delete': 2}]}}}}])
    assert transform_coordinate(move, (2, 'ops', 3)) == (3, 'ops', 1)


def test_lower_handles_cross_level_moves_and_trash_reads(cell_handler):
    base = MoveDelta().insert('AB').insert(cell('Hello')).insert('CD')
    cross = MoveDelta([
        {'retain': 2},
        cell_patch({'cut': {'ref': 'm', 'length': 5}}),
        {'retain': 1},
        {'paste': {'ref': 'm', 'start': 0, 'length': 5}}])
    for move in (cross, trash_read_delta()):
        lowered = MoveDelta(move.lower(base).ops)
        assert not has_moves(lowered)
        assert apply(base, lowered) == apply(base, move), move.ops


def test_check_reports_malformed_specs():
    with pytest.raises(ValueError):
        check(MoveDelta([{'cut': {'ref': 'r'}}]))
    with pytest.raises(ValueError):
        check(MoveDelta([{'paste': {'ref': 'r', 'length': 3}},
                         {'cut': {'ref': 'r', 'length': 3}}]))
    # malformed specs never crash length-consuming paths either
    assert len(MoveDelta([{'paste': {'ref': 'r', 'start': 0}}])) == 1


def test_fuzz_closure_compose_outputs_transform(cell_handler):
    """Feed compose's own output vocabulary (trash reads, change-nested
    moves, chained refs) back through transform and compose."""
    rng = random.Random(13)
    for _ in range(120):
        text = ''.join(rng.choice(LETTERS) for _ in range(rng.randint(2, 5)))
        base = (MoveDelta()
                .insert(''.join(rng.choice(LETTERS) for _ in range(rng.randint(1, 3))))
                .insert(cell(text))
                .insert(''.join(rng.choice(LETTERS) for _ in range(rng.randint(1, 3)))))
        move = random_cross_move(rng, base, ref='a')
        after = apply(base, move)
        later = random_two_level_edit(rng, after) if cell_length(after) is not None \
            else random_ordinary(rng, len(doc_units(after)))
        composed = move.compose(later)  # may contain trash reads, chains
        after2 = apply(base, composed)
        assert after.compose(later) == after2
        # the composed output must transform against concurrent edits
        edit = random_two_level_edit(rng, base)
        for wins in (True, False):
            right_prime = composed.transform(edit, wins)
            left_prime = edit.transform(composed, not wins)
            assert (after2.compose(right_prime)
                    == apply(base, edit).compose(left_prime)), \
                (base.ops, composed.ops, edit.ops, wins)
        # ...compose onwards...
        onwards = (random_two_level_edit(rng, after2)
                   if cell_length(after2) is not None
                   else random_ordinary(rng, len(doc_units(after2))))
        assert after2.compose(onwards) == apply(base, composed.compose(onwards))
        # ...invert, and lower
        assert after2.compose(composed.invert(base)) == base
        assert apply(base, MoveDelta(composed.lower(base).ops)) == after2


# --- trash reads racing concurrent edits and moves ---------------------------

def test_reads_race_cell_edits_convergently(cell_handler):
    base = MoveDelta().insert('AB').insert(cell('Hello')).insert('CD')
    composed = trash_read_delta()  # reads 'ell' out of the trashed cell
    edits = (
        MoveDelta([{'retain': 2}, cell_patch({'delete': 1})]),
        MoveDelta([{'retain': 2},
                   cell_patch({'retain': 1}, {'delete': 1})]),
        MoveDelta([{'retain': 2},
                   cell_patch({'retain': 2},
                              {'retain': 2, 'attributes': {'bold': True}})]),
        MoveDelta([{'retain': 2}, cell_patch({'retain': 2}, {'insert': 'Z'})]),
        # a within-cell move: rearranged dying content loses to the read
        MoveDelta([{'retain': 2},
                   cell_patch({'cut': {'ref': 'w', 'length': 1}},
                              {'retain': 2},
                              {'paste': {'ref': 'w', 'start': 0, 'length': 1}})]),
        # a cell-to-root move: its windows survive, so it wins its claim
        MoveDelta([{'retain': 2},
                   cell_patch({'retain': 1}, {'cut': {'ref': 'o', 'length': 2}}),
                   {'paste': {'ref': 'o', 'start': 0, 'length': 2}}]),
    )
    for edit in edits:
        for wins in (True, False):
            right_prime = composed.transform(edit, wins)
            left_prime = edit.transform(composed, not wins)
            assert (apply(base, composed).compose(right_prime)
                    == apply(base, edit).compose(left_prime)), (edit.ops, wins)


def test_read_against_gap_delete_converges(cell_handler):
    base = MoveDelta().insert('A').insert(cell('Hello')).insert('BCD')
    composed = MoveDelta([
        {'retain': 1},
        {'paste': {'ref': 'trash', 'start': 0, 'length': 3,
                   'unit': 0, 'path': ['ops']}},
        {'cut': {'ref': 'trash', 'length': 1}}])
    # the concurrent move's gap deletes the very embed the read addresses
    mover = MoveDelta([
        {'cut': {'ref': 'b', 'length': 2}},
        {'retain': 1},
        {'paste': {'ref': 'b', 'start': 0, 'length': 1}}])
    # a copy out of trash never survives a concurrent claim on its
    # source: with either priority both sides lose the content
    for wins in (True, False):
        right_prime = composed.transform(mover, not wins)
        left_prime = mover.transform(composed, wins)
        left = apply(base, composed).compose(right_prime)
        right = apply(base, mover).compose(left_prime)
        assert left == right, wins
        assert left == MoveDelta().insert('BACD')


# ---------------------------------------------------------------------------
# moves into newly inserted embeds: the paste rides the insert's payload
# ---------------------------------------------------------------------------

def carried(ref, start, length, attrs=None, wrap=0, filler=False):
    paste_op = {'paste': {'ref': ref, 'start': start, 'length': length}}
    if attrs:
        paste_op['attributes'] = dict(attrs)
    ops = [{'insert': 'x'}, paste_op] if filler else [paste_op]
    payload = {'cell': {'ops': ops}}
    for _ in range(wrap):
        payload = {'cell': {'ops': [{'insert': payload}]}}
    return {'insert': payload}


def test_move_into_inserted_embed_composes(cell_handler):
    doc = MoveDelta().insert('Hello world')
    move = (MoveDelta().cut('m', 5).retain(6)
            .push({'insert': {'cell': {'ops': [
                {'insert': '['},
                {'paste': {'ref': 'm', 'start': 0, 'length': 5}},
                {'insert': ']'}]}}}))
    assert apply(doc, move) == MoveDelta([
        {'insert': ' world'},
        {'insert': {'cell': {'ops': [{'insert': '[Hello]'}]}}}])


def test_move_into_inserted_embed_inverts(cell_handler):
    doc = MoveDelta().insert('Hello world')
    move = (MoveDelta().cut('m', 5).retain(6)
            .push(carried('m', 0, 5)))
    inverse = move.invert(doc)
    assert apply(doc, move).compose(inverse) == doc
    assert not has_moves(inverse)  # the whole copy dies with the insert


def random_carried_move(rng, length, ref='c'):
    """A move whose paste windows ride newly inserted embeds — possibly
    split with a dropped middle, nested two levels deep, or mixed with a
    plain root window."""
    size = rng.randint(1, min(4, length - 1))
    source = rng.randint(0, length - size)
    windows = [(0, size)]
    if size >= 2 and rng.random() < 0.5:
        left = rng.randint(1, size - 1)
        right = rng.randint(left, size - 1)
        windows = [w for w in [(0, left), (right, size - right)] if w[1] > 0]
    carries = [True] + [rng.random() < 0.5 for _ in windows[1:]]
    events = [(source, 0, 'cut', None, None)]
    for order, (window, carry) in enumerate(zip(windows, carries), 1):
        target = rng.choice([q for q in range(length + 1)
                             if q <= source or q >= source + size])
        events.append((target, order, 'carry' if carry else 'paste',
                       window, rng.choice(FUZZ_ATTRS)))
    events.sort(key=lambda event: (event[0], rng.random()))
    delta = MoveDelta()
    position = 0
    for target, _, kind, window, attrs in events:
        delta.retain(max(0, target - position))
        position = max(position, target)
        if kind == 'cut':
            delta.cut(ref, size)
            position += size
        elif kind == 'paste':
            delta.paste(ref, window[0], window[1], **(attrs or {}))
        else:
            delta.push(carried(ref, window[0], window[1], attrs,
                               wrap=1 if rng.random() < 0.3 else 0,
                               filler=rng.random() < 0.5))
    return delta


def test_fuzz_insert_carried_compose_laws(cell_handler):
    rng = random.Random(61)
    for _ in range(300):
        base = random_doc(rng)
        if len(base) < 2:
            continue
        move = random_carried_move(rng, len(base))
        after = apply(base, move)
        edit = random_ordinary(rng, len(after))
        assert after.compose(edit) == apply(base, move.compose(edit))
        pre = random_ordinary(rng, len(base))
        assert (apply(apply(base, pre), move)
                == apply(base, pre.compose(move)))


def test_fuzz_insert_carried_invert(cell_handler):
    rng = random.Random(62)
    for _ in range(300):
        base = random_doc(rng)
        if len(base) < 2:
            continue
        move = random_carried_move(rng, len(base))
        assert apply(base, move).compose(move.invert(base)) == base


def test_fuzz_insert_carried_transform_vs_ordinary(cell_handler):
    rng = random.Random(63)
    for _ in range(300):
        base = random_doc(rng)
        if len(base) < 2:
            continue
        move = random_carried_move(rng, len(base))
        edit = random_ordinary(rng, len(base))
        for priority in (True, False):
            move_prime = edit.transform(move, not priority)
            edit_prime = move.transform(edit, priority)
            assert (apply(apply(base, move), edit_prime)
                    == apply(apply(base, edit), move_prime)), priority


def test_fuzz_insert_carried_transform_vs_move(cell_handler):
    rng = random.Random(64)
    for _ in range(300):
        base = random_doc(rng)
        if len(base) < 3:
            continue
        left = random_carried_move(rng, len(base), ref='c')
        right = random_move(rng, len(base), ref='r')
        for wins in (True, False):
            right_prime = left.transform(right, wins)
            left_prime = right.transform(left, not wins)
            assert (apply(apply(base, left), right_prime)
                    == apply(apply(base, right), left_prime)), wins


def test_compose_splits_carried_windows(cell_handler):
    # an earlier insert into the source span splits the window inside
    # the carried payload, keeping the literal text between the parts
    pre = MoveDelta().retain(2).insert('ZZ')
    move = (MoveDelta().cut('m', 5).retain(6).push(carried('m', 0, 5)))
    combo = pre.compose(move)
    doc = MoveDelta().insert('Hello world')
    assert apply(apply(doc, pre), move) == apply(doc, combo)
    payload = combo.ops[-1]['insert']['cell']['ops']
    assert {'insert': 'ZZ'} in payload


def test_fuzz_insert_carried_closure(cell_handler):
    """Compose outputs carrying split/symbolic windows stay transformable
    and lowerable."""
    rng = random.Random(65)
    for _ in range(200):
        base = random_doc(rng)
        pre = random_ordinary(rng, len(base))
        mid = apply(base, pre)
        if len(mid) < 2:
            continue
        move = random_carried_move(rng, len(mid))
        combo = pre.compose(move)
        after = apply(base, combo)
        assert not has_moves(combo.lower(base)) and \
            apply(base, combo.lower(base)) == after
        edit = random_ordinary(rng, len(base))
        for priority in (True, False):
            combo_prime = edit.transform(combo, not priority)
            edit_prime = combo.transform(edit, priority)
            assert (apply(after, edit_prime)
                    == apply(apply(base, edit), combo_prime)), priority


def test_fuzz_reads_race_concurrent_movers(cell_handler):
    """The refusal this replaces: any concurrent move may now claim (and
    gap-drop) the very embed a trash read addresses, with either
    priority — the copy dies rather than raising."""
    rng = random.Random(66)
    base = MoveDelta().insert('AB').insert(cell('Hello')).insert('CD')
    for _ in range(400):
        composed = trash_read_delta()
        mover = random_move(rng, len(base))
        for wins in (True, False):
            right_prime = composed.transform(mover, not wins)
            left_prime = mover.transform(composed, wins)
            assert (apply(base, composed).compose(right_prime)
                    == apply(base, mover).compose(left_prime)), (mover.ops,
                                                                 wins)


def test_diff_delegates_to_embed_handlers(cell_handler):
    a = MoveDelta().insert('x').insert(cell('Hello')).insert('y')
    b = MoveDelta().insert('x').insert(cell('Help!')).insert('y')
    d = a.diff(b)
    # a structured patch, not a replacement of the whole embed
    assert d.ops[1].get('retain', {}).get('cell'), d.ops
    assert a.compose(d) == b
    # unequal embeds without a diff-capable handler still replace
    fig_a = MoveDelta().insert({'image': '1.png'})
    fig_b = MoveDelta().insert({'image': '2.png'})
    d = fig_a.diff(fig_b)
    assert d == MoveDelta().insert({'image': '2.png'}).delete(1)
    assert fig_a.compose(d) == fig_b


def test_fuzz_embed_diff_law(cell_handler):
    rng = random.Random(67)
    for _ in range(200):
        a = MoveDelta().insert(cell(''.join(
            rng.choice(LETTERS) for _ in range(rng.randint(1, 6)))))
        b = MoveDelta().insert(cell(''.join(
            rng.choice(LETTERS) for _ in range(rng.randint(1, 6)))))
        assert a.compose(a.diff(b)) == b


def test_compose_splits_retain_carried_windows(cell_handler):
    # like the insert-carried split, but the window rides a typed retain
    # payload addressing an EXISTING embed
    doc = MoveDelta().insert('Hello ').insert(cell('x')).insert('World')
    pre = MoveDelta().retain(2).insert('ZZ')
    move = MoveDelta([
        {'cut': {'ref': 'm', 'length': 5}},
        {'retain': 3},
        {'retain': {'cell': {'ops': [
            {'retain': 1},
            {'paste': {'ref': 'm', 'start': 0, 'length': 5}}]}}}])
    combo = pre.compose(move)
    assert apply(apply(doc, pre), move) == apply(doc, combo)
    payload = combo.ops[-1]['retain']['cell']['ops']
    assert {'insert': 'ZZ'} in payload  # the window split around it


def test_compose_expands_a_nested_child_move_once(cell_handler):
    base = MoveDelta().insert({'cell': {'ops': [
        {'insert': cell('AB')}, {'insert': 'x'}]}})
    axis_move = [
        {'paste': {'ref': 'axis', 'start': 0, 'length': 1}},
        {'retain': 1},
        {'cut': {'ref': 'axis', 'length': 1}},
    ]
    first = MoveDelta([cell_patch(cell_patch(*axis_move))])
    second = MoveDelta([cell_patch(
        {'cut': {'ref': 'block', 'length': 1}},
        {'retain': 1},
        {'paste': {'ref': 'block', 'start': 0, 'length': 1}},
    )])

    composed = first.compose(second)

    check(composed)
    carried = composed.ops[0]['retain']['cell']['ops'][2]['paste']['change']
    assert carried['cell']['ops'] == axis_move
    assert apply(base, composed) == apply(base, first).compose(second)


def test_fuzz_retain_carried_compose(cell_handler):
    rng = random.Random(68)
    for _ in range(200):
        text = ''.join(rng.choice(LETTERS) for _ in range(rng.randint(2, 5)))
        base = (MoveDelta()
                .insert(''.join(rng.choice(LETTERS) for _ in range(rng.randint(1, 3))))
                .insert(cell(text))
                .insert(''.join(rng.choice(LETTERS) for _ in range(rng.randint(1, 3)))))
        pre = random_ordinary(rng, len(base))
        mid = apply(base, pre)
        if len(mid) < 3 or not any(
                isinstance(o.get('insert'), dict) for o in mid.ops):
            continue
        move = random_cross_move(rng, mid, ref='a')
        combo = pre.compose(move)
        assert apply(mid, move) == apply(base, combo), (pre.ops, move.ops)


# ---------------------------------------------------------------------------
# moves between two sibling cells
# ---------------------------------------------------------------------------

def cell_sites(document):
    """(root position, content length) of every cell in the document."""
    sites = []
    position = 0
    for operation in document.ops:
        insert = operation['insert']
        if isinstance(insert, dict) and 'cell' in insert:
            sites.append((position, sum(
                op.str_length(o['insert'])
                if isinstance(o.get('insert'), str) else 1
                for o in insert['cell']['ops'])))
        position += op.length(operation)
    return sites


def two_cell_base(rng):
    def text(low, high):
        return ''.join(rng.choice(LETTERS)
                       for _ in range(rng.randint(low, high)))
    return (MoveDelta().insert(text(1, 3)).insert(cell(text(2, 5)))
            .insert(text(1, 3)).insert(cell(text(2, 5))).insert(text(1, 3)))


def random_cell_to_cell_move(rng, document, ref='r'):
    """Cut in one cell's child sequence, paste in a sibling cell's."""
    sites = cell_sites(document)
    src, dst = rng.sample(range(len(sites)), 2)
    src_pos, src_len = sites[src]
    dst_pos, dst_len = sites[dst]
    size = rng.randint(1, min(3, src_len))
    offset = rng.randint(0, src_len - size)
    target = rng.randint(0, dst_len)
    attrs = rng.choice(FUZZ_ATTRS)
    cut = cell_patch(*([{'retain': offset}] if offset else []),
                     {'cut': {'ref': ref, 'length': size}})
    paste_op = {'paste': {'ref': ref, 'start': 0, 'length': size}}
    if attrs:
        paste_op = {**paste_op, 'attributes': dict(attrs)}
    paste = cell_patch(*([{'retain': target}] if target else []), paste_op)
    (first_pos, first), (second_pos, second) = sorted(
        [(src_pos, cut), (dst_pos, paste)], key=lambda site: site[0])
    return (MoveDelta().retain(first_pos).push(first)
            .retain(second_pos - first_pos - 1).push(second))


def random_two_cell_edit(rng, document):
    """Ordinary edits inside each cell; root text is retained."""
    delta = MoveDelta()
    for operation in document.ops:
        insert = operation['insert']
        if isinstance(insert, dict):
            inner = sum(op.str_length(o['insert'])
                        if isinstance(o.get('insert'), str) else 1
                        for o in insert['cell']['ops'])
            if rng.random() < 0.6 and inner:
                child = random_ordinary(rng, inner)
                delta.push({'retain': {'cell': {'ops': child.ops}}})
            else:
                delta.retain(1)
        else:
            delta.retain(op.length(operation))
    return delta


def test_cell_to_cell_transform_follows_source_edits(cell_handler):
    base = MoveDelta().insert(cell('one')).insert(cell('two'))
    move = MoveDelta([
        cell_patch({'cut': {'ref': 'x', 'length': 3}}),
        cell_patch({'retain': 3},
                   {'paste': {'ref': 'x', 'start': 0, 'length': 3}})])
    edit = MoveDelta([cell_patch({'retain': 3, 'attributes': {'bold': True}})])
    for priority in (True, False):
        move_prime = edit.transform(move, not priority)
        edit_prime = move.transform(edit, priority)
        left = apply(apply(base, move), edit_prime)
        right = apply(apply(base, edit), move_prime)
        assert left == right, priority
        assert left.ops == [
            {'insert': {'cell': {'ops': []}}},
            {'insert': {'cell': {'ops': [
                {'insert': 'two'},
                {'insert': 'one', 'attributes': {'bold': True}}]}}}]


def test_cell_to_cell_move_inverts(cell_handler):
    base = MoveDelta().insert(cell('one')).insert(cell('two'))
    move = MoveDelta([
        cell_patch({'cut': {'ref': 'x', 'length': 3}}),
        cell_patch({'retain': 3},
                   {'paste': {'ref': 'x', 'start': 0, 'length': 3}})])
    inverse = move.invert(base)
    assert apply(base, move).compose(inverse) == base
    assert has_moves(inverse)  # the inverse is itself a move


def test_compose_splits_cell_to_cell_paste(cell_handler):
    base = MoveDelta().insert(cell('one')).insert(cell('two'))
    move = MoveDelta([
        cell_patch({'cut': {'ref': 'x', 'length': 3}}),
        cell_patch({'retain': 3},
                   {'paste': {'ref': 'x', 'start': 0, 'length': 3}})])
    later = MoveDelta([{'retain': 1},
                       cell_patch({'retain': 4}, {'insert': '!'})])
    composed = move.compose(later)
    assert apply(base, composed) == apply(base, move).compose(later)
    payload = composed.ops[-1]['retain']['cell']['ops']
    assert {'insert': '!'} in payload  # the window split around it


def test_fuzz_cell_to_cell_moves(cell_handler):
    rng = random.Random(80)
    for _ in range(200):
        base = two_cell_base(rng)
        move = random_cell_to_cell_move(rng, base)
        after = apply(base, move)
        assert after.compose(move.invert(base)) == base, move.ops
        edit = random_two_cell_edit(rng, after)
        assert after.compose(edit) == apply(base, move.compose(edit))
        concurrent = random_two_cell_edit(rng, base)
        for priority in (True, False):
            move_prime = concurrent.transform(move, not priority)
            edit_prime = move.transform(concurrent, priority)
            assert (apply(apply(base, move), edit_prime)
                    == apply(apply(base, concurrent), move_prime)), priority


def test_fuzz_cell_to_cell_vs_concurrent_moves(cell_handler):
    rng = random.Random(81)
    for _ in range(200):
        base = two_cell_base(rng)
        left = random_cell_to_cell_move(rng, base, ref='a')
        right = (random_cell_to_cell_move(rng, base, ref='z')
                 if rng.random() < 0.6
                 else random_move(rng, len(base), ref='z'))
        for wins in (True, False):
            right_prime = left.transform(right, wins)
            left_prime = right.transform(left, not wins)
            assert (apply(apply(base, left), right_prime)
                    == apply(apply(base, right), left_prime)), wins


def test_compose_expands_a_window_riding_a_paste_change(cell_handler):
    # B moves content into a cell that A itself moves: B's window rides
    # the change payload of A's paste and must re-slice through B's cut
    base = MoveDelta().insert('eg').insert(cell('cd')).insert('cb')
    A = MoveDelta([
        {'paste': {'ref': 'a', 'start': 0, 'length': 2}},
        {'retain': 2},
        {'cut': {'ref': 'a', 'length': 3}},
        {'paste': {'ref': 'a', 'start': 2, 'length': 1}}])
    B = MoveDelta([
        cell_patch({'retain': 1},
                   {'paste': {'ref': 'b', 'start': 0, 'length': 1}}),
        {'cut': {'ref': 'b', 'length': 1}}])
    AB = A.compose(B)
    check(AB)
    assert apply(base, AB) == apply(apply(base, A), B)


def test_compose_never_expands_handler_composed_windows_twice(cell_handler):
    # A moves a unit out of the cell; B moves root content (including
    # A's window) back in.  The cell patches compose through the
    # handler, which already expands B's window — the assemble pass
    # must treat that payload as sealed
    base = MoveDelta().insert('cg').insert(cell('bbb')).insert('ga')
    A = MoveDelta([
        {'retain': 2},
        cell_patch({'retain': 2}, {'cut': {'ref': 'a', 'length': 1}}),
        {'paste': {'ref': 'a', 'start': 0, 'length': 1}}])
    B = MoveDelta([
        {'retain': 2},
        cell_patch({'retain': 1},
                   {'paste': {'ref': 'b', 'start': 0, 'length': 3}}),
        {'cut': {'ref': 'b', 'length': 3}}])
    AB = A.compose(B)
    check(AB)
    assert apply(base, AB) == apply(apply(base, A), B)


def test_fuzz_symbolic_outputs_stay_valid(cell_handler):
    """check() every symbolic compose, composed-inverse (undo) and
    transform output — the class of bug where each step applies but a
    nested window gets expanded twice along the way."""
    rng = random.Random(91)
    for _ in range(150):
        base = two_cell_base(rng)
        A = random_cell_to_cell_move(rng, base, 'a')
        after_a = apply(base, A)
        try:
            B = random_cell_to_cell_move(rng, after_a, 'b')
        except ValueError:
            continue
        AB = A.compose(B)
        check(AB)
        assert apply(base, AB) == apply(after_a, B)
        undo = B.invert(after_a).compose(A.invert(base))
        check(undo)
        assert apply(apply(base, AB), undo) == base
        edit = random_two_cell_edit(rng, base)
        for priority in (True, False):
            a_prime = edit.transform(A, not priority)
            e_prime = A.transform(edit, priority)
            check(a_prime)
            check(e_prime)
