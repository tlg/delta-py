from delta import Delta
from delta.block import BlockDelta
from delta.change import apply_change, compose_change


def test_attributed_compose_falls_back_when_structural_lowering_is_not_exact():
    base = Delta().insert('x', bold='true').insert('\n\n')
    first = {
        'delta': Delta().insert('x'),
        'blockDelta': BlockDelta().move(1, 2),
    }
    second = {
        'delta': Delta().insert('\n'),
        'blockDelta': BlockDelta().move(1, 2),
    }

    sequential = apply_change(apply_change(base, first), second)
    composed = compose_change(base, first, second)

    assert apply_change(base, composed) == sequential
    assert composed['blockDelta'] == BlockDelta()
