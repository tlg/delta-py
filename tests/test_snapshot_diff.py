"""
Contract tests for the deterministic typed snapshot diff (delta/diff.py).
"""

import gc
import random
import tracemalloc
from itertools import product

import pytest

from delta import Delta, MoveDelta
from delta.base import has_moves

LETTERS = ["a", "b", "c", "d", "\N{GRINNING FACE}"]
ATTRS = [None, {"bold": True}, {"bold": True, "i": 1}, {"i": 2}]


def fig(n):
    return {"fig": {"src": f"{n}.png"}}


def random_doc(rng):
    doc = MoveDelta()
    for _ in range(rng.randint(0, 6)):
        if rng.random() < 0.25:
            doc.insert(fig(rng.randint(1, 3)))
            continue
        text = "".join(rng.choice(LETTERS) for _ in range(rng.randint(1, 4)))
        attrs = rng.choice(ATTRS)
        doc.insert(text, **(attrs or {}))
    return doc


class RecordingHandler:
    """A diff-capable embed handler that counts its diff calls."""

    def __init__(self):
        self.calls = 0
        self.context = None

    @staticmethod
    def stream_paths(value):
        return [("ops",)] if isinstance(value, dict) and isinstance(value.get("ops"), list) else []

    def apply(self, a, b, _context):
        return b

    def compose(self, a, b, _context):
        return b

    def transform(self, a, b, priority, _context):
        return b

    def invert(self, change, base, _context):
        return base

    def diff(self, a, b, context):
        self.calls += 1
        self.context = context
        return {"src": b["src"]}


@pytest.fixture
def fig_handler():
    handler = RecordingHandler()
    Delta.register_embed("fig", handler)
    yield handler
    Delta.unregister_embed("fig")


def test_fuzz_reconstruction(fig_handler):
    rng = random.Random(70)
    for _ in range(500):
        a, b = random_doc(rng), random_doc(rng)
        assert a.compose(a.diff(b)) == b


def test_equal_documents_yield_an_empty_delta(fig_handler):
    doc = MoveDelta().insert("a", bold=True).insert(fig(1)).insert("b")
    same = MoveDelta().insert("a", bold=True).insert(fig(1)).insert("b")
    assert doc.diff(same) == MoveDelta()
    assert fig_handler.calls == 0


def test_deterministic_repeated_runs(fig_handler):
    rng = random.Random(71)
    for _ in range(100):
        a, b = random_doc(rng), random_doc(rng)
        first = a.diff(b)
        assert all(a.diff(b) == first for _ in range(3))


def test_chunking_independence():
    whole = MoveDelta([{"insert": "hello world"}])
    chunked = MoveDelta([{"insert": "hel"}, {"insert": "lo wo"}, {"insert": "rld"}])
    target = MoveDelta().insert("hello brave world")
    assert whole.diff(target) == chunked.diff(target)
    assert chunked.compose(chunked.diff(target)) == target


def test_dict_key_order_independence():
    a = MoveDelta([{"insert": {"fig": {"x": 1, "y": 2}}}])
    b = MoveDelta([{"insert": {"fig": {"y": 2, "x": 1}}}])
    assert a.diff(b) == MoveDelta()


def test_nul_text_never_aligns_with_an_embed():
    a = MoveDelta().insert("\x00")
    b = MoveDelta().insert(fig(1))
    d = a.diff(b)
    assert d == MoveDelta().insert(fig(1)).delete(1)
    assert a.compose(d) == b


def test_formatting_one_surrogate_half():
    doc = MoveDelta().insert("\N{GRINNING FACE}")
    half = doc.compose(MoveDelta().retain(1, bold=True))
    d = doc.diff(half)
    assert doc.compose(d) == half
    assert half.compose(half.diff(doc)) == doc


def test_unchanged_embed_never_calls_the_handler(fig_handler):
    a = MoveDelta().insert("x").insert(fig(1)).insert("y")
    b = MoveDelta().insert("xx").insert(fig(1))
    a.diff(b)
    assert fig_handler.calls == 0


def test_changed_embed_calls_the_handler_once(fig_handler):
    a = MoveDelta().insert(fig(1))
    b = MoveDelta().insert(fig(2))
    d = a.diff(b)
    assert fig_handler.calls == 1
    assert fig_handler.context is not None
    assert d.ops == [{"retain": {"fig": {"src": "2.png"}}}]


def test_embed_attribute_patch_rides_the_retain(fig_handler):
    a = MoveDelta().insert(fig(1), bold=True)
    b = MoveDelta().insert(fig(2))
    d = a.diff(b)
    assert d.ops == [{"retain": {"fig": {"src": "2.png"}}, "attributes": {"bold": None}}]


def test_different_embed_types_replace():
    a = MoveDelta().insert({"image": "1.png"})
    b = MoveDelta().insert({"video": "1.mp4"})
    d = a.diff(b)
    assert d == MoveDelta().insert({"video": "1.mp4"}).delete(1)


def test_missing_handler_replaces():
    a = MoveDelta().insert({"image": "1.png"})
    b = MoveDelta().insert({"image": "2.png"})
    d = a.diff(b)
    assert d == MoveDelta().insert({"image": "2.png"}).delete(1)


def test_handler_none_for_unequal_is_an_error():
    class NoneHandler(RecordingHandler):
        def diff(self, a, b, _context):
            return None

    Delta.register_embed("fig", NoneHandler())
    try:
        with pytest.raises(ValueError, match="equality only"):
            MoveDelta().insert(fig(1)).diff(MoveDelta().insert(fig(2)))
    finally:
        Delta.unregister_embed("fig")


def test_handler_not_implemented_requests_replacement():
    class OptOutHandler(RecordingHandler):
        def diff(self, a, b, _context):
            return NotImplemented

    Delta.register_embed("fig", OptOutHandler())
    try:
        d = MoveDelta().insert(fig(1)).diff(MoveDelta().insert(fig(2)))
        assert d == MoveDelta().insert(fig(2)).delete(1)
    finally:
        Delta.unregister_embed("fig")


def test_handler_patch_with_moves_is_an_error():
    class MovingHandler(RecordingHandler):
        def diff(self, a, b, _context):
            return {"ops": [{"cut": {"ref": "r", "length": 1}}, {"paste": {"ref": "r", "start": 0, "length": 1}}]}

    Delta.register_embed("fig", MovingHandler())
    try:
        with pytest.raises(ValueError, match="cut or paste"):
            MoveDelta().insert(fig(1)).diff(MoveDelta().insert(fig(2)))
    finally:
        Delta.unregister_embed("fig")


def test_diff_never_emits_moves(fig_handler):
    rng = random.Random(72)
    for _ in range(200):
        a, b = random_doc(rng), random_doc(rng)
        assert not has_moves(a.diff(b))


def test_repeated_embeds_retain_the_exact_match(fig_handler):
    a = MoveDelta().insert(fig(1)).insert(fig(2))
    b = MoveDelta().insert(fig(2)).insert(fig(3))
    d = a.diff(b)
    # the exact fig(2) is retained; never patch 1->2 and 2->3
    assert d.ops == [{"delete": 1}, {"retain": 1}, {"insert": fig(3)}]
    assert fig_handler.calls == 0
    assert a.compose(d) == b


def test_ambiguous_repeated_text_reconstructs_deterministically():
    a = MoveDelta().insert("aaaa")
    b = MoveDelta().insert("aaa")
    first = a.diff(b)
    assert a.compose(first) == b
    assert a.diff(b) == first


def test_bidirectional_myers_ties_are_canonical():
    cases = [
        ("a", "baa", [{"insert": "b"}, {"retain": 1}, {"insert": "a"}]),
        ("aa", "ba", [{"insert": "b"}, {"delete": 1}]),
        ("abb", "bab", [{"delete": 1}, {"retain": 1}, {"insert": "a"}]),
    ]
    for base, target, expected in cases:
        assert MoveDelta().insert(base).diff(MoveDelta().insert(target)).ops == expected


def test_exhaustive_small_binary_text_uses_a_longest_alignment():
    words = ["".join(chars) for length in range(6) for chars in product("ab", repeat=length)]

    def lcs_length(a, b):
        row = [0] * (len(b) + 1)
        for left in a:
            previous = 0
            for j, right in enumerate(b, 1):
                old = row[j]
                row[j] = previous + 1 if left == right else max(row[j - 1], row[j])
                previous = old
        return row[-1]

    for base in words:
        for target in words:
            ours = MoveDelta().insert(base) if base else MoveDelta()
            theirs = MoveDelta().insert(target) if target else MoveDelta()
            change = ours.diff(theirs)
            edits = sum(len(operation.get("insert", "")) + operation.get("delete", 0) for operation in change.ops)
            assert edits == len(base) + len(target) - 2 * lcs_length(base, target)
            assert ours.compose(change) == theirs


def test_common_edges_cross_chunks_utf16_and_embeds(fig_handler):
    base = MoveDelta(
        [
            {"insert": "p", "attributes": {"bold": True}},
            {"insert": "re\N{GRINNING FACE}", "attributes": {"bold": True}},
            {"insert": fig(1)},
            {"insert": "x"},
            {"insert": fig(2)},
            {"insert": "po", "attributes": {"i": True}},
            {"insert": "st", "attributes": {"i": True}},
        ]
    )
    target = MoveDelta(
        [
            {"insert": "pre", "attributes": {"bold": True}},
            {"insert": "\N{GRINNING FACE}", "attributes": {"bold": True}},
            {"insert": fig(1)},
            {"insert": "y"},
            {"insert": fig(2)},
            {"insert": "p", "attributes": {"i": True}},
            {"insert": "ost", "attributes": {"i": True}},
        ]
    )
    change = base.diff(target)
    assert change.ops == [{"retain": 6}, {"insert": "y"}, {"delete": 1}]

    def normalized(document):
        result = MoveDelta()
        for operation in document.ops:
            result.push(dict(operation))
        return result

    assert normalized(base.compose(change)) == normalized(target)
    assert fig_handler.calls == 0


def test_common_prefix_slice_preserves_empty_attributes():
    base = MoveDelta([{"insert": "ca", "attributes": {}}, {"insert": "bc", "attributes": {}}])
    target = MoveDelta([{"insert": "ccb", "attributes": {}}, {"insert": "ac"}])
    assert base.diff(target).ops == [{"retain": 1}, {"delete": 2}, {"retain": 1}, {"insert": "bac"}]


def test_common_edges_are_removed_before_atomization(monkeypatch):
    import delta.diff as snapshot

    lengths = []
    atomize = snapshot._atomize

    def recording_atomize(document):
        lengths.append(len(document))
        return atomize(document)

    monkeypatch.setattr(snapshot, "_atomize", recording_atomize)
    base = MoveDelta().insert("a" * 5_000 + "x" + "b" * 5_000)
    target = MoveDelta().insert("a" * 5_000 + "y" + "b" * 5_000)
    assert base.compose(base.diff(target)) == target
    assert lengths == [1, 1]


def test_edge_trimming_does_not_hide_a_zero_length_non_insert():
    base = MoveDelta([{"insert": "same"}, {"retain": 0}])
    with pytest.raises(ValueError, match="only insert ops"):
        base.diff(MoveDelta().insert("same target"))


def test_disjoint_diff_peak_memory_scales_linearly():
    # Warm caches before comparing tracemalloc peaks; the old per-D
    # frontier snapshots grew by ~4x here when input merely doubled.
    MoveDelta().insert("a" * 8).diff(MoveDelta().insert("b" * 8))

    def peak(units):
        base, target = MoveDelta().insert("a" * units), MoveDelta().insert("b" * units)
        gc.collect()
        tracemalloc.start()
        try:
            change = base.diff(target)
            _, result = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert base.compose(change) == target
        return result

    small, large = peak(200), peak(400)
    assert large < small * 2.75, (small, large)


# ── cursor hint ──


def test_cursor_anchors_an_ambiguous_insert():
    a, b = MoveDelta().insert("xaaay"), MoveDelta().insert("xaaaay")
    assert a.diff(b).ops == [{"retain": 4}, {"insert": "a"}]
    for cursor in (1, 2, 3, 4):
        d = a.diff(b, cursor=cursor)
        assert d.ops == [{"retain": cursor}, {"insert": "a"}]
        assert a.compose(d) == b


def test_cursor_anchors_an_ambiguous_delete():
    a, b = MoveDelta().insert("xaaaay"), MoveDelta().insert("xaaay")
    for cursor in (1, 2, 3, 4):
        d = a.diff(b, cursor=cursor)
        assert d.ops == [{"retain": cursor}, {"delete": 1}]
        assert a.compose(d) == b


def test_cursor_blocked_by_a_format_boundary():
    a = MoveDelta().insert("xa").insert("a", bold=True).insert("ay")
    b = MoveDelta().insert("xa").insert("a", bold=True).insert("aay")
    d = a.diff(b, cursor=1)
    assert d.ops[0]["retain"] == 4  # canonical placement kept
    assert a.compose(d) == b


def test_cursor_anchors_within_an_emoji_run():
    a = MoveDelta().insert("\N{GRINNING FACE}" * 2)
    b = MoveDelta().insert("\N{GRINNING FACE}" * 3)
    d = a.diff(b, cursor=2)  # a valid pair boundary
    assert d.ops == [{"retain": 2}, {"insert": "\N{GRINNING FACE}"}]
    # a cursor inside a surrogate pair cannot anchor: canonical stays
    mid = a.diff(b, cursor=1)
    assert a.compose(mid) == b
    assert mid.ops[0]["retain"] == 4


def test_fuzz_cursor_hint_preserves_the_law():
    rng = random.Random(74)
    for _ in range(300):
        x = "".join(rng.choice(LETTERS) for _ in range(rng.randint(0, 8)))
        y = "".join(rng.choice(LETTERS) for _ in range(rng.randint(0, 8)))
        a = MoveDelta().insert(x) if x else MoveDelta()
        b = MoveDelta().insert(y) if y else MoveDelta()
        cursor = rng.randint(0, len(a) + 1)
        d = a.diff(b, cursor=cursor)
        assert a.compose(d) == b, (x, y, cursor)
        assert d == a.diff(b, cursor=cursor)


def test_cursor_anchors_past_a_typed_embed_retain(fig_handler):
    # a typed embed patch before the edit consumes one unit, not a dict
    a = MoveDelta().insert(fig(1)).insert("xaaay")
    b = MoveDelta().insert(fig(2)).insert("xaaaay")
    for cursor in (2, 3, 4, 5):
        d = a.diff(b, cursor=cursor)
        assert a.compose(d) == b, cursor
        assert d.ops[1] == {"retain": cursor - 1}, (cursor, d.ops)
