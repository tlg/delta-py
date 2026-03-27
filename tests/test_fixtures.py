"""
Fixture-driven tests loaded from shared JSON files (same as TypeScript).
Covers: Op, AttributeMap (op module), Delta, OpIterator, Block.
"""
import json
import math
import os
import pytest
from delta import Delta
from delta import op
from delta.block import (
    apply_move, apply_moves, BlockDelta, compose_moves, diff_to_moves,
    invert_move, project_blocks, resolve_move, transform_move,
)
from delta.boundary_classifier import classify_delta_boundaries
from delta.labeled_state import (
    assert_canonical_document, canonicalize_document, classify_gap_descendants,
    clone_labeled_state, flatten_document_units, get_unit_origin_gap,
    is_canonical_document, labeled_state_from_document, labeled_state_to_delta,
    replay_resolved_delta, resolve_delta, resolve_delta_against_state,
)
from delta.project import (
    block_boundaries, block_boundary_gap_anchors, project_block_spans,
    project_labeled_block_spans,
)
from delta.change import (
    apply_change, compose_change, ensure_resolved_block_moves, invert_change,
    replay_resolved_block_move, replay_resolved_block_moves,
    resolve_block_delta, transform_change,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def load_fixture(name):
    with open(os.path.join(FIXTURES_DIR, name)) as f:
        return json.load(f)


def delta_from_ops(ops):
    return Delta(ops)


def build_delta(steps):
    d = Delta()
    for step in steps:
        method = step[0]
        args = step[1:]
        # Convert null attributes arg to empty kwargs
        if method == "insert":
            if len(args) >= 2 and args[1] is None:
                getattr(d, method)(args[0])
            elif len(args) >= 2 and isinstance(args[1], dict):
                getattr(d, method)(args[0], **args[1])
            else:
                getattr(d, method)(args[0])
        elif method == "retain":
            if len(args) >= 2 and args[1] is None:
                getattr(d, method)(args[0])
            elif len(args) >= 2 and isinstance(args[1], dict):
                getattr(d, method)(args[0], **args[1])
            else:
                getattr(d, method)(args[0])
        elif method == "delete":
            getattr(d, method)(args[0])
    return d


def register_delta_embed():
    class DeltaHandler:
        @staticmethod
        def compose(a, b, *_args):
            return Delta(a).compose(Delta(b)).ops

        @staticmethod
        def transform(a, b, priority=False):
            return Delta(a).transform(Delta(b), priority).ops

        @staticmethod
        def invert(a, b):
            return Delta(a).invert(Delta(b)).ops

    Delta.register_embed("delta", DeltaHandler)


def unregister_delta_embed():
    Delta.unregister_embed("delta")


# ── Op.length ──


class TestOpLength:
    fixture = load_fixture("op-length.json")

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_length(self, test):
        assert op.length(test["op"]) == test["expected"]


# ── AttributeMap ──


class TestAttributeMapCompose:
    fixture = load_fixture("attributes-compose.json")

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_compose(self, test):
        result = op.compose(test["a"], test["b"])
        assert result == test["expected"]


class TestAttributeMapDiff:
    fixture = load_fixture("attributes-diff.json")

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_diff(self, test):
        result = op.diff(test["a"], test["b"])
        assert result == test["expected"]


class TestAttributeMapInvert:
    fixture = load_fixture("attributes-invert.json")

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_invert(self, test):
        result = op.invert(test["attributes"], test["base"])
        assert result == test["expected"]


class TestAttributeMapTransform:
    fixture = load_fixture("attributes-transform.json")

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_transform(self, test):
        result = op.transform(test["a"], test["b"], test["priority"])
        assert result == test["expected"]


# ── Delta builder ──


class TestDeltaBuilder:
    fixture = load_fixture("delta-builder.json")

    @pytest.mark.parametrize(
        "group", fixture["tests"], ids=[g["describe"] for g in fixture["tests"]]
    )
    def test_group(self, group):
        for test in group["tests"]:
            if test.get("ops") is not None:
                delta = Delta(test["ops"])
            elif test.get("build") is not None:
                delta = build_delta(test["build"])
            else:
                delta = Delta()

            if test.get("push"):
                for o in test["push"]:
                    delta.push(o)

            if "expected" in test:
                assert delta.ops == test["expected"], f"Failed: {test['name']}"
            if "expectedLength" in test:
                assert len(delta.ops) == test["expectedLength"], f"Failed: {test['name']}"


# ── Delta compose ──


class TestDeltaCompose:
    fixture = load_fixture("delta-compose.json")

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_compose(self, test):
        a = delta_from_ops(test["a"])
        b = delta_from_ops(test["b"])
        expected = delta_from_ops(test["expected"])
        assert a.compose(b) == expected


class TestDeltaComposeEmbed:
    fixture = load_fixture("delta-compose-embed.json")

    def setup_method(self):
        register_delta_embed()

    def teardown_method(self):
        unregister_delta_embed()

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_compose_embed(self, test):
        a = delta_from_ops(test["a"])
        b = delta_from_ops(test["b"])
        expected = delta_from_ops(test["expected"])
        assert a.compose(b) == expected

    @pytest.mark.parametrize(
        "test",
        fixture.get("errorTests", []),
        ids=[t["name"] for t in fixture.get("errorTests", [])],
    )
    def test_compose_embed_errors(self, test):
        a = delta_from_ops(test["a"])
        b = delta_from_ops(test["b"])
        with pytest.raises(Exception):
            a.compose(b)


class TestDeltaComposeEmbedNoHandler:
    fixture = load_fixture("delta-compose-embed.json")

    @pytest.mark.parametrize(
        "test",
        fixture.get("errorTestsNoHandler", []),
        ids=[t["name"] for t in fixture.get("errorTestsNoHandler", [])],
    )
    def test_compose_no_handler(self, test):
        a = delta_from_ops(test["a"])
        b = delta_from_ops(test["b"])
        with pytest.raises(Exception):
            a.compose(b)


# ── Delta diff ──


class TestDeltaDiff:
    fixture = load_fixture("delta-diff.json")

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_diff(self, test):
        a = delta_from_ops(test["a"])
        b = delta_from_ops(test["b"])
        expected = delta_from_ops(test["expected"])
        assert a.diff(b) == expected

    @pytest.mark.parametrize(
        "test",
        fixture.get("errorTests", []),
        ids=[t["name"] for t in fixture.get("errorTests", [])],
    )
    def test_diff_errors(self, test):
        a = delta_from_ops(test["a"])
        b = delta_from_ops(test["b"])
        with pytest.raises(Exception):
            a.diff(b)


# ── Delta transform ──


class TestDeltaTransform:
    fixture = load_fixture("delta-transform.json")

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_transform(self, test):
        a = delta_from_ops(test["a"])
        b = delta_from_ops(test["b"])
        expected = delta_from_ops(test["expected"])
        assert a.transform(b, test["priority"]) == expected


class TestDeltaTransformEmbed:
    fixture = load_fixture("delta-transform-embed.json")

    def setup_method(self):
        register_delta_embed()

    def teardown_method(self):
        unregister_delta_embed()

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_transform_embed(self, test):
        a = delta_from_ops(test["a"])
        b = delta_from_ops(test["b"])
        expected = delta_from_ops(test["expected"])
        assert a.transform(b, test["priority"]) == expected


# ── Delta transformPosition ──


class TestDeltaTransformPosition:
    fixture = load_fixture("delta-transform-position.json")

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_transform_position(self, test):
        delta = delta_from_ops(test["delta"])
        priority = test.get("priority", False)
        assert delta.transform(test["index"], priority) == test["expected"]


# ── Delta invert ──


class TestDeltaInvert:
    fixture = load_fixture("delta-invert.json")

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_invert(self, test):
        delta = delta_from_ops(test["delta"])
        base = delta_from_ops(test["base"])
        expected = delta_from_ops(test["expected"])
        inverted = delta.invert(base)
        assert inverted == expected
        if test.get("verifyRoundTrip"):
            assert base.compose(delta).compose(inverted) == base


class TestDeltaInvertEmbed:
    fixture = load_fixture("delta-invert-embed.json")

    def setup_method(self):
        register_delta_embed()

    def teardown_method(self):
        unregister_delta_embed()

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_invert_embed(self, test):
        delta = delta_from_ops(test["delta"])
        base = delta_from_ops(test["base"])
        expected = delta_from_ops(test["expected"])
        inverted = delta.invert(base)
        assert inverted == expected
        if test.get("verifyRoundTrip"):
            assert base.compose(delta).compose(inverted) == base

    @pytest.mark.parametrize(
        "test",
        fixture.get("errorTests", []),
        ids=[t["name"] for t in fixture.get("errorTests", [])],
    )
    def test_invert_embed_errors(self, test):
        delta = delta_from_ops(test["delta"])
        base = delta_from_ops(test["base"])
        with pytest.raises(Exception):
            delta.invert(base)


# ── Delta helpers ──


class TestDeltaHelpers:
    fixture = load_fixture("delta-helpers.json")

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["concat"],
        ids=[t["name"] for t in fixture["tests"]["concat"]],
    )
    def test_concat(self, test):
        a = delta_from_ops(test["a"])
        b = delta_from_ops(test["b"])
        expected = delta_from_ops(test["expected"])
        assert a.concat(b) == expected

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["chop"],
        ids=[t["name"] for t in fixture["tests"]["chop"]],
    )
    def test_chop(self, test):
        delta = delta_from_ops(test["input"])
        expected = delta_from_ops(test["expected"])
        assert delta.chop() == expected

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["length"],
        ids=[t["name"] for t in fixture["tests"]["length"]],
    )
    def test_length(self, test):
        delta = delta_from_ops(test["input"])
        assert delta.length() == test["expected"]

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["changeLength"],
        ids=[t["name"] for t in fixture["tests"]["changeLength"]],
    )
    def test_change_length(self, test):
        delta = delta_from_ops(test["input"])
        assert delta.change_length() == test["expected"]

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["slice"],
        ids=[t["name"] for t in fixture["tests"]["slice"]],
    )
    def test_slice(self, test):
        delta = delta_from_ops(test["input"])
        expected = delta_from_ops(test["expected"])
        start = test.get("start")
        end = test.get("end")
        if start is not None:
            result = delta[start:end]
        else:
            result = delta[:]
        assert result == expected


# ── OpIterator ──


class TestOpIterator:
    fixture = load_fixture("op-iterator.json")
    shared_ops = fixture["ops"]

    def _resolve_infinity(self, val):
        if val == "Infinity":
            return math.inf
        if isinstance(val, dict):
            return {k: self._resolve_infinity(v) for k, v in val.items()}
        if isinstance(val, list):
            return [self._resolve_infinity(v) for v in val]
        return val

    def _run_steps(self, ops_spec, steps):
        ops = self.shared_ops if ops_spec == "shared" else ops_spec
        it = op.Iterator(ops)
        for step in steps:
            action = step["action"]
            if action == "peekLength":
                assert it.peek_length() == self._resolve_infinity(step["expected"])
            elif action == "peekType":
                assert it.peek_type() == step["expected"]
            elif action == "next":
                length = step.get("length")
                result = it.next(length)
                if "expected" in step:
                    assert result == self._resolve_infinity(step["expected"])
            elif action == "rest":
                assert it.rest() == step["expected"]
            elif action == "hasNext":
                pass  # handled at test level

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["hasNext"],
        ids=[t["name"] for t in fixture["tests"]["hasNext"]],
    )
    def test_has_next(self, test):
        ops = self.shared_ops if test["ops"] == "shared" else test["ops"]
        it = op.Iterator(ops)
        assert it.has_next() == test["expected"]

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["peekLength"],
        ids=[t["name"] for t in fixture["tests"]["peekLength"]],
    )
    def test_peek_length(self, test):
        self._run_steps(test["ops"], test["steps"])

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["peekType"],
        ids=[t["name"] for t in fixture["tests"]["peekType"]],
    )
    def test_peek_type(self, test):
        self._run_steps(test["ops"], test["steps"])

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["next"],
        ids=[t["name"] for t in fixture["tests"]["next"]],
    )
    def test_next(self, test):
        self._run_steps(test["ops"], test["steps"])

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["rest"],
        ids=[t["name"] for t in fixture["tests"]["rest"]],
    )
    def test_rest(self, test):
        self._run_steps(test["ops"], test["steps"])


# ── Block: projectBlocks ──


def _move_from_spec(m):
    return resolve_move(m["index"], m["count"], m["before"])


def _build_block_delta(steps):
    d = BlockDelta()
    for step in steps:
        method = step[0]
        args = step[1:]
        getattr(d, method)(*args)
    return d


class TestBlockProject:
    fixture = load_fixture("block-project.json")

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_project_blocks(self, test):
        doc = delta_from_ops(test["document"])
        blocks = project_blocks(doc)
        expected = [
            {"delta": delta_from_ops(b["delta"]), "attributes": b["attributes"]}
            for b in test["expected"]
        ]
        assert blocks == expected


# ── Block: applyMove ──


class TestBlockApplyMove:
    fixture = load_fixture("block-apply-move.json")

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_apply_move(self, test):
        move = _move_from_spec(test["move"])
        assert apply_move(self.fixture["base"], move) == test["expected"]


# ── Block: invertMove ──


class TestBlockInvertMove:
    fixture = load_fixture("block-invert-move.json")

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_invert_move(self, test):
        move = _move_from_spec(test["move"])
        inverted = invert_move(move, len(test["base"]))
        expected_inverse = [_move_from_spec(m) for m in test["expectedInverse"]]
        assert inverted == expected_inverse
        if test.get("verifyRoundTrip"):
            assert apply_moves(apply_move(test["base"], move), inverted) == test["base"]


# ── Block: composeMoves ──


class TestBlockComposeMoves:
    fixture = load_fixture("block-compose-moves.json")

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_compose_moves(self, test):
        a = _move_from_spec(test["a"])
        b = _move_from_spec(test["b"])
        composed = compose_moves(a, b, len(test["base"]))
        expected = [_move_from_spec(m) for m in test["expected"]]
        assert composed == expected
        if test.get("verifySequential"):
            assert apply_moves(test["base"], composed) == apply_move(apply_move(test["base"], a), b)


# ── Block: transformMove ──


class TestBlockTransformMove:
    fixture = load_fixture("block-transform-move.json")

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_transform_move(self, test):
        a = _move_from_spec(test["a"])
        b = _move_from_spec(test["b"])
        b_prime = transform_move(a, b, test["priority"], len(test["base"]))
        expected = [_move_from_spec(m) for m in test["expectedBPrime"]]
        assert b_prime == expected
        if test.get("expectedResult"):
            assert apply_moves(apply_move(test["base"], a), b_prime) == test["expectedResult"]
        if test.get("verifyConvergence"):
            a_prime = transform_move(b, a, not test["priority"], len(test["base"]))
            assert apply_moves(apply_move(test["base"], a), b_prime) == \
                apply_moves(apply_move(test["base"], b), a_prime)


# ── BlockDelta ──


class TestBlockDelta:
    fixture = load_fixture("block-delta.json")

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_block_delta(self, test):
        # diffToMoves tests
        if test.get("diffBase") is not None and test.get("diffTarget") is not None:
            moves = diff_to_moves(test["diffBase"], test["diffTarget"])
            if test.get("expectedMoves"):
                expected = [_move_from_spec(m) for m in test["expectedMoves"]]
                assert moves == expected
            if test.get("apply") and test.get("expectedApply"):
                delta = BlockDelta.from_moves(moves)
                assert delta.apply(test["apply"]) == test["expectedApply"]
            return

        # compose tests
        if test.get("a") and test.get("b") and test.get("composeBlockCount") is not None:
            a = _build_block_delta(test["a"]["build"])
            b = _build_block_delta(test["b"]["build"])
            composed = a.compose(b, test["composeBlockCount"])
            if test.get("expectedOps") is not None:
                assert composed.ops == test["expectedOps"]
            if test.get("apply") and test.get("expectedApply"):
                assert composed.apply(test["apply"]) == test["expectedApply"]
            return

        # transform tests
        if test.get("a") and test.get("b") and test.get("transformBlockCount") is not None:
            a = _build_block_delta(test["a"]["build"])
            b = _build_block_delta(test["b"]["build"])
            b_prime = a.transform(b, test["transformBlockCount"], test["priority"])
            if test.get("expectedOps") is not None:
                assert b_prime.ops == test["expectedOps"]
            if test.get("apply") and test.get("expectedApply"):
                assert b_prime.apply(a.apply(test["apply"])) == test["expectedApply"]
            if test.get("verifyConvergence"):
                a_prime = b.transform(a, test["transformBlockCount"], not test["priority"])
                assert b_prime.apply(a.apply(test["apply"])) == \
                    a_prime.apply(b.apply(test["apply"]))
            return

        # fromMoves tests
        if test.get("fromMoves"):
            moves = [_move_from_spec(m) for m in test["fromMoves"]]
            delta = BlockDelta.from_moves(moves)
            if test.get("expectedOps") is not None:
                assert delta.ops == test["expectedOps"]
            if test.get("resolveBlockCount") is not None and test.get("expectedResolve"):
                expected_resolve = [_move_from_spec(m) for m in test["expectedResolve"]]
                assert delta.resolve(test["resolveBlockCount"]) == expected_resolve
            if test.get("apply") and test.get("expectedApply"):
                assert delta.apply(test["apply"]) == test["expectedApply"]
            return

        # build tests
        if test.get("build"):
            delta = _build_block_delta(test["build"])
            if test.get("chop"):
                delta = delta.chop()
            if test.get("expectedOps") is not None:
                assert delta.ops == test["expectedOps"]
            if test.get("apply") and test.get("expectedApply"):
                assert delta.apply(test["apply"]) == test["expectedApply"]
            # invert tests
            if test.get("invertBlockCount") is not None:
                inverted = delta.invert(test["invertBlockCount"])
                if test.get("expectedInvertOps"):
                    assert inverted.ops == test["expectedInvertOps"]
                if test.get("verifyInvertRoundTrip") and test.get("apply"):
                    assert inverted.apply(delta.apply(test["apply"])) == test["apply"]


# ── BoundaryClassifier ──


class TestBoundaryClassifier:
    fixture = load_fixture("boundary-classifier.json")

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_classify(self, test):
        base = delta_from_ops(test["base"])
        delta = delta_from_ops(test["delta"])
        assert classify_delta_boundaries(base, delta) == test["expected"]

    @pytest.mark.parametrize(
        "test",
        fixture.get("errorTests", []),
        ids=[t["name"] for t in fixture.get("errorTests", [])],
    )
    def test_classify_errors(self, test):
        base = delta_from_ops(test["base"])
        delta = delta_from_ops(test["delta"])
        with pytest.raises(Exception):
            classify_delta_boundaries(base, delta)


# ── Project ──


class TestProject:
    fixture = load_fixture("project.json")

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["projectBlockSpans"],
        ids=[t["name"] for t in fixture["tests"]["projectBlockSpans"]],
    )
    def test_project_block_spans(self, test):
        doc = delta_from_ops(test["document"])
        assert project_block_spans(doc) == test["expected"]

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["blockBoundaries"],
        ids=[t["name"] for t in fixture["tests"]["blockBoundaries"]],
    )
    def test_block_boundaries(self, test):
        doc = delta_from_ops(test["document"])
        assert block_boundaries(doc) == test["expected"]

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["projectLabeledBlockSpans"],
        ids=[t["name"] for t in fixture["tests"]["projectLabeledBlockSpans"]],
    )
    def test_project_labeled_block_spans(self, test):
        doc = delta_from_ops(test["document"])
        state = labeled_state_from_document(doc)
        assert project_labeled_block_spans(state) == test["expected"]

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["blockBoundaryGapAnchors"],
        ids=[t["name"] for t in fixture["tests"]["blockBoundaryGapAnchors"]],
    )
    def test_block_boundary_gap_anchors(self, test):
        doc = delta_from_ops(test["document"])
        state = labeled_state_from_document(doc)
        assert block_boundary_gap_anchors(state) == test["expected"]


# ── LabeledState ──


class TestLabeledState:
    fixture = load_fixture("labeled-state.json")

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["canonicalizeDocument"],
        ids=[t["name"] for t in fixture["tests"]["canonicalizeDocument"]],
    )
    def test_canonicalize(self, test):
        doc = delta_from_ops(test["input"])
        expected = delta_from_ops(test["expected"])
        canonical = canonicalize_document(doc)
        assert canonical == expected
        assert is_canonical_document(canonical)

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["canonicalizeDocumentErrors"],
        ids=[t["name"] for t in fixture["tests"]["canonicalizeDocumentErrors"]],
    )
    def test_canonicalize_errors(self, test):
        doc = delta_from_ops(test["input"])
        with pytest.raises(Exception):
            canonicalize_document(doc)
            assert_canonical_document(doc)

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["flattenDocumentUnits"],
        ids=[t["name"] for t in fixture["tests"]["flattenDocumentUnits"]],
    )
    def test_flatten_document_units(self, test):
        doc = delta_from_ops(test["input"])
        assert flatten_document_units(doc) == test["expected"]

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["labeledStateFromDocument"],
        ids=[t["name"] for t in fixture["tests"]["labeledStateFromDocument"]],
    )
    def test_labeled_state_from_document(self, test):
        doc = delta_from_ops(test["input"])
        state = labeled_state_from_document(doc)
        if test.get("expectedUnits"):
            assert state["units"] == test["expectedUnits"]
        if test.get("expectedGaps"):
            assert state["gaps"] == test["expectedGaps"]
        if test.get("expectedRoundTrip"):
            assert labeled_state_to_delta(state) == delta_from_ops(test["expectedRoundTrip"])

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["resolveDelta"],
        ids=[t["name"] for t in fixture["tests"]["resolveDelta"]],
    )
    def test_resolve_delta(self, test):
        base = delta_from_ops(test["base"])
        d = delta_from_ops(test["delta"])
        base_state = labeled_state_from_document(base)
        resolved = resolve_delta_against_state(base_state, d)
        # Strip hidden metadata for comparison
        cleaned = {
            'insertsByGap': [
                {'gap': ins['gap'], 'units': [{k: v for k, v in u.items() if not k.startswith('_')} for u in ins['units']]}
                for ins in resolved['insertsByGap']
            ],
            'deletedUnitIds': resolved['deletedUnitIds'],
            'formatPatchesByUnitId': resolved['formatPatchesByUnitId'],
        }
        assert cleaned == test["expectedResolved"]
        if test.get("expectedResult"):
            assert labeled_state_to_delta(replay_resolved_delta(base_state, resolved)) == delta_from_ops(test["expectedResult"])

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["resolveDeltaErrors"],
        ids=[t["name"] for t in fixture["tests"]["resolveDeltaErrors"]],
    )
    def test_resolve_delta_errors(self, test):
        base = delta_from_ops(test["base"])
        d = delta_from_ops(test["delta"])
        with pytest.raises(Exception):
            resolve_delta(base, d)

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["replayResolvedDelta"],
        ids=[t["name"] for t in fixture["tests"]["replayResolvedDelta"]],
    )
    def test_replay_resolved_delta(self, test):
        base = delta_from_ops(test["base"])
        base_state = labeled_state_from_document(base)
        left = resolve_delta_against_state(base_state, delta_from_ops(test["left"]))
        right = resolve_delta_against_state(base_state, delta_from_ops(test["right"]))
        descendant = replay_resolved_delta(base_state, left)
        assert labeled_state_to_delta(replay_resolved_delta(descendant, right)) == delta_from_ops(test["expected"])

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["classifyGapDescendants"],
        ids=[t["name"] for t in fixture["tests"]["classifyGapDescendants"]],
    )
    def test_classify_gap_descendants(self, test):
        base = delta_from_ops(test["base"])
        base_state = labeled_state_from_document(base)
        d = delta_from_ops(test["delta"])
        descendant = replay_resolved_delta(base_state, resolve_delta_against_state(base_state, d))
        assert classify_gap_descendants(descendant, base_state["gaps"][test["gapIndex"]]) == test["expected"]


# ── Change helpers ──


def _change_from_spec(spec):
    bd = spec.get("blockDelta", [])
    if spec.get("blockDeltaFromMoves"):
        bd_obj = BlockDelta.from_moves([resolve_move(m["index"], m["count"], m["before"]) for m in spec["blockDeltaFromMoves"]])
    elif isinstance(bd, list):
        bd_obj = BlockDelta(bd) if bd else BlockDelta()
    else:
        bd_obj = BlockDelta(bd)
    return {"delta": delta_from_ops(spec.get("delta", [])), "blockDelta": bd_obj}


# ── Change: apply ──


class TestChangeApply:
    fixture = load_fixture("change-apply.json")

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_apply(self, test):
        base = delta_from_ops(test["base"])
        change = _change_from_spec(test["change"])
        expected = delta_from_ops(test["expected"])
        assert apply_change(base, change) == expected


# ── Change: compose ──


class TestChangeCompose:
    fixture = load_fixture("change-compose.json")

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_compose(self, test):
        base = delta_from_ops(test["base"])
        first = _change_from_spec(test["first"])
        second = _change_from_spec(test["second"])
        composed = compose_change(base, first, second)
        expected = delta_from_ops(test["expectedResult"])
        assert apply_change(base, composed) == expected
        if test.get("expectedHasBlockDelta"):
            assert len(composed["blockDelta"].ops) > 0
        if test.get("expectedEmptyBlockDelta"):
            assert composed["blockDelta"] == BlockDelta()


# ── Change: transform ──


class TestChangeTransform:
    fixture = load_fixture("change-transform.json")

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_transform(self, test):
        base = delta_from_ops(test["base"])
        left = _change_from_spec(test["left"])
        right = _change_from_spec(test["right"])

        if test.get("verifyConvergenceBothPriorities"):
            for priority in [False, True]:
                right_prime = transform_change(left, right, base, priority)
                left_prime = transform_change(right, left, base, not priority)
                assert apply_change(apply_change(base, left), right_prime) == \
                    apply_change(apply_change(base, right), left_prime)
            return

        priority = test.get("priority", True)
        transformed_right = transform_change(left, right, base, priority)

        if test.get("expectedTransformed"):
            assert transformed_right == _change_from_spec(test["expectedTransformed"])
        if test.get("expectedTransformedDelta"):
            assert transformed_right["delta"] == delta_from_ops(test["expectedTransformedDelta"])
        if test.get("expectedEmptyBlockDelta"):
            assert transformed_right["blockDelta"] == BlockDelta()
        if test.get("expectedHasBlockDelta"):
            assert len(transformed_right["blockDelta"].ops) > 0
        if test.get("expectedResult"):
            assert apply_change(apply_change(base, left), transformed_right) == delta_from_ops(test["expectedResult"])
        if test.get("verifyConvergence"):
            left_prime = transform_change(right, left, base, not priority)
            assert apply_change(apply_change(base, left), transformed_right) == \
                apply_change(apply_change(base, right), left_prime)


# ── Change: invert ──


class TestChangeInvert:
    fixture = load_fixture("change-invert.json")

    @pytest.mark.parametrize(
        "test", fixture["tests"], ids=[t["name"] for t in fixture["tests"]]
    )
    def test_invert(self, test):
        base = delta_from_ops(test["base"])
        change = _change_from_spec(test["change"])
        inverse = invert_change(base, change)
        if test.get("expectedHasDelta"):
            assert len(inverse["delta"].ops) > 0
        if test.get("expectedHasBlockDelta"):
            assert len(inverse["blockDelta"].ops) > 0
        if test.get("verifyRoundTrip"):
            assert apply_change(apply_change(base, change), inverse) == base


# ── Change: bridge ──


class TestChangeBridge:
    fixture = load_fixture("change-bridge.json")

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["resolveBlockDelta"],
        ids=[t["name"] for t in fixture["tests"]["resolveBlockDelta"]],
    )
    def test_resolve_block_delta(self, test):
        doc = delta_from_ops(test["document"])
        state = labeled_state_from_document(doc)
        if test.get("blockDeltaFromMoves"):
            bd = BlockDelta.from_moves([resolve_move(m["index"], m["count"], m["before"]) for m in test["blockDeltaFromMoves"]])
        else:
            bd = BlockDelta(test["blockDelta"])
        resolved = resolve_block_delta(state, bd)
        if test.get("expected"):
            # Strip hidden metadata keys for comparison
            cleaned = []
            for m in resolved:
                cleaned.append({k: v for k, v in m.items() if not k.startswith('_')})
            assert cleaned == test["expected"]
        if test.get("expectedLength") is not None:
            assert len(resolved) == test["expectedLength"]
        if test.get("expectedResult"):
            assert labeled_state_to_delta(replay_resolved_block_moves(state, resolved)["state"]) == delta_from_ops(test["expectedResult"])

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["replayResolvedBlockMove"],
        ids=[t["name"] for t in fixture["tests"]["replayResolvedBlockMove"]],
    )
    def test_replay_resolved_block_move(self, test):
        doc = delta_from_ops(test["document"])
        state = labeled_state_from_document(doc)
        bd = BlockDelta(test["blockDelta"])
        moves = resolve_block_delta(state, bd)
        move = moves[0]

        target_state = state
        if test.get("preDelta"):
            target_state = replay_resolved_delta(state, resolve_delta_against_state(state, delta_from_ops(test["preDelta"])))

        result = replay_resolved_block_move(target_state, move)
        if test.get("expectedResult"):
            assert labeled_state_to_delta(result["state"]) == delta_from_ops(test["expectedResult"])
        if test.get("expectedRestorations"):
            assert [r["restored"] for r in result["restorations"]] == test["expectedRestorations"]

    @pytest.mark.parametrize(
        "test",
        fixture["tests"]["ensureResolvedBlockMoves"],
        ids=[t["name"] for t in fixture["tests"]["ensureResolvedBlockMoves"]],
    )
    def test_ensure_resolved_block_moves(self, test):
        doc = delta_from_ops(test["document"])
        state = labeled_state_from_document(doc)
        bd = BlockDelta(test["blockDelta"])
        moves = resolve_block_delta(state, bd)
        move = moves[0]

        merged = replay_resolved_delta(state, resolve_delta_against_state(state, delta_from_ops(test["preDelta"])))
        first = ensure_resolved_block_moves(merged, [move])
        second = ensure_resolved_block_moves(first["state"], [move])

        if test.get("expectedAfterFirst"):
            assert labeled_state_to_delta(first["state"]) == delta_from_ops(test["expectedAfterFirst"])
        assert labeled_state_to_delta(second["state"]) == labeled_state_to_delta(first["state"])
        if test.get("firstRestorations"):
            assert [r["restored"] for r in first["restorations"]] == test["firstRestorations"]
        if test.get("secondRestorations"):
            assert [r["restored"] for r in second["restorations"]] == test["secondRestorations"]
