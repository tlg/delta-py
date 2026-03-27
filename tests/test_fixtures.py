"""
Fixture-driven tests loaded from shared JSON files (same as TypeScript).
Covers: Op, AttributeMap (op module), Delta, OpIterator.
Skips: Block, Change, BoundaryClassifier, LabeledState, Project (not in Python port).
"""
import json
import math
import os
import pytest
from delta import Delta
from delta import op

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
