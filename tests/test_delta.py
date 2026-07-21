# Data-driven builder tests are now in fixtures/delta-builder.json
# Run via test_fixtures.py
#
# The test below checks basic construction which is Python-specific.

import pytest

from delta import Delta
from delta.base import Delta as BaseDelta
from delta.base import _PendingRef
from delta.base import get_embed_type_and_data


def test_creation():
    d = Delta()
    assert d.ops == []
    d = Delta([])
    assert d.ops == []
    d2 = Delta(d)
    assert d2.ops == []


def test_numeric_anonymous_embed_builder():
    assert Delta().insert(2).ops == [{"insert": 2}]


def test_constructor_owns_the_source_list_and_nested_values():
    source = [{"insert": {"image": {"src": "before"}}}, {"insert": "a"}]
    delta = Delta(source)
    source[0]["insert"]["image"]["src"] = "after"
    source.append({"insert": "outside"})
    delta.insert("b")
    assert delta.ops == [{"insert": {"image": {"src": "before"}}}, {"insert": "ab"}]
    assert source[1] == {"insert": "a"}


def test_extend_owns_every_source_operation():
    source = [{"insert": "a"}, {"insert": {"image": {"src": "before"}}}]
    delta = Delta().extend(source)
    source[0]["insert"] = "changed"
    source[1]["insert"]["image"]["src"] = "after"
    assert delta.ops == [{"insert": "a"}, {"insert": {"image": {"src": "before"}}}]


def test_constructor_owns_a_source_delta():
    source = Delta().insert({"image": {"src": "before"}})
    clone = Delta(source)
    source.ops[0]["insert"]["image"]["src"] = "after"
    assert clone.ops == [{"insert": {"image": {"src": "before"}}}]


def test_internal_owned_constructor_transfers_without_copying():
    owned = [{"insert": {"image": {"src": "owned"}}}]
    assert Delta._from_owned_ops(owned).ops is owned


def test_equality_defers_unrelated_objects():
    assert Delta().__eq__(object()) is NotImplemented


@pytest.mark.parametrize(
    ("left", "right"),
    (
        ({}, {"cell": {}}),
        ({"cell": {}}, {}),
        ({"cell": {}, "image": {}}, {"cell": {}}),
        ({"cell": {}}, {"cell": {}, "image": {}}),
    ),
)
def test_embed_values_have_exactly_one_type(left, right):
    with pytest.raises(ValueError, match="exactly one type"):
        get_embed_type_and_data(left, right)


def test_base_operations_preserve_subclasses():
    class SubDelta(BaseDelta):
        pass

    assert isinstance(SubDelta().insert("abc")[:1], SubDelta)
    assert isinstance(SubDelta().insert("x").invert(SubDelta()), SubDelta)
    move = SubDelta().cut("r", 1).retain(1).paste("r", 0, 1)
    assert isinstance(move.transform(SubDelta().retain(1, bold=True), True), SubDelta)


def test_base_two_pass_operations_fail_if_still_unresolved():
    class RetryingDelta(BaseDelta):
        def _compose_with(self, other, shared, nested):
            shared["retry"] = True
            return self.__class__().paste("r", 0, 1)

        def _transform_with(self, other, priority, shared, root):
            shared["retry"] = True
            return self.__class__().paste(_PendingRef("r"), 0, 1)

    with pytest.raises(RuntimeError, match="compose remained unresolved"):
        RetryingDelta().compose(RetryingDelta().cut("r", 1).paste("r", 0, 1))
    with pytest.raises(RuntimeError, match="transform remained unresolved"):
        RetryingDelta().transform(RetryingDelta())


def test_base_invert_reads_the_base_with_one_forward_iterator():
    class UnsliceableDelta(BaseDelta):
        def __getitem__(self, item):
            raise AssertionError("invert must not re-slice its base")

    base = UnsliceableDelta().insert("ABCDEF")
    change = UnsliceableDelta(
        [
            {"cut": {"ref": "r", "length": 2}},
            {"retain": 1, "attributes": {}},
            {"delete": 1},
            {"retain": 2},
            {"paste": {"ref": "r", "start": 0, "length": 2}},
        ]
    )
    inverse = change.invert(base)
    plain_base = BaseDelta(base.ops)
    assert plain_base.compose(BaseDelta(change.ops)).compose(BaseDelta(inverse.ops)) == plain_base
