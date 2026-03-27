# Data-driven tests (concat, chop, length, changeLength, slice) are now in
# fixtures/delta-helpers.json. Run via test_fixtures.py
#
# The tests below require runtime callbacks and cannot be expressed as JSON fixtures.

try:
    import mock
except ImportError:
    from unittest import mock

from delta import Delta


def get_args(m, index):
    args, kwargs = m.call_args_list[index]
    return args


def test_each_line():
    fn = mock.Mock()
    delta = Delta().insert('Hello\n\n') \
                   .insert('World', bold=True) \
                   .insert({'image': 'octocat.png'}) \
                   .insert('\n', align='right') \
                   .insert('!')
    delta.each_line(fn)

    assert fn.call_count == 4
    assert get_args(fn, 0) == (Delta().insert('Hello'), {}, 0)
    assert get_args(fn, 1) == (Delta(), {}, 1)
    assert get_args(fn, 2) == (
        Delta().insert('World', bold=True).insert({'image': 'octocat.png'}),
        {'align': 'right'},
        2,
    )
    assert get_args(fn, 3) == (Delta().insert('!'), {}, 3)


def test_each_line_trailing_newline():
    fn = mock.Mock()
    delta = Delta().insert('Hello\nWorld!\n')
    delta.each_line(fn)

    assert fn.call_count == 2
    assert get_args(fn, 0) == (Delta().insert('Hello'), {}, 0)
    assert get_args(fn, 1) == (Delta().insert('World!'), {}, 1)


def test_each_line_non_document():
    fn = mock.Mock()
    delta = Delta().retain(1).delete(2)
    delta.each_line(fn)
    assert fn.call_count == 0


def test_each_line_early_return():
    state = {'count': 0}

    def counter(*args):
        if state['count'] == 1:
            return False
        state['count'] += 1

    delta = Delta().insert('Hello\nNew\nWorld!')
    fn = mock.Mock(side_effect=counter)

    delta.each_line(fn)
    assert fn.call_count == 2


def test_filter():
    delta = Delta().insert('Hello').insert({'image': True}).insert('World!')
    arr = [o for o in delta if isinstance(o.get('insert'), str)]
    assert len(arr) == 2


def test_map():
    delta = Delta().insert('Hello').insert({'image': True}).insert('World!')
    arr = [o['insert'] if isinstance(o.get('insert'), str) else '' for o in delta]
    assert arr == ['Hello', '', 'World!']


def test_partition():
    delta = Delta().insert('Hello').insert({'image': True}).insert('World!')
    passed = [o for o in delta if isinstance(o.get('insert'), str)]
    failed = [o for o in delta if not isinstance(o.get('insert'), str)]
    assert passed == [delta.ops[0], delta.ops[2]]
    assert failed == [delta.ops[1]]
