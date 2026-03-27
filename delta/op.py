import copy
import math


def compose(a, b, keep_null=False):
    """
    Compose two attribute sets into one.

    ``keep_null`` controls whether None values are retained in the result.
    """
    if a is None:
        a = {}
    if b is None:
        b = {}

    attributes = {
        k: copy.deepcopy(v)
        for k, v in b.items()
        if keep_null or v is not None
    }

    for k, v in a.items():
        if k not in b:
            attributes[k] = copy.deepcopy(v)

    return attributes or None


def diff(a, b):
    """Return the attribute difference from a to b."""
    if a is None:
        a = {}
    if b is None:
        b = {}

    attributes = {}
    for k in set(a) | set(b):
        if a.get(k) != b.get(k):
            attributes[k] = b.get(k)

    return attributes or None


def invert(attr, base):
    attr = attr or {}
    base = base or {}

    result = {}

    for k in base:
        if base[k] != attr.get(k) and k in attr:
            result[k] = base[k]

    for k in attr:
        if attr[k] != base.get(k) and k not in base:
            result[k] = None

    return result


def transform(a, b, priority=True):
    """
    Transform attributes b against a.

    If ``priority`` is false, just return b unchanged.
    """
    if a is None:
        a = {}
    if b is None:
        b = {}

    if not priority:
        return b or None

    attributes = {k: v for k, v in b.items() if k not in a}
    return attributes or None


def length_of(op):
    if isinstance(op.get('delete'), int):
        return op['delete']
    if isinstance(op.get('retain'), (int, float)):
        return op['retain']
    if isinstance(op.get('retain'), dict) and op.get('retain'):
        return 1
    if isinstance(op.get('insert'), str):
        return len(op['insert'])
    return 1


def type_of(op):
    if not op:
        return None
    if isinstance(op.get('delete'), int):
        return 'delete'
    if isinstance(op.get('retain'), int) or (isinstance(op.get('retain'), dict) and op.get('retain')):
        return 'retain'
    return 'insert'


class Iterator:
    """
    A stateful iterator over ops that can split operations to exactly
    the length needed via the ``next()`` method.
    """

    def __init__(self, ops=None):
        self.ops = ops if ops is not None else []
        self.index = 0
        self.offset = 0

    def reset(self):
        self.index = 0
        self.offset = 0

    def has_next(self):
        return self.peek_length() < math.inf

    def next(self, length=None):
        if length is None:
            length = math.inf

        op = self.peek()
        if op is None:
            return {'retain': math.inf}

        op_type = type_of(op)
        offset = self.offset
        op_length = length_of(op)
        if length >= op_length - offset:
            length = op_length - offset
            self.index += 1
            self.offset = 0
        else:
            self.offset += length

        if op_type == 'delete':
            return {'delete': length}

        result_op = {}
        if op.get('attributes'):
            result_op['attributes'] = op['attributes']

        if isinstance(op.get('retain'), (int, float)):
            result_op['retain'] = length
        elif isinstance(op.get('retain'), dict) and op.get('retain'):
            result_op['retain'] = op['retain']
        elif isinstance(op.get('insert'), str):
            result_op['insert'] = op['insert'][offset:offset + length]
        else:
            assert offset == 0
            assert length == 1
            if 'insert' in op:
                result_op['insert'] = op['insert']

        return result_op

    __next__ = next

    def __iter__(self):
        return self

    def peek(self):
        if self.index < len(self.ops):
            return self.ops[self.index]
        return None

    def peek_length(self):
        op = self.peek()
        if op is None:
            return math.inf
        return length_of(op) - self.offset

    def peek_type(self):
        op = self.peek()
        if op:
            return type_of(op)
        return 'retain'

    def rest(self):
        if not self.has_next():
            return []
        if self.offset == 0:
            return self.ops[self.index:]
        offset = self.offset
        index = self.index
        result = self.next()
        remaining = self.ops[self.index:]
        self.offset = offset
        self.index = index
        return [result] + remaining


length = length_of
type = type_of


def iterator(x):
    return Iterator(x)
