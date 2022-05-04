from cgitb import handler
import copy
import diff_match_patch

try:
    from functools import reduce
except:
    pass

from . import op


NULL_CHARACTER = chr(0)
DIFF_EQUAL = 0
DIFF_INSERT = 1
DIFF_DELETE = -1


def merge(a, b):
    return copy.deepcopy(a or {}).update(b or {})


def differ(a, b, timeout=1):
    differ = diff_match_patch.diff_match_patch()
    differ.Diff_Timeout = timeout
    return differ.diff_main(a, b)


def smallest(*parts):
    return min(filter(lambda x: x is not None, parts))


def get_embed_type_and_data(a, b):
    if not isinstance(a, dict) or a is None:
        raise Exception('cannot retain a ', type(a))
    if not isinstance(b, dict) or b is None:
        raise Exception('cannot retain b ', type(b))

    embed_type = next(iter(a))
    if not embed_type or embed_type != next(iter(b)):
        raise Exception('embed types not matched: ',
                        embed_type, ' != ', next(iter(b)))
    return [embed_type, a[embed_type], b[embed_type]]


handlers = {}


class Delta(object):

    @staticmethod
    def register_embed(embed_type, handler):
        handlers[embed_type] = handler

    @staticmethod
    def unregister_embed(embed_type):
        if embed_type in handlers:
            handlers.pop(embed_type)

    @staticmethod
    def get_handler(embed_type):
        handler = handlers.get(embed_type)
        if handler is None:
            raise Exception("no handlers for embed type ", embed_type)
        return handler

    def __init__(self, ops=None, **attrs):
        if hasattr(ops, 'ops'):
            ops = ops.ops
        self.ops = ops or []
        self.__dict__.update(attrs)

    def __eq__(self, other):
        return self.ops == other.ops

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.ops)

    def insert(self, text, **attrs):
        if text == "":
            return self
        new_op = {'insert': text}
        if attrs:
            new_op['attributes'] = attrs
        return self.push(new_op)

    def delete(self, length):
        if length <= 0:
            return self
        return self.push({'delete': length})

    def retain(self, length, **attrs):
        if isinstance(length, (int, float)) and length <= 0:
            return self
        new_op = {'retain': length}
        if attrs:
            new_op['attributes'] = attrs
        return self.push(new_op)

    def push(self, operation):
        index = len(self.ops)
        new_op = copy.deepcopy(operation)
        try:
            last_op = self.ops[index - 1]
        except IndexError:
            self.ops.append(new_op)
            return self

        if op.type(new_op) == op.type(last_op) == 'delete':
            last_op['delete'] += new_op['delete']
            return self

        if op.type(last_op) == 'delete' and op.type(new_op) == 'insert':
            index -= 1
            try:
                last_op = self.ops[index - 1]
            except IndexError:
                self.ops.insert(0, new_op)
                return self

        if new_op.get('attributes') == last_op.get('attributes'):
            if isinstance(new_op.get('insert'), str) and isinstance(last_op.get('insert'), str):
                last_op['insert'] += new_op['insert']
                return self

            if isinstance(new_op.get('retain'), (int, float)) and isinstance(last_op.get('retain'), (int, float)):
                last_op['retain'] += new_op['retain']
                if isinstance(new_op.get('attributes'), dict):
                    last_op['attributes'] = new_op.get('attributes')
                return self

        self.ops.insert(index, new_op)
        return self

    def extend(self, ops):
        if hasattr(ops, 'ops'):
            ops = ops.ops
        if not ops:
            return self
        self.push(ops[0])
        self.ops.extend(ops[1:])
        return self

    def concat(self, other):
        delta = self.__class__(copy.deepcopy(self.ops))
        delta.extend(other)
        return delta

    def chop(self):
        try:
            last_op = self.ops[-1]
            if isinstance(last_op.get('retain'), (int, float)) and not last_op.get('attributes'):
                self.ops.pop()
        except IndexError:
            pass
        return self

    def document(self):
        parts = []
        for op in self:
            insert = op.get('insert')
            if insert or insert == '':
                if isinstance(insert, str):
                    parts.append(insert)
                else:
                    parts.append(NULL_CHARACTER)
            else:
                raise ValueError(
                    "document() can only be called on Deltas that have only insert ops")
        return "".join(parts)

    def __iter__(self):
        return iter(self.ops)

    def __getitem__(self, index):
        if isinstance(index, int):
            start = index
            stop = index + 1

        elif isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or None

            if index.step is not None:
                raise ValueError("no support for step slices")

        else:
            raise TypeError("Invalid argument type.")

        if (start is not None and start < 0) or (stop is not None and stop < 0):
            raise ValueError("no support for negative indexing.")

        ops = []
        iter = self.iterator()
        index = 0
        while iter.has_next():
            if stop is not None and index >= stop:
                break
            if index < start:
                next_op = iter.next(start - index)
            else:
                if stop is not None:
                    next_op = iter.next(stop-index)
                else:
                    next_op = iter.next()
                ops.append(next_op)
            index += op.length(next_op)

        return Delta(ops)

    def __len__(self):
        return sum(op.length(o) for o in self.ops)

    def iterator(self):
        return op.iterator(self.ops)

    def change_length(self):
        length = 0
        for operator in self:
            if op.type(operator) == 'delete':
                length -= operator['delete']
            else:
                length += op.length(operator)
        return length

    def length(self):
        return sum(op.length(o) for o in self)

    def compose(self, other):
        self_it = self.iterator()
        other_it = other.iterator()
        ops = []
        first_other = other_it.peek()

        if first_other and isinstance(first_other.get('retain'), (int, float)) and first_other.get('attributes') is None:
            first_left = first_other.get('retain')
            while self_it.peek_type() == 'insert' and self_it.peek_length() <= first_left:
                first_left -= self_it.peek_length()
                ops.append(self_it.next())
            if (first_other.get('retain') - first_left > 0):
                other_it.next(first_other.get('retain') - first_left)

        delta = self.__class__(ops)
        while self_it.has_next() or other_it.has_next():
            if other_it.peek_type() == 'insert':
                delta.push(other_it.next())
            elif self_it.peek_type() == 'delete':
                delta.push(self_it.next())
            else:
                length = min(self_it.peek_length(),
                             other_it.peek_length())
                self_op = self_it.next(length)
                other_op = other_it.next(length)
                if other_op.get('retain'):
                    new_op = {}
                    if isinstance(self_op.get('retain'), (int, float)):
                        new_op['retain'] = isinstance(self_op.get(
                            'retain'), (int, float)) and length or other_op.get('retain')
                    else:
                        if isinstance(other_op.get('retain'), (int, float)):
                            if self_op.get('retain') is None:
                                new_op['insert'] = self_op.get('insert')
                            else:
                                new_op['retain'] = self_op.get('retain')
                        else:
                            action = self_op.get(
                                "retain") is None and 'insert' or 'retain'
                            [embed_type, self_data, other_data] = get_embed_type_and_data(
                                self_op.get(action), other_op.get('retain'))
                            handler = Delta.get_handler(embed_type)
                            new_op[action] = {}
                            new_op[action][embed_type] = handler.compose(
                                self_data, other_data, action == 'retain')

                    # Preserve null when composing with a retain, otherwise remove it for inserts
                    attributes = op.compose(self_op.get('attributes'), other_op.get(
                        'attributes'), isinstance(self_op.get('retain'), (int, float)))
                    if (attributes):
                        new_op['attributes'] = attributes
                    delta.push(new_op)

                    if not other_it.has_next() and delta.ops[-1] == new_op:
                        rest = Delta(self_it.rest())

                        return delta.concat(rest).chop()
                # Other op should be delete, we could be an insert or retain
                # Insert + delete cancels out
                elif op.type(other_op) == 'delete' and isinstance(self_op.get('retain'), (int, float, dict)):
                    delta.push(other_op)
        return delta.chop()

    def diff(self, other):
        """
        Returns a diff of two *documents*, which is defined as a delta
        with only inserts.
        """
        if self.ops == other.ops:
            return self.__class__()

        self_doc = self.document()
        other_doc = other.document()
        self_it = self.iterator()
        other_it = other.iterator()

        delta = self.__class__()
        for code, text in differ(self_doc, other_doc):
            length = len(text)
            while length > 0:
                op_length = 0
                if code == DIFF_INSERT:
                    op_length = min(other_it.peek_length(), length)
                    delta.push(other_it.next(op_length))
                elif code == DIFF_DELETE:
                    op_length = min(length, self_it.peek_length())
                    self_it.next(op_length)
                    delta.delete(op_length)
                elif code == DIFF_EQUAL:
                    op_length = min(self_it.peek_length(),
                                    other_it.peek_length(), length)
                    self_op = self_it.next(op_length)
                    other_op = other_it.next(op_length)
                    if self_op.get('insert') == other_op.get('insert'):
                        attributes = op.diff(self_op.get(
                            'attributes'), other_op.get('attributes'))
                        delta.retain(op_length, **(attributes or {}))
                    else:
                        delta.push(other_op).delete(op_length)
                else:
                    raise RuntimeError(
                        "Diff library returned unknown op code: %r", code)
                if op_length == 0:
                    return
                length -= op_length
        return delta.chop()

    def each_line(self, fn, newline='\n'):
        for line, attributes, index in self.iter_lines():
            if fn(line, attributes, index) is False:
                break

    def iter_lines(self, newline='\n'):
        iter = self.iterator()
        line = self.__class__()
        i = 0
        is_previous_newline = False
        while iter.has_next():
            if iter.peek_type() != 'insert':
                return
            self_op = iter.peek()
            start = op.length(self_op) - iter.peek_length()
            if isinstance(self_op.get('insert'), str):
                index = self_op['insert'][start:].find(newline)
            else:
                index = -1

            if index < 0:
                line.push(iter.next())
                is_previous_newline = False
            elif index > 0:
                line.push(iter.next(index))
                is_previous_newline = False
            else:
                attributes = iter.next(1).get('attributes', {})
                if not line and attributes.get('code-block'):
                    # Code block's <pre> html tag handles newline characters automatically
                    yield Delta([{'insert': '\n'}]), attributes, i
                else:
                    yield line, attributes, i
                i += 1
                if is_previous_newline and not attributes.get('code-block'):
                    yield Delta([{'insert': ''}]), attributes, i  # adds <br> tag
                    i += 1
                line = Delta()
                is_previous_newline = True
        if len(line) > 0:
            yield line, {}, i

    def invert(self, base):
        inverted = Delta()

        def fn(base_index, operator):
            if op.type(operator) == 'insert':
                inverted.delete(op.length(operator))
            elif isinstance(operator.get('retain'), (int, float)) and operator.get("attributes") is None:
                inverted.retain(operator.get("retain"))
                return base_index + operator.get("retain")
            elif op.type(operator) == 'delete' or isinstance(operator.get('retain'), (int, float)):
                length = int(operator.get("delete") or operator.get("retain"))

                for base_operator in base[base_index:base_index + length]:
                    if op.type(operator) == 'delete':
                        inverted.push(base_operator)
                    elif operator.get('retain') and operator.get("attributes"):
                        inverted.retain(op.length(base_operator), **(op.invert(
                            operator.get("attributes"), base_operator.get("attributes")) or {}))
                return base_index + length
            elif isinstance(operator.get('retain'), dict):
                base_operator = op.iterator(base[base_index].ops).next()
                [embed_type, op_data, base_op_data] = get_embed_type_and_data(
                    operator.get('retain'), base_operator.get('insert'))
                handler = Delta.get_handler(embed_type)
                new_embed = {}
                new_embed[embed_type] = handler.invert(op_data, base_op_data)
                inverted.retain(new_embed, **(op.invert(operator.get(
                    'attributes'), base_operator.get('attributes')) or {}))
                return base_index + 1

            return base_index

        reduce(fn, self.ops, 0)
        return inverted.chop()

    def transform(self, other, priority=False):
        if isinstance(other, (int, float)):
            return self.transform_position(other, priority)

        self_it = self.iterator()
        other_it = other.iterator()
        delta = Delta()

        while self_it.has_next() or other_it.has_next():
            if self_it.peek_type() == 'insert' and (priority or other_it.peek_type() != 'insert'):
                delta.retain(op.length(self_it.next()))
            elif other_it.peek_type() == 'insert':
                delta.push(other_it.next())
            else:
                length = min(self_it.peek_length(),
                             other_it.peek_length())
                self_op = self_it.next(length)
                other_op = other_it.next(length)
                if self_op.get('delete'):
                    # Our delete either makes their delete redundant or removes their retain
                    continue
                elif other_op.get('delete'):
                    delta.push(other_op)
                else:
                    self_data = self_op.get('retain')
                    other_data = other_op.get('retain')
                    transformed_data = isinstance(
                        other_data, dict) and other_data or length

                    if isinstance(self_data, dict) and isinstance(other_data, dict):
                        embed_type = next(iter(self_data))
                        if embed_type == next(iter(other_data)):
                            handler = Delta.get_handler(embed_type)
                            if handler:
                                transformed_data = {}
                                transformed_data[embed_type] = handler.transform(
                                    self_data.get(embed_type), other_data.get(embed_type), priority)
                    # We retain either their retain or insert
                    delta.retain(transformed_data, **(op.transform(self_op.get('attributes'),
                                                                   other_op.get('attributes'), priority) or {}))

        return delta.chop()

    def transform_position(self, index, priority=False):
        iter = self.iterator()
        offset = 0
        while iter.has_next() and offset <= index:
            length = iter.peek_length()
            next_type = iter.peek_type()
            iter.next()
            if next_type == 'delete':
                index -= min(length, index - offset)
                continue
            elif next_type == 'insert' and (offset < index or not priority):
                index += length
            offset += length
        return index
