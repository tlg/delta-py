import copy
from functools import reduce

import diff_match_patch

from . import op


NULL_CHARACTER = chr(0)
DIFF_EQUAL = 0
DIFF_INSERT = 1
DIFF_DELETE = -1


def differ(a, b, timeout=1):
    dmp = diff_match_patch.diff_match_patch()
    dmp.Diff_Timeout = timeout
    result = dmp.diff_main(a, b)
    dmp.diff_cleanupSemantic(result)
    return result


def get_embed_type_and_data(a, b):
    if not isinstance(a, dict) or a is None:
        raise TypeError(f'cannot retain a {type(a).__name__}')
    if not isinstance(b, dict) or b is None:
        raise TypeError(f'cannot retain a {type(b).__name__}')

    embed_type = next(iter(a))
    b_type = next(iter(b))
    if not embed_type or embed_type != b_type:
        raise ValueError(f'embed types not matched: {embed_type} != {b_type}')
    return [embed_type, a[embed_type], b[embed_type]]


handlers = {}


class Delta:

    @staticmethod
    def register_embed(embed_type, handler):
        handlers[embed_type] = handler

    @staticmethod
    def unregister_embed(embed_type):
        handlers.pop(embed_type, None)

    @staticmethod
    def get_handler(embed_type):
        handler = handlers.get(embed_type)
        if handler is None:
            raise ValueError(f'no handlers for embed type "{embed_type}"')
        return handler

    def __init__(self, ops=None, **attrs):
        if hasattr(ops, 'ops'):
            ops = ops.ops
        self.ops = ops or []
        self.__dict__.update(attrs)

    def __eq__(self, other):
        return self.ops == other.ops

    def __repr__(self):
        return f'{self.__class__.__name__}({self.ops})'

    def insert(self, text, **attrs):
        if text == '':
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
        if index == 0:
            self.ops.append(new_op)
            return self

        last_op = self.ops[index - 1]

        if op.type(new_op) == op.type(last_op) == 'delete':
            last_op['delete'] += new_op['delete']
            return self

        if op.type(last_op) == 'delete' and op.type(new_op) == 'insert':
            index -= 1
            if index == 0:
                self.ops.insert(0, new_op)
                return self
            last_op = self.ops[index - 1]

        if new_op.get('attributes') == last_op.get('attributes'):
            if isinstance(new_op.get('insert'), str) and isinstance(last_op.get('insert'), str):
                last_op['insert'] += new_op['insert']
                return self

            if isinstance(new_op.get('retain'), (int, float)) and isinstance(last_op.get('retain'), (int, float)):
                last_op['retain'] += new_op['retain']
                if isinstance(new_op.get('attributes'), dict):
                    last_op['attributes'] = new_op['attributes']
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
        if self.ops:
            last_op = self.ops[-1]
            if isinstance(last_op.get('retain'), (int, float)) and not last_op.get('attributes'):
                self.ops.pop()
        return self

    def document(self):
        parts = []
        for o in self:
            insert = o.get('insert')
            if insert or insert == '':
                if isinstance(insert, str):
                    parts.append(insert)
                else:
                    parts.append(NULL_CHARACTER)
            else:
                raise ValueError(
                    'document() can only be called on Deltas that have only insert ops')
        return ''.join(parts)

    def __iter__(self):
        return iter(self.ops)

    def __getitem__(self, item):
        if isinstance(item, int):
            start = item
            stop = item + 1
        elif isinstance(item, slice):
            start = item.start or 0
            stop = item.stop
            if item.step is not None:
                raise ValueError('no support for step slices')
        else:
            raise TypeError('Invalid argument type.')

        if (start is not None and start < 0) or (stop is not None and stop < 0):
            raise ValueError('no support for negative indexing.')

        ops = []
        it = self.iterator()
        pos = 0
        while it.has_next():
            if stop is not None and pos >= stop:
                break
            if pos < start:
                next_op = it.next(start - pos)
            else:
                next_op = it.next(stop - pos if stop is not None else None)
                ops.append(next_op)
            pos += op.length(next_op)

        return Delta(ops)

    def __len__(self):
        return sum(op.length(o) for o in self.ops)

    def iterator(self):
        return op.iterator(self.ops)

    def change_length(self):
        length = 0
        for o in self:
            if op.type(o) == 'delete':
                length -= o['delete']
            elif op.type(o) == 'insert':
                length += op.length(o)
        return length

    def length(self):
        return sum(op.length(o) for o in self)

    def compose(self, other):
        self_it = self.iterator()
        other_it = other.iterator()
        ops = []
        first_other = other_it.peek()

        if (first_other
                and isinstance(first_other.get('retain'), (int, float))
                and first_other.get('attributes') is None):
            first_left = first_other['retain']
            while self_it.peek_type() == 'insert' and self_it.peek_length() <= first_left:
                first_left -= self_it.peek_length()
                ops.append(self_it.next())
            if first_other['retain'] - first_left > 0:
                other_it.next(first_other['retain'] - first_left)

        delta = self.__class__(ops)
        while self_it.has_next() or other_it.has_next():
            if other_it.peek_type() == 'insert':
                delta.push(other_it.next())
            elif self_it.peek_type() == 'delete':
                delta.push(self_it.next())
            else:
                length = min(self_it.peek_length(), other_it.peek_length())
                self_op = self_it.next(length)
                other_op = other_it.next(length)
                if other_op.get('retain'):
                    new_op = {}
                    if isinstance(self_op.get('retain'), (int, float)):
                        new_op['retain'] = (
                            length if isinstance(other_op.get('retain'), (int, float))
                            else other_op['retain']
                        )
                    elif isinstance(other_op.get('retain'), (int, float)):
                        if self_op.get('retain') is None:
                            new_op['insert'] = self_op.get('insert')
                        else:
                            new_op['retain'] = self_op.get('retain')
                    else:
                        action = 'insert' if self_op.get('retain') is None else 'retain'
                        embed_type, self_data, other_data = get_embed_type_and_data(
                            self_op.get(action), other_op.get('retain'))
                        handler = Delta.get_handler(embed_type)
                        new_op[action] = {
                            embed_type: handler.compose(
                                self_data, other_data, action == 'retain')
                        }

                    attributes = op.compose(
                        self_op.get('attributes'),
                        other_op.get('attributes'),
                        isinstance(self_op.get('retain'), (int, float)))
                    if attributes:
                        new_op['attributes'] = attributes
                    delta.push(new_op)

                    if not other_it.has_next() and delta.ops[-1] == new_op:
                        rest = Delta(self_it.rest())
                        return delta.concat(rest).chop()
                elif (op.type(other_op) == 'delete'
                      and isinstance(self_op.get('retain'), (int, float, dict))):
                    delta.push(other_op)
        return delta.chop()

    def diff(self, other):
        """
        Returns a diff of two *documents* (Deltas with only insert ops).
        """
        if self.ops == other.ops:
            return self.__class__()

        self_doc = self.document()
        other_doc = other.document()
        self_it = self.iterator()
        other_it = other.iterator()

        delta = self.__class__()
        for code, text in differ(self_doc, other_doc):
            length = op.utf16_len(text)
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
                    op_length = min(
                        self_it.peek_length(),
                        other_it.peek_length(),
                        length)
                    self_op = self_it.next(op_length)
                    other_op = other_it.next(op_length)
                    if self_op.get('insert') == other_op.get('insert'):
                        attributes = op.diff(
                            self_op.get('attributes'),
                            other_op.get('attributes'))
                        delta.retain(op_length, **(attributes or {}))
                    else:
                        delta.push(other_op).delete(op_length)
                else:
                    raise RuntimeError(
                        f'Diff library returned unknown op code: {code!r}')
                if op_length == 0:
                    return
                length -= op_length
        return delta.chop()

    def each_line(self, fn, newline='\n'):
        for line, attributes, index in self.iter_lines(newline):
            if fn(line, attributes, index) is False:
                break

    def iter_lines(self, newline='\n'):
        it = self.iterator()
        line = self.__class__()
        i = 0
        while it.has_next():
            if it.peek_type() != 'insert':
                return
            current_op = it.peek()
            start = op.length(current_op) - it.peek_length()
            if isinstance(current_op.get('insert'), str):
                nl_index = current_op['insert'][start:].find(newline)
            else:
                nl_index = -1

            if nl_index < 0:
                line.push(it.next())
            elif nl_index > 0:
                line.push(it.next(nl_index))
            else:
                attributes = it.next(1).get('attributes', {})
                yield line, attributes, i
                i += 1
                line = Delta()
        if len(line) > 0:
            yield line, {}, i

    def invert(self, base):
        inverted = Delta()

        def fn(base_index, operator):
            if op.type(operator) == 'insert':
                inverted.delete(op.length(operator))
            elif isinstance(operator.get('retain'), (int, float)) and operator.get('attributes') is None:
                inverted.retain(operator['retain'])
                return base_index + operator['retain']
            elif op.type(operator) == 'delete' or isinstance(operator.get('retain'), (int, float)):
                length = int(operator.get('delete') or operator.get('retain'))
                for base_op in base[base_index:base_index + length]:
                    if op.type(operator) == 'delete':
                        inverted.push(base_op)
                    elif operator.get('retain') and operator.get('attributes'):
                        inverted.retain(
                            op.length(base_op),
                            **(op.invert(
                                operator.get('attributes'),
                                base_op.get('attributes')) or {}))
                return base_index + length
            elif isinstance(operator.get('retain'), dict):
                base_op = op.iterator(base[base_index].ops).next()
                embed_type, op_data, base_op_data = get_embed_type_and_data(
                    operator['retain'], base_op.get('insert'))
                handler = Delta.get_handler(embed_type)
                new_embed = {embed_type: handler.invert(op_data, base_op_data)}
                inverted.retain(
                    new_embed,
                    **(op.invert(
                        operator.get('attributes'),
                        base_op.get('attributes')) or {}))
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
                length = min(self_it.peek_length(), other_it.peek_length())
                self_op = self_it.next(length)
                other_op = other_it.next(length)
                if self_op.get('delete'):
                    continue
                elif other_op.get('delete'):
                    delta.push(other_op)
                else:
                    self_data = self_op.get('retain')
                    other_data = other_op.get('retain')
                    transformed_data = other_data if isinstance(other_data, dict) else length

                    if isinstance(self_data, dict) and isinstance(other_data, dict):
                        embed_type = next(iter(self_data))
                        if embed_type == next(iter(other_data)):
                            handler = Delta.get_handler(embed_type)
                            if handler:
                                transformed_data = {
                                    embed_type: handler.transform(
                                        self_data[embed_type],
                                        other_data[embed_type],
                                        priority)
                                }
                    delta.retain(
                        transformed_data,
                        **(op.transform(
                            self_op.get('attributes'),
                            other_op.get('attributes'),
                            priority) or {}))

        return delta.chop()

    def transform_position(self, index, priority=False):
        it = self.iterator()
        offset = 0
        while it.has_next() and offset <= index:
            length = it.peek_length()
            next_type = it.peek_type()
            it.next()
            if next_type == 'delete':
                index -= min(length, index - offset)
                continue
            elif next_type == 'insert' and (offset < index or not priority):
                index += length
            offset += length
        return index
