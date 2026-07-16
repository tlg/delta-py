"""
Generate cross-language golden fixtures for the positional move algebra.

The Python implementation is the reference: this script dumps seeded
hand and fuzz cases (compose, transform, invert, transform_position,
coordinates) as JSON so the TypeScript port can assert byte-identical
behavior.  Cases whose documents contain ``cell`` embeds require the
reference cell handler (child ``ops`` sequences run through Delta);
they carry ``"handlers": ["cell"]``.

Usage: python tools/generate_move_fixtures.py <output-dir>
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'tests'))

from delta import Delta, MoveDelta, transform_coordinate  # noqa: E402
import test_moves as T  # noqa: E402


class CellHandler:
    @staticmethod
    def compose(a, b, keep_null):
        return {'ops': MoveDelta(list(a['ops'])).compose(MoveDelta(list(b['ops']))).ops}

    @staticmethod
    def transform(a, b, priority):
        return {'ops': MoveDelta(list(a['ops'])).transform(MoveDelta(list(b['ops'])), priority).ops}

    @staticmethod
    def invert(change, base):
        return {'ops': MoveDelta(list(change['ops'])).invert(MoveDelta(list(base['ops']))).ops}


Delta.register_embed('cell', CellHandler)


def uses_cell(*deltas):
    def walk(value):
        if isinstance(value, dict):
            return 'cell' in value or any(walk(v) for v in value.values())
        if isinstance(value, list):
            return any(walk(v) for v in value)
        return False
    return any(walk(d.ops) for d in deltas)


def case(**fields):
    deltas = [v for v in fields.values() if isinstance(v, MoveDelta)]
    out = {k: (v.ops if isinstance(v, MoveDelta) else v) for k, v in fields.items()}
    if uses_cell(*deltas):
        out['handlers'] = ['cell']
    return out


def main(out_dir):
    rng = random.Random(20260716)
    apply_cases, compose_cases, transform_cases = [], [], []
    invert_cases, position_cases, coordinate_cases = [], [], []

    for _ in range(120):
        base = T.random_doc(rng)
        if len(base) < 3:
            continue
        kind = rng.random()
        if kind < 0.3:
            move = T.random_move(rng, len(base))
        elif kind < 0.55:
            move = T.random_carried_move(rng, len(base))
        elif kind < 0.8:
            base = (MoveDelta().insert('AB')
                    .insert(T.cell('Hello')).insert('CD'))
            move = T.random_cross_move(rng, base)
        else:
            base = T.two_cell_base(rng)
            move = T.random_cell_to_cell_move(rng, base)
        after = T.apply(base, move)
        apply_cases.append(case(doc=base, delta=move, expected=after))

        edit = T.random_ordinary(rng, len(after))
        compose_cases.append(case(first=move, second=edit,
                                  expected=move.compose(edit)))

        concurrent = T.random_ordinary(rng, len(base))
        for priority in (True, False):
            transform_cases.append(case(
                left=move, right=concurrent, priority=priority,
                expected=move.transform(concurrent, priority)))
        other_move = T.random_move(rng, len(base), ref='z')
        transform_cases.append(case(
            left=move, right=other_move, priority=True,
            expected=move.transform(other_move, True)))

        invert_cases.append(case(delta=move, base=base,
                                 expected=move.invert(base)))

        for index in range(0, len(base) + 1, max(1, len(base) // 3)):
            position_cases.append(case(
                delta=move, position=index, priority=False,
                expected=move.transform_position(index)))

    # cross-level, trash reads and coordinates over the shared deep doc
    doc = MoveDelta().insert('AB').insert(T.cell('Hello')).insert('CD')
    read = T.trash_read_delta()
    apply_cases.append(case(doc=doc, delta=read, expected=T.apply(doc, read)))
    invert_cases.append(case(delta=read, base=doc,
                             expected=read.invert(doc)))
    for coordinate in [(1,), (4,), (2, 'ops', 1), (2, 'ops', 4)]:
        coordinate_cases.append(case(
            delta=read, coordinate=list(coordinate),
            expected=(list(transform_coordinate(read, coordinate))
                      if transform_coordinate(read, coordinate) is not None
                      else None)))

    # UTF-16: moves across an astral character
    emoji_doc = MoveDelta().insert('a\N{GRINNING FACE}b')
    swap = MoveDelta().cut('m', 1).retain(2).paste('m', 0, 1)
    apply_cases.append(case(doc=emoji_doc, delta=swap,
                            expected=T.apply(emoji_doc, swap)))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, cases in [('moves-apply', apply_cases),
                        ('moves-compose', compose_cases),
                        ('moves-transform', transform_cases),
                        ('moves-invert', invert_cases),
                        ('moves-transform-position', position_cases),
                        ('moves-coordinates', coordinate_cases)]:
        path = out / f'{name}.json'
        # ensure_ascii escapes lone surrogates (\ud83d) from UTF-16
        # splits — valid JSON that JavaScript strings hold natively
        path.write_text(json.dumps({'tests': cases}, indent=1) + '\n')
        print(f'{path}: {len(cases)} cases')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'fixtures-out')
