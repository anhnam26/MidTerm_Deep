"""Local, backed-up paired DataNew cleanup. Never modifies acquisition history.

Run prepare first, inspect its report, then promote explicitly. All ambiguous
perceptual matches are excluded, not silently treated as independent sources.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

from PIL import AvifImagePlugin, Image, ImageOps

from .common import CLASSES, ROOT, read_csv, write_csv, write_json
from .data import dhash
from .dataset_acquire import prepare as prepare_native
from .dataset_review import load_approval, snapshot_records

# Avoid starting dozens of codec threads for every tiny image on Windows.
AvifImagePlugin.MAX_THREADS = 1
LIMITS = {224: 25_000_000, 32: 20_000_000}
EXTENSIONS = {224: '.avif', 32: '.webp'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return (row['split'], row['class'], Path(row['output']).stem)


def image_path(base, identity, size):
    split, label, stem = identity
    return Path(base) / f'Mid{size}' / split / label / (stem + EXTENSIONS[size])


def visual_hashes(image):
    w, h = image.size
    return [dhash(image), dhash(ImageOps.mirror(image)), dhash(ImageOps.flip(image)),
            dhash(image.crop((w // 10, h // 10, w * 9 // 10, h * 9 // 10)))]


def inspect_image(path, size):
    with Image.open(path) as image:
        image.load()
        if image.mode != 'RGB' or image.size != (size, size) or image.n_frames != 1:
            raise ValueError(f'Wrong image geometry/mode/frames: {path}')
        if image.format != ('AVIF' if size == 224 else 'WEBP'):
            raise ValueError(f'Wrong image format: {path}')
        pixels = hashlib.sha256(image.tobytes()).hexdigest()
        visual = visual_hashes(image) if size == 224 else []
    return {'sha256': sha(path), 'pixel_sha256': pixels,
            'bytes': path.stat().st_size, 'visual': visual}


class NearIndex:
    """Seven disjoint bit chunks: distance <=6 must share at least one chunk."""
    def __init__(self):
        self.buckets = defaultdict(set)
        self.values = {}

    @staticmethod
    def chunks(value):
        return [(i, (value >> (i * 9)) & (1023 if i == 6 else 511)) for i in range(7)]

    def add(self, identifier, hashes):
        # Base + center crop as references; mirror/flip also tested as queries.
        for text in (hashes[0], hashes[-1]):
            value = int(text, 16)
            token = (identifier, value)
            self.values[token] = value
            for chunk in self.chunks(value):
                self.buckets[chunk].add(token)

    def matches(self, hashes):
        matches = set()
        for text in hashes:
            value = int(text, 16)
            candidates = set()
            for chunk in self.chunks(value):
                candidates.update(self.buckets.get(chunk, ()))
            matches.update(token[0] for token in candidates
                           if (self.values[token] ^ value).bit_count() <= 6)
        return sorted(matches)


def inventory(base, rows):
    result = []
    expected = set()
    for n, row in enumerate(rows):
        identity = key(row)
        if identity in expected:
            raise ValueError(f'Duplicate manifest output: {identity}')
        if row['class'] not in CLASSES or row['split'] not in ('train', 'test'):
            raise ValueError(f'Unknown class/split: {row}')
        if row['type'] not in ('real', 'augmented'):
            raise ValueError(f'Unknown manifest type: {row}')
        expected.add(identity)
        result.append({'key': list(identity), 'row': row,
                       'images': {str(s): inspect_image(image_path(base, identity, s), s)
                                  for s in (224, 32)}})
        if (n + 1) % 1000 == 0:
            print('INVENTORY', n + 1, flush=True)
    for size in (224, 32):
        actual = {tuple([*p.relative_to(Path(base) / f'Mid{size}').parts[:2], p.stem])
                  for p in (Path(base) / f'Mid{size}').rglob('*') if p.is_file()}
        if actual != expected:
            raise ValueError(f'Manifest/files mismatch Mid{size}')
        if len(list((Path(base) / f'Mid{size}').rglob('*' + EXTENSIONS[size]))) != len(rows):
            raise ValueError('Unexpected extension or duplicate stem')
    return result


def choose_retained(items):
    """Test wins over train; fail closed on near matches, never keep augments."""
    kept, removed = [], []
    exact = {}
    sources = {}
    near = NearIndex()
    for item in sorted(items, key=lambda x: (x['key'][0] != 'test', x['key'])):
        identity = '/'.join(item['key'])
        row = item['row']
        reason, matches = None, []
        if row['type'] == 'augmented':
            reason = 'manifest_augmented'
        else:
            source = (row['class'], row['source'])
            matches = [sources[source]] if source in sources else []
            for s in ('224', '32'):
                for field in ('sha256', 'pixel_sha256'):
                    match = exact.get((s, field, item['images'][s][field]))
                    if match:
                        matches.append(match)
            if matches:
                reason = 'exact_content_or_source_duplicate'
            else:
                matches = near.matches(item['images']['224']['visual'])
                if matches:
                    reason = 'near_match_unresolved_excluded'
        if reason:
            removed.append({'key': item['key'], 'reason': reason, 'matches': sorted(set(matches))})
            continue
        kept.append(item)
        sources[(row['class'], row['source'])] = identity
        for s in ('224', '32'):
            for field in ('sha256', 'pixel_sha256'):
                exact[(s, field, item['images'][s][field])] = identity
        near.add(identity, item['images']['224']['visual'])
    return kept, removed


def metadata_baseline(root):
    base = root / 'data/metadata'
    names = ('attempts.jsonl', 'user_approval.json', 'quota_retirements.json',
             'progress.json', 'candidates.json', 'additional_candidates.json')
    paths = [base / 'fruit_acquisition_08' / n for n in names]
    paths += list(base.rglob('development_reservations.json'))
    return {p.relative_to(root).as_posix(): sha(p) for p in paths if p.exists()}


def assert_baseline(root, baseline):
    if metadata_baseline(root) != baseline:
        raise ValueError('Acquisition metadata/reservations changed; abort')
    meta = root / 'data/metadata/fruit_acquisition_08'
    if (meta / 'worker.lock').exists() or (meta / 'STOP').exists():
        raise ValueError('Acquisition lock/STOP exists; abort')


def verify_copy(source, destination):
    original = {p.relative_to(source).as_posix(): sha(p)
                for p in source.rglob('*') if p.is_file()}
    copied = {p.relative_to(destination).as_posix(): sha(p)
              for p in destination.rglob('*') if p.is_file()}
    if original != copied:
        raise ValueError('Backup file hashes do not match')
    return original


def encode_pairs(pairs, target, identity):
    for size in (224, 32):
        path = image_path(target, identity, size)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        if size == 224:
            pairs[size].save(path, format='AVIF', quality=6, speed=6, max_threads=1)
        else:
            pairs[size].save(path, format='WEBP', quality=90, method=4)
    return {str(s): inspect_image(image_path(target, identity, s), s) for s in (224, 32)}


def audit_items(items):
    counts = Counter((i['key'][0], i['key'][1]) for i in items)
    totals = {s: sum(i['images'][str(s)]['bytes'] for i in items) for s in (224, 32)}
    exact = defaultdict(list)
    for item in items:
        for s in ('224', '32'):
            for field in ('sha256', 'pixel_sha256'):
                exact[(s, field, item['images'][s][field])].append(item['key'])
    duplicates = [v for v in exact.values() if len(v) > 1]
    shortage = {split: {c: max(0, quota - counts[split, c]) for c in CLASSES}
                for split, quota in (('train', 5000), ('test', 50))}
    return {'pairs': len(items), 'counts': {split: {c: counts[split, c] for c in CLASSES}
                                          for split in ('train', 'test')},
            'bytes': {str(s): totals[s] for s in totals}, 'limits_bytes': LIMITS,
            'capacity_pass': all(totals[s] <= LIMITS[s] for s in totals),
            'duplicate_groups_by_hash_kind': len(duplicates),
            'duplicate_examples': duplicates[:10], 'shortage': shortage,
            'quota_pass': all(counts[split, c] == quota for split, quota in
                              (('train', 5000), ('test', 50)) for c in CLASSES),
            'augmented': sum(i['row']['type'] == 'augmented' for i in items)}


def prepare_run(root, work):
    work.mkdir(parents=True, exist_ok=False)
    baseline = metadata_baseline(root)
    assert_baseline(root, baseline)
    write_json(work / 'baseline.json', baseline)
    source = root / 'DataNew'
    backup = work / 'backup/DataNew'
    shutil.copytree(source, backup)
    manifest = root / 'results/dataset_manifest.csv'
    shutil.copy2(manifest, work / 'backup/dataset_manifest.csv')
    backup_hashes = verify_copy(source, backup)
    if sha(manifest) != sha(work / 'backup/dataset_manifest.csv'):
        raise ValueError('Manifest backup mismatch')
    write_json(work / 'backup_hashes.json', backup_hashes)
    print('BACKUP_VERIFIED', len(backup_hashes), flush=True)
    original = inventory(backup, read_csv(work / 'backup/dataset_manifest.csv'))
    write_json(work / 'original_inventory.json', original)
    kept, removed = choose_retained(original)
    write_json(work / 'removed.json', removed)
    print('RETAINED', len(kept), 'REMOVED', len(removed), dict(Counter(r['reason'] for r in removed)), flush=True)
    stage = work / 'staging/DataNew'
    for item in kept:
        for s in (224, 32):
            target = image_path(stage, item['key'], s)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path(backup, item['key'], s), target)
    # All original previews are forbidden references for NEW replacements, even
    # removed augments, so recompressed/cropped old sources cannot fill a gap.
    index = NearIndex()
    exact = set()
    for item in original:
        index.add('/'.join(item['key']), item['images']['224']['visual'])
        exact.update((s, field, item['images'][s][field]) for s in ('224', '32')
                     for field in ('sha256', 'pixel_sha256'))
    counts = Counter(i['key'][1] for i in kept if i['key'][0] == 'train')
    meta = root / 'data/metadata/fruit_acquisition_08'
    records = snapshot_records(meta / 'attempts.jsonl')
    approval = load_approval(meta / 'user_approval.json')
    approved = {r['id'] for r in approval['accepted']} if approval else set()
    records.sort(key=lambda r: (r['id'] not in approved, r['class'], r['id']))
    decisions, selected, source_hashes = [], [], set()
    for n, row in enumerate(records):
        label = row['class']
        if row['status'] != 'pending_review' or counts[label] >= 5000:
            continue
        decision = {'id': row['id'], 'class': label}
        try:
            original_path = root / row['original_path']
            if sha(original_path) != row['original_sha256']:
                raise ValueError('Original SHA mismatch')
            if any(h in row.get('source_name', '').lower() for h in ('augment', 'synthetic', 'edited')):
                raise ValueError('Augmentation/synthetic source hint')
            if row['original_sha256'] in source_hashes:
                raise ValueError('Already selected source SHA')
            with Image.open(original_path) as im:
                native, pairs = prepare_native(im)
            native_sha = hashlib.sha256(str(native.size).encode() + native.tobytes()).hexdigest()
            if native_sha != row['native_pixel_sha256']:
                raise ValueError('Native pixel SHA mismatch')
            if hashlib.sha256(pairs[224].tobytes()).hexdigest() != row['pixel224_sha256']:
                raise ValueError('Prepared pixel SHA mismatch')
            matches = index.matches(visual_hashes(native))
            matches += index.matches(visual_hashes(pairs[224]))
            if matches:
                decision.update(reason='near_match_existing_or_selected', matches=sorted(set(matches))[:20])
                decisions.append(decision)
                continue
            identity = ('train', label, f'{label}_source_{row["id"]}')
            images = encode_pairs(pairs, stage, identity)
            collisions = [(s, field) for s in ('224', '32') for field in ('sha256', 'pixel_sha256')
                          if (s, field, images[s][field]) in exact]
            matches = index.matches(images['224']['visual'])
            if collisions or matches:
                for s in (224, 32):
                    image_path(stage, identity, s).unlink()
                decision.update(reason='encoded_duplicate_or_near_match', matches=matches[:20])
                decisions.append(decision)
                continue
            newrow = {'class': label, 'split': 'train', 'type': 'real',
                      'source': row['source_identity'], 'output': identity[-1] + '.avif'}
            item = {'key': list(identity), 'row': newrow, 'images': images}
            kept.append(item)
            selected.append({'key': list(identity), 'acquisition': row,
                             'user_confirmed': row['id'] in approved,
                             'label_rights_review': 'preserved_not_autoapproved'})
            counts[label] += 1
            source_hashes.add(row['original_sha256'])
            source_hashes.add(native_sha)
            index.add(row['id'], visual_hashes(native))
            index.add(row['id'] + ':prepared', images['224']['visual'])
            exact.update((s, field, images[s][field]) for s in ('224', '32')
                         for field in ('sha256', 'pixel_sha256'))
            decision['reason'] = 'selected'
        except (OSError, ValueError, KeyError) as error:
            decision.update(reason='invalid_or_unverified_source', error=str(error))
        decisions.append(decision)
        if len(decisions) % 100 == 0:
            print('CANDIDATES', len(decisions), 'SELECTED', len(selected), dict(counts), flush=True)
            write_json(work / 'candidate_decisions.json', decisions)
            write_json(work / 'selected_sources.json', selected)
    write_json(work / 'candidate_decisions.json', decisions)
    write_json(work / 'selected_sources.json', selected)
    rows = [i['row'] for i in sorted(kept, key=lambda i: i['key'])]
    write_csv(work / 'staging/dataset_manifest.csv', rows)
    # Independent final decode and pixel hashes, not reused inventory.
    verified = inventory(stage, rows)
    write_json(work / 'final_inventory.json', verified)
    report = audit_items(verified)
    report.update({'removed_pairs': len(removed), 'removed_reasons': dict(Counter(r['reason'] for r in removed)),
                   'selected_pairs': len(selected), 'candidate_decisions': dict(Counter(r['reason'] for r in decisions)),
                   'backup': str(backup), 'codec': {'224': 'AVIF quality6 speed6', '32': 'WebP quality90 method4'},
                   'near_screening': 'dHash distance<=6 full/mirror/flip/center, fail closed; not exhaustive proof',
                   'semantic_rights_review_complete': False, 'training_ready': False,
                   'retained_provenance': 'Legacy real tag/source basename; original rights/group provenance unverified',
                   'selected_unapproved': sum(not s['user_confirmed'] for s in selected),
                   'status': 'prepared_not_promoted', 'updated_unix': time.time()})
    assert_baseline(root, baseline)
    write_json(work / 'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


def promote(root, work):
    baseline = json.loads((work / 'baseline.json').read_text(encoding='utf-8'))
    assert_baseline(root, baseline)
    report = json.loads((work / 'report.json').read_text(encoding='utf-8'))
    if not report['capacity_pass'] or report['augmented'] or report['duplicate_groups_by_hash_kind']:
        raise ValueError('Staging failed capacity/augmentation/duplicate gate')
    if report['status'] != 'prepared_not_promoted':
        raise ValueError('Already promoted or incomplete run')
    original_hashes = json.loads((work / 'backup_hashes.json').read_text(encoding='utf-8'))
    if verify_copy(root / 'DataNew', work / 'backup/DataNew') != original_hashes:
        raise ValueError('DataNew changed after backup')
    if sha(root / 'results/dataset_manifest.csv') != sha(work / 'backup/dataset_manifest.csv'):
        raise ValueError('Manifest changed after backup')
    expected = json.loads((work / 'final_inventory.json').read_text(encoding='utf-8'))
    for item in expected:
        for s in (224, 32):
            if sha(image_path(work / 'staging/DataNew', item['key'], s)) != item['images'][str(s)]['sha256']:
                raise ValueError('Staging changed after audit')
    previous = work / 'previous_DataNew'
    (root / 'DataNew').rename(previous)
    try:
        (work / 'staging/DataNew').rename(root / 'DataNew')
        shutil.copy2(work / 'staging/dataset_manifest.csv', root / 'results/dataset_manifest.csv')
    except Exception:
        if (root / 'DataNew').exists():
            (root / 'DataNew').rename(work / 'failed_promote_DataNew')
        previous.rename(root / 'DataNew')
        shutil.copy2(work / 'backup/dataset_manifest.csv', root / 'results/dataset_manifest.csv')
        raise
    # Keep verified backup AND original directory. Nothing destroyed permanently.
    report['status'] = 'promoted_partial' if not report['quota_pass'] else 'promoted_technical_checks_pass'
    report['updated_unix'] = time.time()
    assert_baseline(root, baseline)
    write_json(work / 'report.json', report)
    write_json(root / 'DataNew/cleanup_report.json', report)
    write_json(root / 'DataNew/source_provenance.json', {
        'selected': json.loads((work / 'selected_sources.json').read_text(encoding='utf-8')),
        'retained': [i['row'] for i in expected if '_source_' not in i['key'][-1]],
        'development_rule': 'All selected sources used for train only; acquisition reservations unchanged',
        'semantic_rights_review_complete': False})
    print('PROMOTED', json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'promote'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare_run(args.root.resolve(), args.work.resolve())
    else:
        promote(args.root.resolve(), args.work.resolve())


if __name__ == '__main__':
    main()