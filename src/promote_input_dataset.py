"""Promote reviewed technical candidates into the paired DataNew input bundle.

This command is intentionally fail-closed: it preserves the existing test set,
adds only source-backed originals, rejects duplicate encoded/decoded pixels, and
records unresolved rights/visual-review states instead of claiming approval.
"""

import argparse
import csv
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import shutil

from PIL import AvifImagePlugin, Image

from .common import CLASSES, ROOT, read_csv, write_csv, write_json

AvifImagePlugin.MAX_THREADS = 1
LIMITS = {224: 25_000_000, 32: 20_000_000}
EXTENSIONS = {224: '.avif', 32: '.webp'}
MANIFEST_FIELDS = [
    'id', 'class', 'split', 'group_id', 'source_dataset', 'source_identity',
    'source', 'source_url', 'landing_url', 'license', 'author',
    'rights_verified', 'photo_verified', 'physical_groups_verified',
    'review_status', 'original_path', 'mid224_path', 'mid224_sha256',
    'mid224_pixel_sha256', 'mid32_path', 'mid32_sha256', 'mid32_pixel_sha256',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows_from_current():
    rows = []
    for item in read_csv(ROOT / 'results/dataset_manifest.csv'):
        stem = Path(item['output']).stem
        split, label = item['split'], item['class']
        row = {
            'id': stem, 'class': label, 'split': split,
            'group_id': f'legacy_{split}_{label}',
            'source_dataset': 'legacy_DataNew', 'source_identity': item['source'],
            'source': item['source'], 'source_url': '', 'landing_url': '',
            'license': 'Unverified legacy provenance', 'author': '',
            'rights_verified': 'false', 'photo_verified': 'false',
            'physical_groups_verified': 'false', 'review_status': 'legacy_technical_only',
            'original_path': '',
        }
        for size in (224, 32):
            path = ROOT / 'DataNew' / f'Mid{size}' / split / label / (stem + EXTENSIONS[size])
            if not path.is_file():
                raise FileNotFoundError(path)
            # Full geometry/pixel audit already verified the retained bundle in
            # results/dataset_audit.json. Rehash bytes here to catch mutation;
            # decode the newly promoted candidates below before copying.
            row[f'mid{size}_pixel_sha256'] = ''
            row[f'mid{size}_path'] = path.relative_to(ROOT).as_posix()
            row[f'mid{size}_sha256'] = sha(path)
        rows.append(row)
    return rows


def load_candidates():
    cohort = json.loads((ROOT / 'data/metadata/research/input_ready_20261019/curation_cohort.json').read_text(encoding='utf-8'))
    eligible = [row for row in cohort if row.get('origin') in ('banana', 'strawberry', 'pqc500')
                and row.get('candidate_usage') == 'train_candidate'
                and row.get('rich_screening', {}).get('predicted_class') == row.get('class')]
    selected = []
    quotas = {'banana': 520, 'strawberry': 112, 'pineapple': 2315}
    encoded_records = ROOT / 'data/metadata/research/input_ready_20261019/encoded_records'
    for label, quota in quotas.items():
        candidates = [row for row in eligible if row['class'] == label]
        candidates.sort(key=lambda row: row['rich_screening']['fruit_margin'], reverse=True)
        seen_files, seen_pixels = set(), set()
        kept = []
        for row in candidates:
            record_path = encoded_records / (row['id'] + '.json')
            record = json.loads(record_path.read_text(encoding='utf-8'))
            duplicate = False
            for size in ('224', '32'):
                item = record['encoded'][size]
                if item['sha256'] in seen_files or item['pixel_sha256'] in seen_pixels:
                    duplicate = True
                seen_files.add(item['sha256'])
                seen_pixels.add(item['pixel_sha256'])
            if not duplicate:
                kept.append(row)
            if len(kept) == quota:
                break
        if len(kept) < quota:
            raise ValueError(f'Only {len(kept)} unique screened candidates for {label}; need {quota}')
        selected.extend(kept)
    if len({row['id'] for row in selected}) != len(selected):
        raise ValueError('Candidate IDs are not unique')
    return selected


def candidate_row(row):
    label = row['class']
    result = {
        'id': row['id'], 'class': label, 'split': 'train',
        'group_id': row['group_id'], 'source_dataset': row['dataset_ref'],
        'source_identity': row['source_identity'], 'source': row.get('source_name', ''),
        'source_url': row.get('source_url', row.get('download_url', '')),
        'landing_url': row.get('landing_url', ''), 'license': row.get('license', row.get('license_declared', '')),
        'author': row.get('author', ''), 'rights_verified': 'false',
        'photo_verified': 'false', 'physical_groups_verified': 'false',
        'review_status': 'technical_screen_only_rights_visual_review_pending',
        'original_path': row['original_path'],
    }
    encoded_root = ROOT / 'data/input_encoded_20261019'
    for size in (224, 32):
        source = encoded_root / f'Mid{size}' / label / (row['id'] + EXTENSIONS[size])
        if not source.is_file():
            raise FileNotFoundError(source)
        result[f'mid{size}_source'] = source
        result[f'mid{size}_sha256'] = sha(source)
        with Image.open(source) as image:
            image.load()
            if image.mode != 'RGB' or image.size != (size, size):
                raise ValueError(f'Candidate geometry invalid: {source}')
            result[f'mid{size}_pixel_sha256'] = hashlib.sha256(image.tobytes()).hexdigest()
    return result


def assert_no_duplicates(existing, selected):
    for size in (224, 32):
        file_hashes = {row[f'mid{size}_sha256'] for row in existing}
        pixel_hashes = {row[f'mid{size}_pixel_sha256'] for row in existing if row[f'mid{size}_pixel_sha256']}
        for row in selected:
            if row[f'mid{size}_sha256'] in file_hashes or (row[f'mid{size}_pixel_sha256'] and row[f'mid{size}_pixel_sha256'] in pixel_hashes):
                raise ValueError(f'Duplicate candidate for Mid{size}: {row["id"]}')
            file_hashes.add(row[f'mid{size}_sha256'])
            pixel_hashes.add(row[f'mid{size}_pixel_sha256'])


def promote(dry_run=False):
    existing = rows_from_current()
    selected = [candidate_row(row) for row in load_candidates()]
    assert_no_duplicates(existing, selected)
    all_rows = existing + selected
    counts = Counter((row['split'], row['class']) for row in all_rows)
    expected = {('train', label): 5000 for label in CLASSES} | {('test', label): 50 for label in CLASSES}
    if any(counts[key] != value for key, value in expected.items()):
        raise ValueError(f'Quota mismatch: {counts}')
    if dry_run:
        return {'selected': len(selected), 'counts': {f'{split}/{label}': counts[split, label]
                                                        for split, label in expected}, 'status': 'dry_run'}
    for row in selected:
        for size in (224, 32):
            destination = ROOT / 'DataNew' / f'Mid{size}' / 'train' / row['class'] / (row['id'] + EXTENSIONS[size])
            if destination.exists():
                raise FileExistsError(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row[f'mid{size}_source'], destination)
            if sha(destination) != row[f'mid{size}_sha256']:
                raise ValueError(f'Copy hash mismatch: {destination}')
            row[f'mid{size}_path'] = destination.relative_to(ROOT).as_posix()
            del row[f'mid{size}_source']
    metadata = ROOT / 'data/metadata'
    write_csv(metadata / 'splits.csv', all_rows, MANIFEST_FIELDS)
    write_json(metadata / 'build_settings.json', {
        'layout': 'paired-compressed-v1', 'source': 'DataNew',
        'dataset_limit_bytes': {'224': LIMITS[224], '32': LIMITS[32]},
        'capacity_unit': 'MB (1,000,000 bytes)', 'official_test_preserved': True,
        'source_rights_review_complete': False, 'visual_review_complete': False,
        'technical_quota_complete': True,
    })
    write_csv(ROOT / 'results/final_input_manifest.csv', all_rows, MANIFEST_FIELDS)
    write_json(ROOT / 'results/input_source_review.json', {
        'status': 'technical_quota_complete_rights_and_full_visual_review_pending',
        'counts': {split: {label: counts[split, label] for label in CLASSES} for split in ('train', 'test')},
        'added': {label: sum(row['class'] == label for row in selected) for label in ('banana', 'strawberry', 'pineapple')},
        'sources': {label: sorted({row['source_dataset'] for row in selected if row['class'] == label})
                    for label in ('banana', 'strawberry', 'pineapple')},
        'rights_verified': False, 'visual_review_complete': False,
        'note': 'Technical input quota and paired encoding are complete. Do not submit until rights, full labels, capture groups and near-duplicate review are complete.',
    })
    return {'selected': len(selected), 'counts': {f'{split}/{label}': counts[split, label]
                                                  for split, label in expected}, 'status': 'promoted'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    print(json.dumps(promote(args.dry_run), indent=2))


if __name__ == '__main__':
    main()
