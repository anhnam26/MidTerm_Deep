"""Isolated replenishment screening; metadata and hashes never imply approval.

Keep existing datasets, journals, reservations and test files read-only. Outputs
are original photographs awaiting review, not quota-approved training images.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

from PIL import Image

from .common import CLASSES, ROOT, write_json
from .datanew_cleanup import NearIndex, metadata_baseline, visual_hashes
from .dataset_acquire import MAX_FILE_BYTES, atomic_bytes, prepare


def source_keys(row):
    keys = set()
    for field in ('id', 'source_identity', 'source_url', 'landing_url', 'download_url'):
        value = str(row.get(field) or '').rstrip('/')
        if value:
            keys.add(value)
        match = re.search(r'/photos/[^/]+/(\d+)', value)
        if not match:
            match = re.search(r'/(\d+)_[^/]+\.(?:jpg|jpeg)', value, re.I)
        if match:
            keys.add('flickr:' + match[1])
        if 'commons.wikimedia.org/wiki/File:' in value:
            keys.add('commons:' + unquote(value.split('/wiki/', 1)[1]).replace('_', ' '))
    return keys


def commons_candidates(payload, label):
    if label not in CLASSES:
        raise ValueError('Unknown class')
    result = []
    for page in payload.get('query', {}).get('pages', {}).values():
        info = page.get('imageinfo', [{}])[0]
        ext = info.get('extmetadata', {})
        value = lambda field: ext.get(field, {}).get('value', '')
        url = info.get('url', '')
        license_url = value('LicenseUrl')
        # Individual file license + author/source are required, not category license.
        if (page.get('ns') != 6 or not permitted_original_url(url)
                or not value('Artist') or not value('Credit')
                or not re.match(r'^https?://creativecommons.org/(?:licenses/|publicdomain/)', license_url)
                or min(info.get('width', 0), info.get('height', 0)) < 224
                or info.get('width', 0) * info.get('height', 0) > 40_000_000
                or max(info.get('width', 0), info.get('height', 0)) > 8000
                or not 0 < info.get('size', 0) <= MAX_FILE_BYTES):
            continue
        title = page['title']
        result.append({'id': 'commons_' + str(page['pageid']), 'class': label,
                       'source_identity': 'commons:' + title.replace('_', ' '),
                       'source_name': title, 'download_url': url,
                       'landing_url': info.get('descriptionurl', ''),
                       'license_declared': license_url, 'author': value('Artist'),
                       'credit': value('Credit'), 'description': value('ImageDescription'),
                       'capture_time': value('DateTimeOriginal'),
                       'photo_verified': False, 'rights_verified': False,
                       'physical_groups_verified': False, 'provider': 'commons'})
    return result


def permitted_original_url(url):
    parsed = urlsplit(url)
    return (parsed.scheme == 'https' and parsed.hostname == 'upload.wikimedia.org'
            and parsed.netloc == 'upload.wikimedia.org'
            and parsed.path.startswith('/wikipedia/commons/')
            and '/thumb/' not in parsed.path
            and Path(parsed.path).suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})


def fingerprint(blob):
    with Image.open(io.BytesIO(blob)) as opened:
        native, pairs = prepare(opened)
    return {'original_sha256': hashlib.sha256(blob).hexdigest(),
            'native_pixel_sha256': hashlib.sha256(str(native.size).encode() + native.tobytes()).hexdigest(),
            'pixel224_sha256': hashlib.sha256(pairs[224].tobytes()).hexdigest(),
            'native_size': list(native.size), 'visual': visual_hashes(native)}


class ScreeningIndex:
    def __init__(self, rows=(), final_items=()):
        self.sources = defaultdict(set)
        self.hashes = defaultdict(set)
        self.near = NearIndex()
        for row in rows:
            self.add(row)
        for item in final_items:
            identifier = 'final:' + '/'.join(item['key'])
            self.sources[item['row']['source']].add(identifier)
            for field in ('sha256', 'pixel_sha256'):
                self.hashes[item['images']['224'][field]].add(identifier)
            self.near.add(identifier, item['images']['224']['visual'])

    def add(self, row):
        identifier = row['id']
        for value in source_keys(row):
            self.sources[value].add(identifier)
        for field in ('original_sha256', 'native_pixel_sha256', 'pixel224_sha256'):
            if row.get(field):
                self.hashes[row[field]].add(identifier)
        hashes = row.get('visual')
        if not hashes and row.get('dhash'):
            hashes = [row['dhash']] * 3 + [row.get('center_crop_dhash', row['dhash'])]
        if hashes:
            self.near.add(identifier, hashes)

    def matches(self, row):
        exact = set()
        for value in source_keys(row):
            exact.update(self.sources.get(value, ()))
        for field in ('original_sha256', 'native_pixel_sha256', 'pixel224_sha256'):
            exact.update(self.hashes.get(row.get(field), ()))
        near = set(self.near.matches(row['visual'])) if row.get('visual') else set()
        return sorted(exact), sorted(near - exact)


def download_original(url):
    if not permitted_original_url(url):
        raise ValueError('Not a permitted original photograph URL')
    # Isolation caps slow trickle as well as normal read timeouts, as in session14.
    code = '''import requests,sys,time
start=time.monotonic()
with requests.get(sys.argv[1],stream=True,timeout=(10,15),allow_redirects=False,
 headers={'User-Agent':'FruitDatasetResearch/1.0 (educational provenance research)'}) as r:
 if r.status_code in (401,403,429):
  print('ACCESS_BLOCK_'+str(r.status_code),file=sys.stderr);sys.exit(3)
 r.raise_for_status()
 if 300<=r.status_code<400:raise ValueError('Unexpected redirect')
 if int(r.headers.get('Content-Length','0'))>12000000:raise ValueError('Size cap')
 data=[];size=0
 for chunk in r.iter_content(65536):
  size+=len(chunk)
  if size>12000000 or time.monotonic()-start>45:raise ValueError('Size/time cap')
  data.append(chunk)
 sys.stdout.buffer.write(b''.join(data))
'''
    result = subprocess.run([sys.executable, '-B', '-c', code, url], timeout=60,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        message = result.stderr.decode('utf-8', errors='replace')[-1500:]
        if 'ACCESS_BLOCK_' in message:
            raise PermissionError(message)
        raise ValueError(message)
    if not 0 < len(result.stdout) <= MAX_FILE_BYTES:
        raise ValueError('Invalid downloaded size')
    return result.stdout


def stage(candidates, index, out, fetch=download_original):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / 'screening.json'
    decisions = json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else []
    done = {row['id'] for row in decisions}
    for row in decisions:
        if row.get('original_path'):
            blob = (out / row['original_path']).read_bytes()
            if hashlib.sha256(blob).hexdigest() != row['original_sha256']:
                raise ValueError('Changed/missing staged original')
        index.add(row)
    if len({row['id'] for row in candidates}) != len(candidates):
        raise ValueError('Duplicate candidate IDs')
    for candidate in candidates:
        if candidate['id'] in done:
            continue
        if not re.fullmatch(r'[a-z0-9_]+', candidate['id']) or candidate['class'] not in CLASSES:
            raise ValueError('Unsafe candidate identity/class')
        row = dict(candidate, status='failed')
        exact, _ = index.matches(row)
        try:
            if exact:
                row.update(status='duplicate_identity', exact_matches=exact)
            else:
                blob = fetch(row['download_url'])
                row.update(fingerprint(blob))
                path = Path('originals') / row['class'] / (row['id'] + Path(urlsplit(row['download_url']).path).suffix.lower())
                atomic_bytes(out / path, blob)
                row['original_path'] = path.as_posix()
                exact, near = index.matches(row)
                row.update(exact_matches=exact, near_matches=near,
                           status='duplicate_exact' if exact else 'near_unresolved' if near else 'needs_review')
                index.add(row)
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            row['error'] = str(error)
            if isinstance(error, PermissionError):
                row['status'] = 'access_blocked'
        decisions.append(row)
        write_json(checkpoint, decisions)
        print('SCREENED', len(decisions), row['id'], row['status'], flush=True)
        if row['status'] == 'access_blocked':
            break
    write_json(out / 'summary.json', {'statuses': dict(Counter(row['status'] for row in decisions)),
                                     'accepted': 0, 'official_export': False,
                                     'review_complete': False, 'candidate_count': len(candidates)})
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--baseline-inventory', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    protected = metadata_baseline(ROOT)
    history = json.loads(args.history.read_text(encoding='utf-8'))['rows']
    journal = ROOT / 'data/metadata/fruit_acquisition_08/attempts.jsonl'
    history.extend(json.loads(line) for line in journal.read_bytes().splitlines())
    baseline = json.loads(args.baseline_inventory.read_text(encoding='utf-8'))
    index = ScreeningIndex(history, baseline)
    stage(json.loads(args.candidates.read_text(encoding='utf-8')), index, args.out)
    if metadata_baseline(ROOT) != protected:
        raise ValueError('Protected metadata changed during isolated screening')


if __name__ == '__main__':
    main()