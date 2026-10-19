import csv
import hashlib
import json

from PIL import Image
import pytest

from src import data, train
from src import input_dataset as inputs


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(inputs, 'CLASSES', ('apple',))
    monkeypatch.setattr(inputs, 'SPLIT_QUOTAS', {'train': 1, 'test': 1})
    rows = []
    for split, color in [('train', (180, 20, 30)), ('test', (20, 100, 180))]:
        row = {'id': 'apple_' + split, 'class': 'apple', 'split': split,
               'group_id': 'capture_' + split}
        for size, extension in [(224, '.avif'), (32, '.webp')]:
            path = tmp_path / 'DataNew' / f'Mid{size}' / split / 'apple' / (row['id'] + extension)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (size, size), color).save(path)
            row[f'mid{size}_path'] = path.relative_to(tmp_path).as_posix()
            row[f'mid{size}_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(row)
    metadata = tmp_path / 'data/metadata'
    metadata.mkdir(parents=True)
    (metadata / 'build_settings.json').write_text(json.dumps({'layout': 'paired-compressed-v1'}))
    with (metadata / 'splits.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path, rows


def test_compressed_bundle_is_checked_and_consumed_by_training(bundle, monkeypatch):
    root, rows = bundle
    report = inputs.audit(root)
    assert report['structural_pass']
    assert report['Mid224']['images'] == 2
    assert report['Mid32']['total_bytes'] <= 20_000_000
    monkeypatch.setattr(train, 'ROOT', root)
    dataset = train.ManifestDataset(32, 'train_pool', training=False)
    tensor, label = dataset[0]
    assert tuple(tensor.shape) == (3, 32, 32)
    assert label == 0
    monkeypatch.setattr(data, 'ROOT', root)
    monkeypatch.setattr(data, 'META', root / 'data/metadata')
    assert data.audit()['structural_pass']


def test_missing_pair_blocks_training_audit(bundle):
    root, rows = bundle
    (root / rows[0]['mid32_path']).unlink()
    with pytest.raises(ValueError, match='Missing'):
        inputs.audit(root)


def test_unrecorded_extra_image_is_rejected(bundle):
    root, rows = bundle
    extra = root / 'DataNew/Mid32/train/apple/extra.webp'
    extra.write_bytes((root / rows[0]['mid32_path']).read_bytes())
    with pytest.raises(ValueError, match='Unmanifested'):
        inputs.audit(root)


def test_shortage_is_not_reported_as_ready(bundle, monkeypatch):
    root, _ = bundle
    monkeypatch.setattr(inputs, 'SPLIT_QUOTAS', {'train': 2, 'test': 1})
    with pytest.raises(ValueError, match='quota'):
        inputs.audit(root)


def test_capacity_limit_is_enforced_on_real_files(bundle, monkeypatch):
    root, _ = bundle
    monkeypatch.setattr(inputs, 'LIMITS', {224: 1, 32: 20_000_000})
    with pytest.raises(ValueError, match='capacity'):
        inputs.audit(root)


def test_hash_tampering_is_rejected(bundle):
    root, rows = bundle
    with (root / rows[0]['mid224_path']).open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='hash'):
        inputs.audit(root)


def test_wrong_size_is_rejected(bundle):
    root, rows = bundle
    path = root / rows[0]['mid32_path']
    Image.new('RGB', (30, 32), 'green').save(path)
    with pytest.raises(ValueError, match='geometry'):
        inputs.audit(root)


def test_parent_groups_cannot_cross_splits(bundle):
    root, rows = bundle
    rows[1]['group_id'] = rows[0]['group_id']
    with pytest.raises(ValueError, match='group'):
        inputs.validate(root, rows)


def test_identical_decoded_images_are_not_counted_twice(bundle):
    root, rows = bundle
    source = root / rows[0]['mid32_path']
    target = root / rows[1]['mid32_path']
    target.write_bytes(source.read_bytes())
    rows[1]['mid32_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='Duplicate'):
        inputs.validate(root, rows)


def test_manifest_cannot_escape_the_dataset_directory(bundle):
    root, rows = bundle
    rows[0]['mid32_path'] = '../outside.webp'
    with pytest.raises(ValueError, match='Unsafe'):
        inputs.validate(root, rows)
