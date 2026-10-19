import os

import pytest

from src import input_audit_cache as cache
from src import input_dataset as inputs
from test_input_dataset import bundle


def test_valid_cache_rechecks_bytes_without_decoding(bundle, monkeypatch):
    root, rows = bundle
    target = root / 'cache/audit.json'
    first, reused = cache.validate_inputs(root, rows, target)
    assert first['structural_pass'] and not reused

    def no_decode(*args, **kwargs):
        pytest.fail('Unchanged images must not be decoded again')

    monkeypatch.setattr(inputs.Image, 'open', no_decode)
    second, reused = cache.validate_inputs(root, rows, target)
    assert reused and second == first


def test_cached_hash_check_detects_same_length_change_with_restored_mtime(bundle):
    root, rows = bundle
    target = root / 'cache/audit.json'
    cache.validate_inputs(root, rows, target)
    path = root / rows[0]['mid224_path']
    original = path.stat()
    contents = bytearray(path.read_bytes())
    contents[-1] ^= 1
    path.write_bytes(contents)
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(ValueError, match='hash'):
        cache.validate_inputs(root, rows, target)


@pytest.mark.parametrize('change', ['missing', 'extra'])
def test_cached_check_rejects_changed_file_inventory(bundle, change):
    root, rows = bundle
    target = root / 'cache/audit.json'
    cache.validate_inputs(root, rows, target)
    if change == 'missing':
        (root / rows[0]['mid32_path']).unlink()
    else:
        (root / 'DataNew/Mid32/extra.webp').write_bytes(b'extra')
    with pytest.raises(ValueError, match='inventory'):
        cache.validate_inputs(root, rows, target)


def test_changed_policy_cannot_reuse_old_quota_pass(bundle, monkeypatch):
    root, rows = bundle
    target = root / 'cache/audit.json'
    cache.validate_inputs(root, rows, target)
    monkeypatch.setattr(inputs, 'SPLIT_QUOTAS', {'train': 2, 'test': 1})
    with pytest.raises(ValueError, match='quota'):
        cache.validate_inputs(root, rows, target)


def test_changed_manifest_cannot_reuse_old_group_pass(bundle):
    root, rows = bundle
    target = root / 'cache/audit.json'
    cache.validate_inputs(root, rows, target)
    rows[1]['group_id'] = rows[0]['group_id']
    with pytest.raises(ValueError, match='group'):
        cache.validate_inputs(root, rows, target)


@pytest.mark.parametrize('broken', ['{', '{"schema": 1}', '{"schema": 999}'])
def test_invalid_cache_falls_back_to_full_validation(bundle, broken):
    root, rows = bundle
    target = root / 'cache/audit.json'
    target.parent.mkdir(parents=True)
    target.write_text(broken)
    report, reused = cache.validate_inputs(root, rows, target)
    assert report['structural_pass'] and not reused


def test_forced_audit_decodes_again(bundle, monkeypatch):
    root, rows = bundle
    target = root / 'cache/audit.json'
    cache.validate_inputs(root, rows, target)
    opened = []
    original = inputs.Image.open

    def track(*args, **kwargs):
        opened.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(inputs.Image, 'open', track)
    report, reused = cache.validate_inputs(root, rows, target, force_full=True)
    assert report['structural_pass'] and not reused and len(opened) == 4


def test_failed_full_audit_never_creates_reusable_cache(bundle):
    root, rows = bundle
    target = root / 'cache/audit.json'
    (root / rows[0]['mid32_path']).unlink()
    with pytest.raises(ValueError):
        cache.validate_inputs(root, rows, target)
    assert not target.exists()
