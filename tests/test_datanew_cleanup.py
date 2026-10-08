import pytest
from PIL import Image

from src import datanew_cleanup as clean


def item(stem, split='train', kind='real', hash224=None, hash32=None, visual=None):
    return {'key': [split, 'banana', stem],
            'row': {'split': split, 'class': 'banana', 'type': kind,
                    'source': stem + '.jpg', 'output': stem + '.jpg'},
            'images': {str(s): {'sha256': (hash224 if s == 224 else hash32) or stem + str(s),
                               'pixel_sha256': (hash224 if s == 224 else hash32) or stem + str(s),
                               'bytes': 100, 'visual': visual or ['0123456789abcdef'] * 4}
                       for s in (224, 32)}}


def test_augments_are_excluded_even_when_unique():
    kept, removed = clean.choose_retained([item('one', kind='augmented')])
    assert not kept
    assert removed[0]['reason'] == 'manifest_augmented'


@pytest.mark.parametrize('size', [224, 32])
def test_duplicate_in_either_size_removes_the_whole_train_pair(size):
    kwargs = {'hash' + str(size): 'same'}
    kept, removed = clean.choose_retained([item('train', **kwargs), item('test', split='test', **kwargs)])
    assert [i['key'][0] for i in kept] == ['test']
    assert removed[0]['key'][0] == 'train'
    assert removed[0]['reason'] == 'exact_content_or_source_duplicate'


def test_existing_similarity_is_flagged_not_deleted_without_proof():
    flags = []
    kept, removed = clean.choose_retained([item('one'), item('two')], flags)
    assert len(kept) == 2
    assert not removed
    assert len(flags) == 1
    assert flags[0]['status'] == 'unresolved_not_deleted_by_dhash_alone'


def test_seven_chunk_index_covers_six_changed_bits():
    index = clean.NearIndex()
    index.add('source', ['0123456789abcdef'] * 4)
    for mask in [0, 63, (1 << 63) | (1 << 54) | (1 << 45) | (1 << 36) | (1 << 27) | (1 << 18)]:
        value = int('0123456789abcdef', 16) ^ mask
        assert index.matches([f'{value:016x}']) == ['source']
    assert not index.matches(['fedcba9876543210'])


def test_encoding_preserves_pair_shapes_and_formats(tmp_path):
    images = clean.encode_pairs({s: Image.new('RGB', (s, s), (25, 90, 160)) for s in (224, 32)},
                                tmp_path, ('train', 'banana', 'new'))
    assert set(images) == {'224', '32'}
    with pytest.raises(FileExistsError):
        clean.encode_pairs({s: Image.new('RGB', (s, s)) for s in (224, 32)},
                           tmp_path, ('train', 'banana', 'new'))


def test_partial_dataset_is_not_quota_pass():
    report = clean.audit_items([item('one')])
    assert not report['quota_pass']
    assert report['shortage']['train']['banana'] == 4999
    assert report['capacity_pass']


def test_backup_detects_modified_copy(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir(); b.mkdir()
    (a / 'image').write_bytes(b'one')
    (b / 'image').write_bytes(b'two')
    with pytest.raises(ValueError, match='Backup'):
        clean.verify_copy(a, b)