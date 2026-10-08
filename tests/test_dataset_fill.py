import hashlib
import io
import json

from PIL import Image
import pytest

from src import dataset_fill as fill


def blob(size=(300, 240)):
    stream = io.BytesIO()
    Image.new('RGB', size, 'red').save(stream, format='JPEG')
    return stream.getvalue()


def candidate(identifier='commons_1'):
    return {'id': identifier, 'class': 'pineapple', 'source_identity': identifier,
            'download_url': 'https://upload.wikimedia.org/wikipedia/commons/a/ab/Pineapple.jpg'}


def test_flickr_alias_survives_provider_and_username_changes():
    index = fill.ScreeningIndex([{'id': 'old', 'landing_url': 'https://www.flickr.com/photos/one/123'}])
    exact, near = index.matches({'id': 'new', 'source_url': 'https://www.flickr.com/photos/two/123/'})
    assert exact == ['old'] and not near


@pytest.mark.parametrize('url', [
    'http://upload.wikimedia.org/wikipedia/commons/a/ab/a.jpg',
    'https://upload.wikimedia.org.evil.invalid/wikipedia/commons/a/ab/a.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/a.jpg/500px-a.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/a/ab/a.svg',
    'https://user@upload.wikimedia.org/wikipedia/commons/a/ab/a.jpg'])
def test_only_original_public_photos_are_permitted(url):
    assert not fill.permitted_original_url(url)


def test_per_file_rights_required_but_not_automatically_approved():
    info = {'url': candidate()['download_url'], 'width': 300, 'height': 240, 'size': 2000,
            'extmetadata': {k: {'value': v} for k, v in {
                'Artist': 'Author', 'Credit': 'Own work',
                'LicenseUrl': 'https://creativecommons.org/licenses/by-sa/4.0/'}.items()}}
    page = {'ns': 6, 'pageid': 1, 'title': 'File:Pineapple.jpg', 'imageinfo': [info]}
    payload = {'query': {'pages': {'1': page}}}
    rows = fill.commons_candidates(payload, 'pineapple')
    assert len(rows) == 1
    assert not rows[0]['photo_verified'] and not rows[0]['rights_verified']
    assert not rows[0]['physical_groups_verified']
    del info['extmetadata']['LicenseUrl']
    assert not fill.commons_candidates(payload, 'pineapple')


def test_small_original_cannot_be_upscaled():
    with pytest.raises(ValueError, match='upscaling'):
        fill.fingerprint(blob((223, 300)))


def test_exact_and_near_are_distinct_statuses():
    old = dict(id='old', **fill.fingerprint(blob()))
    exact, near = fill.ScreeningIndex([old]).matches(dict(candidate(), **fill.fingerprint(blob())))
    assert exact == ['old'] and not near
    old.pop('original_sha256'); old.pop('native_pixel_sha256'); old.pop('pixel224_sha256')
    exact, near = fill.ScreeningIndex([old]).matches(dict(candidate(), **fill.fingerprint(blob())))
    assert not exact and near == ['old']


def test_resume_hash_checks_and_never_auto_accepts(tmp_path):
    calls = []
    def fetch(url):
        calls.append(url)
        return blob()
    result = fill.stage([candidate()], fill.ScreeningIndex(), tmp_path, fetch)
    assert result[0]['status'] == 'needs_review' and len(calls) == 1
    fill.stage([candidate()], fill.ScreeningIndex(), tmp_path, fetch)
    assert len(calls) == 1
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['accepted'] == 0 and not summary['official_export']
    (tmp_path / result[0]['original_path']).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Changed/missing'):
        fill.stage([candidate()], fill.ScreeningIndex(), tmp_path, fetch)


def test_access_block_stops_without_trying_next_candidate(tmp_path):
    calls = []
    def fetch(url):
        calls.append(url)
        raise PermissionError('ACCESS_BLOCK_429')
    result = fill.stage([candidate(), candidate('commons_2')], fill.ScreeningIndex(), tmp_path, fetch)
    assert len(result) == len(calls) == 1 and result[0]['status'] == 'access_blocked'


def test_history_duplicate_is_not_downloaded(tmp_path):
    def fetch(url):
        raise AssertionError('Duplicate must not be downloaded')
    result = fill.stage([candidate()], fill.ScreeningIndex([candidate()]), tmp_path, fetch)
    assert result[0]['status'] == 'duplicate_identity'


def test_original_hash_matches_staged_bytes(tmp_path):
    result = fill.stage([candidate()], fill.ScreeningIndex(), tmp_path, lambda _: blob())
    assert result[0]['original_sha256'] == hashlib.sha256((tmp_path / result[0]['original_path']).read_bytes()).hexdigest()