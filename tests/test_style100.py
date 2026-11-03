"""``style100._fetch`` takes the BVHs it is asked for out of the Zenodo archive, only once
the archive matches the checksum the record lists, and gives every request ``TIMEOUT``."""

import hashlib
import io
import json
import zipfile

import pytest

from mjswan_playground.msm import style100

CONTENT = "https://zenodo.org/api/records/8127870/files/100STYLE.zip/content"


def _archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("100STYLE/Neutral/Neutral_FW.bvh", "HIERARCHY")
        bundle.writestr("100STYLE/Old/Old_FW.bvh", "unwanted")
        bundle.writestr("__MACOSX/100STYLE/Neutral/._Neutral_FW.bvh", "resource fork")
    return buffer.getvalue()


def _zenodo(monkeypatch, archive: bytes, md5: str) -> list[float]:
    record = {
        "files": [
            {
                "key": style100.ARCHIVE,
                "size": len(archive),
                "checksum": f"md5:{md5}",
                "links": {"self": CONTENT},
            }
        ]
    }
    timeouts = []

    def urlopen(url, timeout=None):
        timeouts.append(timeout)
        if url == f"https://zenodo.org/api/records/{style100.ZENODO_RECORD}":
            return io.BytesIO(json.dumps(record).encode())
        assert url == CONTENT
        return io.BytesIO(archive)

    monkeypatch.setattr(style100.urllib.request, "urlopen", urlopen)
    return timeouts


def test_fetch_extracts_what_is_wanted(tmp_path, monkeypatch):
    archive = _archive()
    timeouts = _zenodo(monkeypatch, archive, hashlib.md5(archive).hexdigest())

    style100._fetch(tmp_path, {"Neutral/Neutral_FW.bvh"})

    assert (tmp_path / "Neutral" / "Neutral_FW.bvh").read_text() == "HIERARCHY"
    assert not (tmp_path / "Old").exists()
    assert timeouts == [style100.TIMEOUT, style100.TIMEOUT]


def test_fetch_refuses_an_archive_that_does_not_match(tmp_path, monkeypatch):
    _zenodo(monkeypatch, _archive(), "0" * 32)

    with pytest.raises(RuntimeError, match="md5"):
        style100._fetch(tmp_path, {"Neutral/Neutral_FW.bvh"})
    assert not any(tmp_path.iterdir())
