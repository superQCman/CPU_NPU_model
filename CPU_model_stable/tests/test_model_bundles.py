"""Verify that delivered models preserve the original trained pickle bytes."""

import hashlib
import json
import lzma
from pathlib import Path
import pickle
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibrator.static_feature_mapping.predict_from_config import _load_model_bundle


@pytest.mark.parametrize("hardware", ["s920x20", "kunpeng920"])
def test_delivered_model_has_original_payload(hardware):
    manifest = json.loads((ROOT / "models/manifest.json").read_text())
    entry = next(model for model in manifest["models"] if model["hardware"] == hardware)
    path = ROOT / entry["path"]
    assert path.stat().st_size == entry["size_bytes"]
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]
    digest = hashlib.sha256()
    size = 0
    with tempfile.TemporaryFile() as compressed:
        if entry["compression"] == "xz_parts":
            parts = json.loads(path.read_text())["parts"]
            for part in parts:
                part_path = path.with_name(part["filename"])
                assert part_path.stat().st_size == part["size_bytes"]
                with part_path.open("rb") as source:
                    assert hashlib.file_digest(source, "sha256").hexdigest() == part["sha256"]
                    source.seek(0)
                    while block := source.read(1024 * 1024):
                        compressed.write(block)
        else:
            with path.open("rb") as source:
                while block := source.read(1024 * 1024):
                    compressed.write(block)
        compressed.seek(0)
        with lzma.open(compressed, "rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
                size += len(block)
    assert size == entry["original_size_bytes"]
    assert digest.hexdigest() == entry["original_sha256"]


def test_raw_and_xz_bundle_loading_are_compatible(tmp_path):
    bundle = {"route": "test", "prior_bundle": {"scales": [1.0, 2.0]}, "model_sha256": "original"}
    payload = pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
    raw = tmp_path / "bundle.pkl"
    compressed = tmp_path / "bundle.pkl.xz"
    raw.write_bytes(payload)
    compressed.write_bytes(lzma.compress(payload))
    assert _load_model_bundle(raw) == _load_model_bundle(compressed) == bundle


def test_split_xz_bundle_loading_preserves_model(tmp_path):
    bundle = {"route": "test", "prior_bundle": {"scales": [1.0, 2.0]}}
    payload = lzma.compress(pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL))
    midpoint = len(payload) // 2
    (tmp_path / "part01").write_bytes(payload[:midpoint])
    (tmp_path / "part02").write_bytes(payload[midpoint:])
    manifest = tmp_path / "bundle.pkl.xz.parts.json"
    manifest.write_text(json.dumps({"format": "xz_parts_v1", "parts": [{"filename": "part01"}, {"filename": "part02"}]}))
    assert _load_model_bundle(manifest) == bundle
