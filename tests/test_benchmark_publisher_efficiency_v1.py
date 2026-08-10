from pathlib import Path

from scripts.evaluate.benchmark_publisher_efficiency_v1 import artifact_bytes


def test_artifact_bytes_counts_nested_files(tmp_path: Path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "one.bin").write_bytes(b"123")
    (tmp_path / "nested" / "two.bin").write_bytes(b"4567")
    assert artifact_bytes(tmp_path) == 7
