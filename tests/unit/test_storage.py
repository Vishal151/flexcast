from pathlib import Path

import pandas as pd
import pytest

from flexcast.storage import Storage


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    return Storage(f"file://{tmp_path}")


def test_parquet_round_trip_keeps_tz_aware_utc(storage: Storage) -> None:
    df = pd.DataFrame(
        {"start_time": pd.to_datetime(["2025-06-10T23:00Z", "2025-06-10T23:30Z"]), "v": [1.5, -2.0]}
    )
    storage.write_parquet(df, "raw/x/date=2025-06-11/part.parquet")
    out = storage.read_parquet("raw/x/date=2025-06-11/part.parquet")
    pd.testing.assert_frame_equal(out, df)
    assert str(out["start_time"].dt.tz) == "UTC"


def test_write_is_deterministic(storage: Storage, tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [3, 1, 2]})
    storage.write_parquet(df, "a.parquet")
    first = (tmp_path / "a.parquet").read_bytes()
    storage.write_parquet(df, "a.parquet")
    assert (tmp_path / "a.parquet").read_bytes() == first


def test_write_leaves_no_temp_files(storage: Storage, tmp_path: Path) -> None:
    storage.write_parquet(pd.DataFrame({"a": [1]}), "d/a.parquet")
    assert [p.name for p in (tmp_path / "d").iterdir()] == ["a.parquet"]


def test_exists_and_list(storage: Storage) -> None:
    assert not storage.exists("raw/x/date=2025-06-11/part.parquet")
    for d in ("2025-06-12", "2025-06-11"):
        storage.write_parquet(pd.DataFrame({"a": [1]}), f"raw/x/date={d}/part.parquet")
    assert storage.exists("raw/x/date=2025-06-11/part.parquet")
    assert storage.list("raw/x") == [
        "raw/x/date=2025-06-11/part.parquet",
        "raw/x/date=2025-06-12/part.parquet",
    ]
    assert storage.list("raw/missing") == []


def test_text_round_trip(storage: Storage) -> None:
    storage.write_text("hello\n", "rejects/x/a.jsonl")
    assert storage.read_text("rejects/x/a.jsonl") == "hello\n"


def test_relative_file_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    Storage("file://./data").write_text("x", "a.txt")
    assert (tmp_path / "data" / "a.txt").read_text() == "x"


def test_read_parquet_dir_concatenates_partitions(storage: Storage) -> None:
    for i, d in enumerate(("2025-06-11", "2025-06-12")):
        storage.write_parquet(pd.DataFrame({"a": [i]}), f"raw/x/date={d}/part.parquet")
    out = storage.read_parquet_dir("raw/x")
    assert sorted(out["a"]) == [0, 1]
    assert list(out.columns) == ["a"]
    assert storage.read_parquet_dir("raw/missing").empty
