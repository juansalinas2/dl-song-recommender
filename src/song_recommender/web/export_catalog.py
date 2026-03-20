from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "catalog.json"
INPUT_PATHS = (
    ROOT / "data" / "processed" / "train.parquet",
    ROOT / "data" / "processed" / "val.parquet",
    ROOT / "data" / "processed" / "test.parquet",
)


def normalize_tags(value) -> list[str]:
    if isinstance(value, list):
        return [str(tag).strip() for tag in value if str(tag).strip()]
    if isinstance(value, str):
        return [tag.strip() for tag in value.split(",") if tag.strip()]
    return []


def _read_catalog_frame_pyarrow(path: Path) -> pd.DataFrame:
    schema_names = set(pq.ParquetFile(path).schema.names)
    tag_column = "tags" if "tags" in schema_names else "tag_list" if "tag_list" in schema_names else None
    columns = ["spotify_id", "name", "artist"]
    if tag_column:
        columns.append(tag_column)
    frame = pq.read_table(path, columns=columns).to_pandas()
    if tag_column and tag_column != "tags":
        frame = frame.rename(columns={tag_column: "tags"})
    if "tags" not in frame.columns:
        frame["tags"] = [[] for _ in range(len(frame))]
    return frame


def _read_catalog_frame_duckdb(path: Path) -> pd.DataFrame:
    import duckdb

    schema_names = {row[0] for row in duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}')").fetchall()}
    tag_column = "tags" if "tags" in schema_names else "tag_list" if "tag_list" in schema_names else None
    selected = ["spotify_id", "name", "artist"]
    if tag_column:
        selected.append(tag_column)

    rows = duckdb.sql(
        f"SELECT {', '.join(selected)} FROM read_parquet('{path.as_posix()}')"
    ).fetchall()
    columns = ["spotify_id", "name", "artist", "tags"] if tag_column else ["spotify_id", "name", "artist"]
    frame = pd.DataFrame(rows, columns=columns)
    if "tags" not in frame.columns:
        frame["tags"] = [[] for _ in range(len(frame))]
    return frame


def read_catalog_frame(path: Path) -> pd.DataFrame:
    try:
        return _read_catalog_frame_pyarrow(path)
    except OSError:
        return _read_catalog_frame_duckdb(path)


def main() -> None:
    catalog: dict[str, dict[str, object]] = {}

    for path in INPUT_PATHS:
        split = path.stem
        frame = read_catalog_frame(path)
        for spotify_id, name, artist, tags in frame.itertuples(index=False):
            catalog[str(spotify_id)] = {
                "name": str(name),
                "artist": str(artist),
                "tags": normalize_tags(tags),
                "split": split,
            }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(catalog, indent=2, sort_keys=True))
    print(f"wrote {len(catalog)} catalog rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
