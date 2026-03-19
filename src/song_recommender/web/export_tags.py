from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "tags.json"
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


def main() -> None:
    mapping: dict[str, list[str]] = {}

    for path in INPUT_PATHS:
        frame = pd.read_parquet(path, columns=["spotify_id", "tag_list"])
        for spotify_id, tag_list in frame.itertuples(index=False):
            tags = normalize_tags(tag_list)
            if tags:
                mapping[str(spotify_id)] = tags

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    print(f"wrote {len(mapping)} tag rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
