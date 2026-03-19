from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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


def main() -> None:
    catalog: dict[str, dict[str, object]] = {}

    for path in INPUT_PATHS:
        split = path.stem
        frame = pd.read_parquet(path, columns=["spotify_id", "name", "artist", "tag_list"])
        for spotify_id, name, artist, tag_list in frame.itertuples(index=False):
            catalog[str(spotify_id)] = {
                "name": str(name),
                "artist": str(artist),
                "tags": normalize_tags(tag_list),
                "split": split,
            }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(catalog, indent=2, sort_keys=True))
    print(f"wrote {len(catalog)} catalog rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
