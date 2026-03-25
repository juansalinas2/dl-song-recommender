from __future__ import annotations

from song_recommender.web.audio_query import _load_encoder, _load_separator
from song_recommender.web.evaluation_store import ensure_schema
from song_recommender.web.recommender import available_models


def main() -> None:
    ensure_schema()
    separator = _load_separator()
    print(f"render predeploy: separator ready at {separator.samplerate} Hz")

    ready_models = []
    for spec in available_models():
        if not spec.supports_uploaded_audio:
            continue
        _load_encoder(spec)
        ready_models.append(spec.model_id)
    print(f"render predeploy: upload-ready models {ready_models}")


if __name__ == "__main__":
    main()
