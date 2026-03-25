from song_recommender.web.recommender import (
    RecommenderIndex,
    available_models,
    metadata_lookup,
    resolve_model,
    split_lookup,
    tags_lookup,
)


def main() -> None:
    models = available_models()
    assert models, "expected at least one available model"
    assert models[0].model_id == "04_resnet18_contrastive_tags", "expected ResNet04 to be listed first"
    baseline = next((model for model in models if model.model_id == "baseline"), None)
    assert baseline is not None, "expected baseline model to be present"
    assert not baseline.supports_uploaded_audio, "expected baseline upload support to stay disabled"

    index = RecommenderIndex(
        resolve_model(None),
        metadata_lookup=metadata_lookup(),
        tags_lookup=tags_lookup(),
        split_lookup=split_lookup(),
    )
    sample = index.random(1)
    assert sample, "expected at least one sample track"
    payload = index.recommend(sample[0]["spotify_id"], 3)
    assert len(payload["recommendations"]) == 3, "expected exactly three recommendations"
    assert payload["model"]["model_id"] == index.spec.model_id, "expected default model payload"

    if index.queryable_count("evaluation"):
        evaluation_sample = index.random(1, mode="evaluation")
        assert evaluation_sample, "expected at least one held-out evaluation sample"
        evaluation_payload = index.recommend(evaluation_sample[0]["spotify_id"], 3, mode="evaluation")
        assert evaluation_payload["mode"] == "evaluation", "expected evaluation mode payload"
        assert evaluation_payload["query"]["split"] == index.evaluation_split, "expected held-out evaluation split"
    manifold = index.manifold_projection(sample[0]["spotify_id"], sample_limit=60, neighbor_limit=12)
    assert manifold["points"], "expected manifold projection points"
    stats = index.global_stats(pair_limit=5, song_limit=5)
    assert stats["closest_pairs"], "expected closest-pair stats"
    print("smoke test passed")


if __name__ == "__main__":
    main()
