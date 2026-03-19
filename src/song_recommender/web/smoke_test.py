from song_recommender.web.recommender import ROOT, RecommenderIndex, available_models


def main() -> None:
    models = available_models()
    assert models, "expected at least one available model"

    model = next((candidate for candidate in models if candidate.model_id == "baseline"), models[0])
    index = RecommenderIndex(model)
    sample = index.random(1)
    assert sample, "expected at least one sample track"
    payload = index.recommend(sample[0]["spotify_id"], 3)
    assert len(payload["recommendations"]) == 3, "expected exactly three recommendations"
    manifold = index.manifold_projection(sample[0]["spotify_id"], sample_limit=60, neighbor_limit=12)
    assert manifold["points"], "expected manifold projection points"
    stats = index.global_stats(pair_limit=5, song_limit=5)
    assert stats["closest_pairs"], "expected closest-pair stats"
    print("smoke test passed")


if __name__ == "__main__":
    main()
