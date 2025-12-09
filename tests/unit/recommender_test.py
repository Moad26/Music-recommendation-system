from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.music_recommender.models.recommender import MusicRecommender


@patch("src.music_recommender.models.recommender.joblib.load")
@patch("src.music_recommender.models.recommender.pd.read_csv")
def test_recommender_end_to_end(mock_read_csv, mock_joblib, fake_spotify_data):
    mock_read_csv.return_value = fake_spotify_data
    fake_model = MagicMock()
    feature_cols = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "key",
        "mode",
        "tempo_bins",
    ]
    fake_prediction = pd.DataFrame(
        np.zeros((1, len(feature_cols))), columns=feature_cols
    )
    fake_model.predict.return_value = fake_prediction
    mock_joblib.return_value = fake_model
    recommender = MusicRecommender(
        hybrid_model_path=Path("dummy_model.joblib"),
        spotify_dataset_path=Path("dummy_data.csv"),
    )
    recommendations = recommender.get_recommendations(
        features=fake_prediction, top_n=1, features_type="predicted"
    )

    assert isinstance(recommendations, pd.DataFrame)
    assert len(recommendations) == 1
    assert "similarity_score" in recommendations.columns
    assert recommendations.iloc[0]["name"] in ["Song A", "Song B"]
