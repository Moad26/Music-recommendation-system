import numpy as np
import pandas as pd
import pytest

REQUIRED_COLUMNS = [
    "track_id",
    "name",
    "album",
    "artists",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "key",
    "mode",
]


@pytest.fixture
def synthetic_audio() -> tuple[np.ndarray, float]:
    sr = 22050
    duration = 35  # to test if the loader really truncate to 30s

    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t)
    return audio, sr


@pytest.fixture
def fake_spotify_data():
    return pd.DataFrame({col: [0.5] * 2 for col in REQUIRED_COLUMNS}).assign(
        track_id=["t1", "t2"],
        name=["Song A", "Song B"],
        artists=["['Artist A']", "['Artist B']"],
        tempo=[120.0, 100.0],
        key=[1, 2],
        mode=[1, 0],
    )
