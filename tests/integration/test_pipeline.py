from unittest.mock import patch

import pandas as pd

from src.music_recommender.config import Config
from src.music_recommender.data.pipeline import create_extraction_pipeline


@patch("src.music_recommender.data.loaders.librosa.load")
def test_full_pipeline_integration(mock_load, synthetic_audio):
    audio, sr = synthetic_audio
    mock_load.return_value = (audio, sr)
    cfg = Config()
    pipeline = create_extraction_pipeline(cfg, save_cache=False, enable_cache=False)
    output = pipeline.fit_transform(["fake_song.mp3"])
    assert isinstance(output, pd.DataFrame)
    assert not output.empty
    assert "tempo" in output.columns
    assert "mfcc_0_mean" in output.columns
