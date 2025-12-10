from unittest.mock import patch

import numpy as np

from src.music_recommender.data.loaders import AudioLoader


@patch("src.music_recommender.data.loaders.librosa.load")
def test_audio_loader_handles_paths(mock_librosa_load, synthetic_audio):
    audio, sr = synthetic_audio

    mock_librosa_load.return_value = (audio, sr)

    loader = AudioLoader()
    fake_path = "non_existent_song.mp3"

    result = loader.transform([fake_path])
    assert isinstance(result, np.ndarray)
    assert len(result) == 1

    single_item = result[0]
    assert "audio" in single_item
    assert "sr" in single_item
    assert single_item["source_type"] == "path"

    mock_librosa_load.assert_called_once()
