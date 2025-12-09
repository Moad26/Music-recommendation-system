import numpy as np

from src.music_recommender.data.extractors import SpectrogramExtractor


def test_spectrogram_extractor_truncates_correctly(synthetic_audio):
    audio, sr = synthetic_audio
    extractor = SpectrogramExtractor(target_duration=30)

    pr_audio, pr_sr = extractor._pad_or_truncate((audio, sr), 30.0)
    expected_samples = int(30.0 * 22050)
    assert pr_audio.shape[-1] == expected_samples
    assert pr_sr == sr
    assert np.allclose(pr_audio[:100], audio[:100])
