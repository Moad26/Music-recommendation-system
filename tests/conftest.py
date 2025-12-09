import numpy as np
import pytest


@pytest.fixture
def synthetic_audio() -> tuple[np.ndarray, float]:
    sr = 22050
    duration = 35  # to test if the loader really truncate to 30s

    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t)
    return audio, sr
