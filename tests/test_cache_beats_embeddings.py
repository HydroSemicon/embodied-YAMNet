from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from daily_sound_detector.manifest import ManifestItem


SCRIPT = Path(__file__).parents[1] / "scripts" / "cache_beats_embeddings.py"
SPEC = importlib.util.spec_from_file_location("cache_beats_embeddings", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_wav(path: Path, seconds: float, rate: int = 16_000) -> None:
    samples = np.zeros(int(seconds * rate), dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


class CacheBeatsEmbeddingTests(unittest.TestCase):
    def test_window_plan_matches_training_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            short = root / "short.wav"
            long = root / "long.wav"
            write_wav(short, 1.0)
            write_wav(long, 5.0, 44_100)
            items = [
                ManifestItem(short, "train", ("cough",), "s1", "a"),
                ManifestItem(long, "validation", ("alarm",), "s2", "b"),
            ]

            samples = list(MODULE.iter_window_samples(items))

            self.assertEqual(len(samples), 3)
            self.assertTrue(all(sample.waveform.shape == (32_000,) for sample in samples))
            np.testing.assert_array_equal(samples[0].target, [1, 0, 0, 0, 0])
            np.testing.assert_array_equal(samples[1].target, [0, 0, 0, 1, 0])
            self.assertEqual([sample.window_index for sample in samples], [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
