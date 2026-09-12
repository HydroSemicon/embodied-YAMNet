from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_beats_from_cache.py"
SPEC = importlib.util.spec_from_file_location("train_beats_from_cache", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TrainBeatsFromCacheTests(unittest.TestCase):
    def test_clip_max_aggregation(self):
        logits = np.asarray([[0.1, 0.5], [0.8, 0.2], [0.3, 0.9]], np.float32)
        groups = np.asarray([0, 0, 1], np.int64)
        result = MODULE.aggregate_logits(logits, groups, 2)
        np.testing.assert_allclose(result, [[0.8, 0.5], [0.3, 0.9]])

    def test_load_cache_requires_complete_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cache.json").write_text(json.dumps({"complete": False}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not complete"):
                MODULE.load_cache(root)


if __name__ == "__main__":
    unittest.main()
