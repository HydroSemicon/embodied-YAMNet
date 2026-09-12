from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "combine_manifests.py"
SPEC = importlib.util.spec_from_file_location("combine_manifests", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CombineManifestTests(unittest.TestCase):
    @staticmethod
    def _manifest(root: Path, name: str, split: str, source_id: str, label: str) -> Path:
        directory = root / name
        directory.mkdir()
        audio = directory / "audio.wav"
        audio.write_bytes(b"RIFF-test")
        manifest = directory / "manifest.jsonl"
        manifest.write_text(json.dumps({
            "path": "audio.wav",
            "split": split,
            "labels": [label],
            "source_id": source_id,
            "session_id": source_id + ":session",
        }) + "\n", encoding="utf-8")
        return manifest

    def test_combines_and_rebases_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._manifest(root, "one", "train", "source-one", "cough")
            second = self._manifest(root, "two", "validation", "source-two", "alarm")
            output = root / "combined" / "public.jsonl"
            result = MODULE.combine([first, second], output)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(result["items"], 2)
            self.assertTrue(all((output.parent / row["path"]).is_file() for row in rows))
            self.assertFalse(result["identity_leakage"])

    def test_rejects_cross_split_identity_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._manifest(root, "one", "train", "same-source", "cough")
            second = self._manifest(root, "two", "test", "same-source", "sneeze")
            with self.assertRaisesRegex(ValueError, "identity leakage"):
                MODULE.combine([first, second], root / "combined.jsonl")


if __name__ == "__main__":
    unittest.main()
