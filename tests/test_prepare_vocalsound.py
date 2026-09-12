from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_vocalsound.py"
SPEC = importlib.util.spec_from_file_location("prepare_vocalsound", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareVocalSoundTests(unittest.TestCase):
    def _archive(self, root: Path, leaked_speaker: bool = False) -> Path:
        archive = root / "mini-vocalsound.zip"
        mids = list(MODULE.MID_TO_SOURCE_LABEL)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            handle.writestr("LICENSE", "CC BY-SA 4.0")
            handle.writestr("class_labels_indices_vs.csv", "index,mid,display_name\n")
            for split_index, (split, member) in enumerate(MODULE.SPLIT_MEMBERS.items()):
                rows = []
                for label_index, mid in enumerate(mids):
                    speaker = "f0001" if leaked_speaker and label_index == 0 else f"f{split_index}{label_index:03d}"
                    filename = f"{speaker}_{split_index}_{MODULE.MID_TO_SOURCE_LABEL[mid]}.wav"
                    rows.append({"spk_id": speaker, "wav": f"/irrelevant/{filename}", "labels": mid})
                    handle.writestr(f"audio_16k/{filename}", b"RIFF-test")
                handle.writestr(member, json.dumps({"data": rows}))
                meta_name = {"train": "tr", "validation": "val", "test": "te"}[split]
                handle.writestr(f"meta/{meta_name}_meta.csv", "")
        return archive

    def test_preserves_official_split_and_maps_hard_negatives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self._archive(root)
            manifest = root / "manifests" / "vocalsound.jsonl"
            result = MODULE.prepare(
                archive,
                root / "dataset",
                manifest,
                expected_sha256=None,
            )
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(result["clips"], 18)
            self.assertEqual(result["split_counts"], {"train": 6, "validation": 6, "test": 6})
            self.assertEqual({row["split"] for row in rows}, {"train", "validation", "test"})
            self.assertEqual(sum(bool(row["labels"]) for row in rows), 9)
            self.assertEqual(sum(row["is_hard_negative"] for row in rows), 9)
            self.assertTrue(all((manifest.parent / row["path"]).is_file() for row in rows))
            self.assertTrue(all("__MACOSX" not in row["path"] for row in rows))

    def test_rejects_speaker_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self._archive(root, leaked_speaker=True)
            with self.assertRaisesRegex(ValueError, "speaker leakage"):
                MODULE.prepare(
                    archive,
                    root / "dataset",
                    root / "manifest.jsonl",
                    expected_sha256=None,
                )


if __name__ == "__main__":
    unittest.main()
