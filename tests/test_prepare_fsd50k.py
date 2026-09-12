from __future__ import annotations

import csv
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_fsd50k.py"
SPEC = importlib.util.spec_from_file_location("prepare_fsd50k", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareFsd50kTests(unittest.TestCase):
    def _fixtures(self, root: Path) -> tuple[Path, Path, Path]:
        ground_truth = root / "ground-truth.zip"
        metadata_archive = root / "metadata.zip"
        audio_archive = root / "audio.zip"
        sources = [
            ("1", "Cough,Human_voice", "alice"),
            ("2", "Siren,Alarm", "bob"),
            ("3", "Ringtone,Telephone,Alarm", "carol"),
            ("4", "Doorbell,Alarm", "dana"),
            ("5", "Electric_guitar,Music", "eve"),
            ("6", "Alarm", "frank"),
        ]

        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=("fname", "labels", "mids", "split"))
        writer.writeheader()
        for clip_id, labels, _ in sources:
            writer.writerow({"fname": clip_id, "labels": labels, "mids": "", "split": "train"})
        with zipfile.ZipFile(ground_truth, "w") as archive:
            archive.writestr("FSD50K.ground_truth/dev.csv", stream.getvalue())

        metadata = {
            clip_id: {
                "uploader": uploader,
                "license": "https://creativecommons.org/publicdomain/zero/1.0/",
                "title": f"clip {clip_id}",
            }
            for clip_id, _, uploader in sources
        }
        with zipfile.ZipFile(metadata_archive, "w") as archive:
            archive.writestr(
                "FSD50K.metadata/dev_clips_info_FSD50K.json",
                json.dumps(metadata),
            )
        with zipfile.ZipFile(audio_archive, "w") as archive:
            for clip_id, _, _ in sources:
                archive.writestr(f"FSD50K.dev_audio/{clip_id}.wav", b"RIFF-test")
        return ground_truth, metadata_archive, audio_archive

    def test_maps_targets_negatives_and_excludes_ambiguous_alarm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ground_truth, metadata, audio = self._fixtures(root)
            manifest = root / "manifests" / "fsd.jsonl"
            attribution = root / "manifests" / "attribution.jsonl"
            result = MODULE.prepare(
                ground_truth,
                metadata,
                audio,
                root / "dataset",
                manifest,
                attribution,
                general_per_primary_label=1,
                validation_percent=25,
                verify_metadata_archives=False,
            )
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            by_clip = {row["session_id"].rsplit(":", 1)[-1]: row for row in rows}

            self.assertEqual(result["clips"], 5)
            self.assertEqual(by_clip["1"]["labels"], ["cough"])
            self.assertEqual(by_clip["2"]["labels"], ["alarm"])
            self.assertEqual(by_clip["3"]["labels"], ["timer_or_ringtone"])
            self.assertTrue(by_clip["4"]["is_hard_negative"])
            self.assertFalse(by_clip["5"]["is_hard_negative"])
            self.assertNotIn("6", by_clip)
            self.assertTrue(all((manifest.parent / row["path"]).is_file() for row in rows))

    def test_uploader_is_never_split(self):
        metadata = {
            "1": {"uploader": "same", "license": "cc0", "title": "a"},
            "2": {"uploader": "same", "license": "cc0", "title": "b"},
        }
        rows = [
            {"fname": "1", "labels": "Cough"},
            {"fname": "2", "labels": "Siren,Alarm"},
        ]
        selected = MODULE.select_rows(rows, metadata, general_per_primary_label=0)
        self.assertEqual(len({row["split"] for row in selected}), 1)


if __name__ == "__main__":
    unittest.main()
