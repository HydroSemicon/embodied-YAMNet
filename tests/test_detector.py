from __future__ import annotations

import json
import io
import tempfile
import unittest
import wave
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from daily_sound_detector.audio import AudioQuality, RingBuffer, iter_windows, read_wav, resample, to_mono
from daily_sound_detector.config import ClassDecisionConfig, DetectorConfig, load_config
from daily_sound_detector.evaluation import evaluate_clips
from daily_sound_detector.manifest import ManifestItem, leakage_report, write_prepared_index
from daily_sound_detector.models import LinearHead, MockExtractor, TransferScorer, train_linear_head
from daily_sound_detector.temporal import TemporalDecisionEngine
from daily_sound_detector.training import label_vector
from daily_sound_detector.cli import main as cli_main
from daily_sound_detector.cli import build_parser
from daily_sound_detector.live import _flush_live_events, _print_live_update

ROOT = Path(__file__).resolve().parents[1]
GOOD = AudioQuality(.1, 0, 0, False, False, ())


def write_fixture_wav(path: Path, frequency: float = 440.0, seconds: float = .5, rate: int = 8000, stereo: bool = False):
    """Creates the same tiny deterministic PCM fixture on every CI platform."""
    t = np.arange(int(rate * seconds)) / rate
    mono = (0.2 * np.sin(2 * np.pi * frequency * t) * 32767).astype("<i2")
    pcm = np.column_stack((mono, mono // 2)).astype("<i2").tobytes() if stereo else mono.tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2 if stereo else 1); handle.setsampwidth(2); handle.setframerate(rate); handle.writeframes(pcm)


def test_config(start=.7, end=.3, cooldown=500, merge=400, required=1, history=1, minimum=100):
    classes = {label: ClassDecisionConfig(start, end, history, required, minimum, cooldown, merge, 5000)
               for label in ("cough", "sneeze", "laughter", "alarm", "timer_or_ringtone")}
    return DetectorConfig(classes=classes, ambiguity_margin=.05, min_rms=0)


class AudioTests(unittest.TestCase):
    def test_mono_conversion(self):
        x = np.array([[1, 0], [-1, 1]], np.float32)
        np.testing.assert_allclose(to_mono(x), [.5, 0])

    def test_resampling(self):
        x = np.linspace(-1, 1, 8000, dtype=np.float32)
        y = resample(x, 8000, 16000)
        self.assertEqual(len(y), 16000); self.assertTrue(np.isfinite(y).all())

    def test_ring_buffer_wrap_and_padding(self):
        ring = RingBuffer(5); ring.append(np.array([1, 2], np.float32))
        np.testing.assert_array_equal(ring.latest(4), [0, 0, 1, 2])
        ring.append(np.array([3, 4, 5, 6], np.float32))
        np.testing.assert_array_equal(ring.latest(5), [2, 3, 4, 5, 6])

    def test_windows(self):
        rows = list(iter_windows(np.ones(20000), 16000, 4000))
        self.assertEqual(len(rows), 5); self.assertTrue(all(len(x[1]) == 16000 for x in rows))

    def test_fixed_wav_integration_and_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp, "fixed.wav"); write_fixture_wav(wav, stereo=True)
            first, rate = read_wav(wav); second, _ = read_wav(wav)
            self.assertEqual(rate, 16000); self.assertEqual(len(first), 8000)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(MockExtractor().extract(first), MockExtractor().extract(second))


class ManifestTests(unittest.TestCase):
    def test_label_mapping_is_multilabel(self):
        np.testing.assert_array_equal(label_vector(["cough", "alarm"]), [1, 0, 0, 1, 0])

    def test_source_leakage(self):
        a = ManifestItem(Path("a"), "train", (), "same", "one")
        b = ManifestItem(Path("b"), "test", (), "same", "two")
        self.assertFalse(leakage_report([a, b])["ok"])

    def test_prepare_rejects_leakage(self):
        items = [ManifestItem(Path("a"), "train", (), "same", "one"), ManifestItem(Path("b"), "test", (), "same", "two")]
        with self.assertRaisesRegex(ValueError, "leakage"):
            write_prepared_index(items, "unused")


class ConfigTests(unittest.TestCase):
    def test_default_config(self):
        cfg = load_config(ROOT / "configs" / "default.json")
        self.assertEqual(cfg.sample_rate, 16000); self.assertEqual(set(cfg.classes), set(label_vector.__globals__["LABELS"]))

    def test_invalid_hysteresis(self):
        raw = json.loads((ROOT / "configs" / "default.json").read_text())
        raw["classes"]["cough"]["end_threshold"] = .99
        with self.assertRaisesRegex(ValueError, "thresholds"):
            DetectorConfig.from_dict(raw)

    def test_yamnet_baseline_live_profile(self):
        cfg = load_config(ROOT / "configs" / "yamnet_baseline_live.json")
        self.assertEqual(cfg.sample_rate, 16000)
        self.assertLess(cfg.ambiguity_margin, .08)
        self.assertLess(cfg.classes["laughter"].start_threshold, .75)
        self.assertLess(cfg.classes["alarm"].start_threshold, .82)
        self.assertGreaterEqual(cfg.classes["timer_or_ringtone"].history_frames, 8)


class TemporalTests(unittest.TestCase):
    def engine(self, **kwargs):
        return TemporalDecisionEngine(test_config(**kwargs), {"architecture": "mock", "checkpoint": "none", "classifier_version": "test"}, 0)

    def test_threshold_and_hysteresis(self):
        engine = self.engine()
        engine.update(np.array([.8, 0, 0, 0, 0]), 0, GOOD)
        self.assertTrue(engine.states["cough"].active)
        engine.update(np.array([.5, 0, 0, 0, 0]), 250, GOOD)
        self.assertTrue(engine.states["cough"].active)
        engine.update(np.array([.1, 0, 0, 0, 0]), 500, GOOD)
        events = engine.flush(600)
        self.assertEqual(events[0]["label"], "cough")

    def test_required_frames(self):
        engine = self.engine(required=2, history=3)
        engine.update(np.array([.8, 0, 0, 0, 0]), 0, GOOD)
        self.assertFalse(engine.states["cough"].active)
        engine.update(np.array([.8, 0, 0, 0, 0]), 250, GOOD)
        self.assertTrue(engine.states["cough"].active)

    def test_ambiguity_rejects_unknown(self):
        engine = self.engine()
        _, decision = engine.update(np.array([.9, .88, 0, 0, 0]), 0, GOOD)
        self.assertTrue(decision["ambiguous"]); self.assertFalse(any(s.active for s in engine.states.values()))

    def test_quality_rejects(self):
        poor = AudioQuality(0, 0, 0, True, True, ("silent",))
        engine = self.engine(); _, decision = engine.update(np.array([.9, 0, 0, 0, 0]), 0, poor)
        self.assertIn("silent", decision["quality_reasons"]); self.assertFalse(engine.states["cough"].active)

    def test_cooldown(self):
        engine = self.engine(cooldown=1000, merge=100)
        engine.update(np.array([.9, 0, 0, 0, 0]), 0, GOOD)
        engine.update(np.array([.1, 0, 0, 0, 0]), 250, GOOD)
        engine.update(np.array([.1, 0, 0, 0, 0]), 500, GOOD)  # emits pending
        engine.update(np.array([.9, 0, 0, 0, 0]), 750, GOOD)
        self.assertFalse(engine.states["cough"].active)

    def test_event_merge(self):
        engine = self.engine(cooldown=1000, merge=500)
        engine.update(np.array([.9, 0, 0, 0, 0]), 0, GOOD)
        engine.update(np.array([.1, 0, 0, 0, 0]), 250, GOOD)
        engine.update(np.array([.9, 0, 0, 0, 0]), 500, GOOD)
        self.assertTrue(engine.states["cough"].active)
        engine.update(np.array([.1, 0, 0, 0, 0]), 750, GOOD)
        event = engine.flush(800)[0]
        self.assertEqual(event["duration_ms"], 750)

    def test_event_schema_contract(self):
        engine = self.engine(); engine.update(np.array([.9, 0, 0, 0, 0]), 0, GOOD)
        event = engine.flush(250)[0]
        required = set(json.loads((ROOT / "schemas" / "event.schema.json").read_text())["required"])
        self.assertTrue(required <= event.keys()); self.assertEqual(event["schema_version"], "1.0")


class ModelPipelineTests(unittest.TestCase):
    def test_head_is_multilabel_and_deterministic(self):
        rng = np.random.default_rng(3); x = rng.normal(size=(30, 16)).astype(np.float32)
        y = (rng.random((30, 5)) > .7).astype(np.float32)
        a, _ = train_linear_head(x, y, epochs=5, seed=4); b, _ = train_linear_head(x, y, epochs=5, seed=4)
        np.testing.assert_array_equal(a.predict(x), b.predict(x))
        self.assertEqual(a.predict(x).shape, (30, 5))

    def test_mock_clip_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); wav = root / "fixed.wav"; write_fixture_wav(wav, rate=16000)
            item = ManifestItem(wav, "test", ("cough",), "s1", "session1")
            head = LinearHead(np.zeros((16, 5), np.float32), np.array([3, -3, -3, -3, -3], np.float32))
            scorer = TransferScorer(MockExtractor(), head, "mock", "test")
            result = evaluate_clips([item], scorer, test_config(), root / "out", raw_scores=True)
            self.assertEqual(result["per_class"]["cough"]["tp"], 1)
            self.assertTrue((root / "out" / "raw_scores.jsonl").is_file())

    def test_cli_pipeline_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = root / "manifest.jsonl"
            rows = []
            specs = [("train", "cough", 330), ("train", None, 510), ("validation", "cough", 350),
                     ("validation", None, 530), ("test", "cough", 370), ("test", None, 550)]
            for index, (split, label, frequency) in enumerate(specs):
                wav = root / f"{split}-{index}.wav"; write_fixture_wav(wav, frequency, 1.0, 16000)
                rows.append({"path": str(wav), "split": split, "labels": [label] if label else [],
                             "source_id": f"source-{index}", "session_id": f"session-{index}",
                             "events": ([{"label": label, "start_ms": 0, "end_ms": 900}] if label else [])})
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            model_dir, run_dir = root / "model", root / "run"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(["prepare-dataset", "--manifest", str(manifest), "--output", str(root / "prepared.jsonl")]), 0)
                self.assertEqual(cli_main(["train", "--manifest", str(manifest), "--extractor", "mock", "--output-dir", str(model_dir), "--epochs", "5"]), 0)
                self.assertEqual(cli_main(["calibrate", "--manifest", str(manifest), "--model", "mock_transfer", "--head", str(model_dir / "head.npz"),
                                           "--config", str(ROOT / "configs" / "default.json"), "--output-dir", str(model_dir)]), 0)
                common = ["--manifest", str(manifest), "--model", "mock_transfer", "--head", str(model_dir / "head_calibrated.npz"),
                          "--config", str(model_dir / "config_calibrated.json"), "--output-dir", str(run_dir)]
                self.assertEqual(cli_main(["evaluate", *common, "--raw-scores"]), 0)
                self.assertEqual(cli_main(["evaluate-continuous", *common]), 0)
                self.assertEqual(cli_main(["export-report", "--run-dir", str(run_dir), "--output", str(root / "report.html")]), 0)
            self.assertTrue((root / "report.html").is_file()); self.assertTrue((run_dir / "metrics.json").is_file())


class LiveConsoleTests(unittest.TestCase):
    def decision(self, status="idle"):
        return {"statuses": {label: status for label in label_vector.__globals__["LABELS"]}}

    def test_events_only_prints_nothing_without_confirmed_event(self):
        output = io.StringIO()
        with redirect_stdout(output):
            _print_live_update(True, [], "2026-09-13T00:00:00.000Z",
                               np.array([.9, .1, .1, .1, .1]), self.decision("pending"), test_config())
        self.assertEqual(output.getvalue(), "")

    def test_events_only_prints_one_line_per_confirmed_event(self):
        event = {
            "ended_at": "2026-09-13T00:00:01.250Z", "label": "cough", "confidence": .934,
            "duration_ms": 750, "model": {"architecture": "yamnet_transfer"},
        }
        output = io.StringIO()
        with redirect_stdout(output):
            _print_live_update(True, [event], "unused", np.zeros(5), self.decision(), test_config())
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn("2026-09-13T00:00:01.250Z", lines[0])
        self.assertIn("label=cough", lines[0]); self.assertIn("confidence=0.934", lines[0])
        self.assertIn("duration_ms=750", lines[0]); self.assertIn("model=yamnet_transfer", lines[0])

    def test_legacy_live_detail_output_is_preserved(self):
        output = io.StringIO()
        with redirect_stdout(output):
            _print_live_update(False, [], "2026-09-13T00:00:00.000Z",
                               np.array([.9, .2, .1, .1, .1]), self.decision("pending"), test_config())
        line = output.getvalue().strip()
        self.assertIn("cough, sneeze", line); self.assertIn("0.900", line)
        self.assertIn("0.700", line); self.assertTrue(line.endswith("pending"))

    def test_cli_accepts_events_only(self):
        args = build_parser().parse_args([
            "live", "--model", "mock_transfer", "--head", "head.npz", "--config", "config.json",
            "--event-log", "events.jsonl", "--events-only",
        ])
        self.assertTrue(args.events_only)

    def test_events_only_prints_and_persists_flush_event(self):
        event = {
            "schema_version": "1.0", "event_id": "evt_test", "label": "alarm", "confidence": .91,
            "started_at": "2026-09-13T00:00:00.000Z", "ended_at": "2026-09-13T00:00:01.000Z",
            "duration_ms": 1000, "source": "microphone",
            "model": {"architecture": "beats_transfer", "checkpoint": "test", "classifier_version": "test"},
            "decision": {"start_threshold": .8, "peak_score": .91, "frames_above_threshold": 4},
        }

        class FakeEngine:
            def flush(self, elapsed_ms):
                self.elapsed_ms = elapsed_ms
                return [event]

        with tempfile.TemporaryDirectory() as tmp:
            event_log = Path(tmp) / "events.jsonl"; engine = FakeEngine(); output = io.StringIO()
            with redirect_stdout(output):
                flushed = _flush_live_events(engine, 1250, event_log, True)
            self.assertEqual(flushed, [event]); self.assertEqual(engine.elapsed_ms, 1250)
            self.assertEqual(len(output.getvalue().splitlines()), 1)
            self.assertEqual(json.loads(event_log.read_text(encoding="utf-8"))["event_id"], "evt_test")


if __name__ == "__main__":
    unittest.main()
