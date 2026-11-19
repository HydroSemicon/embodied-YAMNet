# Cough-only detector handoff

This repository owns microphone capture, BEATs inference, cough classification, temporal decisions, and event serialization. The detector remains a Python process; its PyTorch, CUDA, BEATs source, checkpoint, and classifier artifacts are not embedded into Kokomi Kernel.

## Current artifact

The public-data candidate is generated under `artifacts/beats-cough-public/`:

- `head_calibrated.npz`
- `config_calibrated.json`
- `pipeline_result.json`
- `metrics.json`

The directory is intentionally ignored by Git. Reproduce it from the complete embedding cache with:

```bat
python scripts\train_beats_cough_from_cache.py
```

The head uses the existing five-column serialization format for compatibility, but only the `cough` column is trained and enabled. All other columns are fixed near zero and the cough-only configuration disables their temporal state and ambiguity participation.

## Process contract

Human-operated mode:

```bat
sound-detector live --model beats_transfer --head artifacts\beats-cough-public\head_calibrated.npz --checkpoint checkpoints\BEATs_iter3+_AS2M.pt --beats-source vendor\unilm\beats\BEATs.py --config artifacts\beats-cough-public\config_calibrated.json --event-log artifacts\live\cough-events.jsonl --raw-log artifacts\live\cough-raw.jsonl --inference-device cuda --events-only
```

Machine-operated mode replaces `--events-only` with `--stdout-jsonl`:

```bat
sound-detector live --model beats_transfer --head artifacts\beats-cough-public\head_calibrated.npz --checkpoint checkpoints\BEATs_iter3+_AS2M.pt --beats-source vendor\unilm\beats\BEATs.py --config artifacts\beats-cough-public\config_calibrated.json --event-log artifacts\live\cough-events.jsonl --inference-device cuda --stdout-jsonl
```

In machine mode:

- stdout contains only finalized event JSON objects, one per line;
- stderr contains library warnings and audio diagnostics;
- no startup or shutdown summary is written to stdout;
- every stdout event is also appended to `--event-log`;
- `Ctrl+C` flushes a pending event and exits normally;
- audio is not retained unless `--record-wav` is explicit.

The serialized event remains governed by `schemas/event.schema.json`. Consumers should validate `schema_version`, require `label == "cough"`, and deduplicate by `event_id`.

## Microphone selection and coexistence

List input devices with:

```bat
sound-detector list-input-devices
```

Pass the selected numeric index with `--device`. The detector opens its own PortAudio input stream. Kokomi's browser conversation opens a separate `getUserMedia` stream, so both can normally use the same Windows shared-mode device. This must still be verified on the deployed microphone and driver. Exclusive-mode drivers may reject the second stream.

The detector hears room playback independently of the browser's echo cancellation. Kokomi TTS and filler playback therefore belong in cough hard-negative validation audio.

## Integration boundary

No Kokomi API, HTTP client, MQTT client, or WebSocket transport exists here. A future Kernel adapter should start and supervise the command above, read stdout JSONL, and convert accepted events into Kernel observations. Keeping transport and behavior policy in Kernel prevents the model package from depending on Kernel internals.

If simultaneous microphone acquisition fails, do not silently fall back to a different device. Revisit the boundary and explicitly design a single-capture PCM transport between the browser and this process.

## Deployment gate

The public-data candidate is not production-selected. Completion still requires an untouched target-room test set with:

- at least 50 timestamped cough events;
- at least 10 hours of cough-free continuous audio;
- Kokomi conversation, TTS, fillers, normal speech, throat clearing, sneeze, and room noise;
- precision at least 0.95;
- recall at least 0.80;
- false positives at most 0.5 per hour;
- p95 event latency at most 1.5 seconds.

Training, validation, and test recordings must be split by person and recording session rather than by windows cut from the same recording.
