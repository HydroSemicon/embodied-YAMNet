# Data, model, and license ledger

No audio, third-party source tree, or pretrained weight is included in this directory. Copy
`provenance/registry.example.json` to an artifact directory, fill every checksum, and retain it with the run.

## Primary sources

- YAMNet: `https://tfhub.dev/google/yamnet/1`. TensorFlow's official tutorial specifies 16 kHz mono input and documents its 521 AudioSet outputs.
- BEATs: `https://github.com/microsoft/unilm/tree/master/beats`. The initial candidate is the official pretrained `BEATs_iter3+ (AS2M)` checkpoint, not an AudioSet fine-tuned classifier. The code in the official repository is MIT-licensed; verify the checkpoint's terms when downloading.
- VocalSound: `https://github.com/YuanGongND/vocalsound`. The official page identifies the dataset as CC BY-SA 4.0. Preserve speaker IDs for group-disjoint splits.
- FSD50K: `https://doi.org/10.5281/zenodo.4060432`. The release includes a dataset license and per-clip license metadata. Eligibility must be checked clip-by-clip, especially for commercial use.
- AudioSet: use its ontology/segment metadata only in accordance with Google's terms and the underlying YouTube content terms. Do not assume the source audio is redistributable.
- DCASE data: record the exact task, year, subset, URL, and license; "DCASE" is not one license.

## Checksum procedure (PowerShell)

```powershell
Get-FileHash -Algorithm SHA256 .\artifacts\checkpoints\BEATs_iter3+_AS2M.pt
```

Record the exact filename, final resolved URL, SHA-256, retrieval date, license, and source commit for
`BEATs.py`/`backbone.py`. `prepare-dataset` also writes a SHA-256 for every referenced WAV into the prepared index.

## Redistribution rule

The project's `.gitignore` excludes audio and model artifacts. A permissive dataset-level license does not override
per-clip attribution or share-alike obligations. Do not commit a dataset, checkpoint, trained head derived from
restricted material, or a false-positive audio excerpt until its redistribution rights have been reviewed.

