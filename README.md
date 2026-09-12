# Daily Sound Event Detector

マイク／WAVから `cough`、`sneeze`、`laughter`、`alarm`、`timer_or_ringtone` を検出する独立実装です。
5 sigmoid 出力を時系列ルールへ通し、低信頼、曖昧、低品質、OOD候補を `unknown` として棄却します。
Kokomi Kernel、MQTT、HTTP、WebSocketへの接続コードはありません。

このディレクトリは既存リポジトリのファイルから完全に分離して新規作成されています。既存ファイルの変更・移動はしていません。

## 実装範囲

- Baseline A: TF Hub YAMNetのAudioSetスコアを5クラスへ明示的にマッピング
- Baseline B: 凍結YAMNet 1024次元埋め込み + 5出力multi-label sigmoid線形ヘッド
- Candidate C: 公式BEATs埋め込み + 同じ分類ヘッド（encoder凍結）
- 16 kHz mono変換、resampling、1秒／2秒窓、250 ms hop、ring buffer
- clipping、DC offset、無音の検査とraw log
- クラス別threshold、hysteresis、連続フレーム、最小時間、cooldown、merge、最大時間
- validation-only temperature scaling、クラス別PR探索、OOD距離閾値
- clip評価、イベント評価、FP/hour、見逃し／誤検出一覧、遅延、RTF、メモリ、モデルサイズ
- ライブ表示とJSONLイベント出力。録音は `--record-wav` 指定時だけ
- モデル比較HTML/JSON、PR CSV/SVG、confusion matrix CSV

## セットアップ

Python 3.11を推奨します。新しい環境をこのディレクトリ内へ作成してください。

```powershell
cd daily_sound_event_detector
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

YAMNetを使う環境では `pip install -e ".[yamnet,dev]"`、BEATsでは `pip install -e ".[beats,dev]"` を使います。
全方式を同居させる場合は `requirements.lock` から固定バージョンを導入できます。CPU/GPU用PyTorch wheelは環境に応じて公式手順で選び、最終的な `pip freeze` もrun artifactへ保存してください。

## データmanifest

1行1 JSONです。形式は `schemas/manifest.schema.json`、例は `examples/manifest.example.jsonl` にあります。
`source_id` と `session_id` は必須です。人物・機器を識別できる場合は `person_id` と `device_id` も必ず付けます。
連続評価用の正解イベントは `events: [{"label": ..., "start_ms": ..., "end_ms": ...}]` で記録します。
再生音は `playback: true` にし、現実の発声と混ぜません。

分割は人物、音源、機器、録音セッション単位です。`prepare-dataset` は識別子が複数splitに現れた時点で失敗します。

```powershell
sound-detector prepare-dataset --manifest .\my_manifest.jsonl --output .\artifacts\prepared.jsonl
```

最低test条件は各クラス50イベント、対象音なし連続環境音10時間です。testは一度も学習・校正に使いません。
収録には距離・方向・日・セッション、空調、会話、TV、音楽、対象マイク、心海TTS、フィラー、スピーカー再生の対象音を含めます。

## 重みの準備

YAMNetは初回実行時にTF Hubから取得されます。BEATsは自動ダウンロードしません。公式 `microsoft/unilm` の `beats/BEATs.py` と依存する `backbone.py`、および `BEATs_iter3+ (AS2M)` checkpointをローカルに置きます。実行前にSHA-256を `provenance/registry.example.json` のコピーへ記録してください。詳細は `DATA_AND_LICENSES.md` にあります。

## 学習

YAMNet転移学習:

```powershell
sound-detector train --manifest .\my_manifest.jsonl --extractor yamnet --output-dir .\artifacts\yamnet-transfer --augmentation-config .\configs\augmentation.json
```

BEATs転移学習:

```powershell
sound-detector train --manifest .\my_manifest.jsonl --extractor beats --checkpoint .\artifacts\checkpoints\BEATs_iter3+_AS2M.pt --beats-source .\vendor\unilm\beats\BEATs.py --output-dir .\artifacts\beats-transfer --augmentation-config .\configs\augmentation.json
```

初期実装は仕様どおりencoderを凍結します。実環境validationで不足が実証された場合に限り、別runとして最終blockから段階的fine-tuningを追加してください。全層fine-tuningは初手にしません。

augmentationはtrainだけへ適用されます。gain、time shift、背景mix、RIR、軽いtime stretch、maskを設定できます。背景音やRIRを接続する場合も、元録音のsplitを先に固定してください。

## 校正

校正はvalidation splitだけを読みます。温度、開始／終了閾値、OOD距離を保存します。

```powershell
sound-detector calibrate --manifest .\my_manifest.jsonl --model yamnet_standard --config .\configs\default.json --output-dir .\artifacts\yamnet-standard
sound-detector calibrate --manifest .\my_manifest.jsonl --model yamnet_transfer --head .\artifacts\yamnet-transfer\head.npz --config .\configs\default.json --output-dir .\artifacts\yamnet-transfer
```

BEATsでは同じコマンドへ `--checkpoint` と `--beats-source` を加えます。生成された `head_calibrated.npz` と `config_calibrated.json` を評価に使います。

## 同一test setで3方式を評価

```powershell
sound-detector evaluate --manifest .\my_manifest.jsonl --model yamnet_standard --config .\artifacts\yamnet-standard\config_calibrated.json --output-dir .\artifacts\yamnet-standard --raw-scores
sound-detector evaluate --manifest .\my_manifest.jsonl --model yamnet_transfer --head .\artifacts\yamnet-transfer\head_calibrated.npz --config .\artifacts\yamnet-transfer\config_calibrated.json --output-dir .\artifacts\yamnet-transfer --raw-scores
sound-detector evaluate --manifest .\my_manifest.jsonl --model beats_transfer --head .\artifacts\beats-transfer\head_calibrated.npz --checkpoint .\artifacts\checkpoints\BEATs_iter3+_AS2M.pt --beats-source .\vendor\unilm\beats\BEATs.py --config .\artifacts\beats-transfer\config_calibrated.json --output-dir .\artifacts\beats-transfer --raw-scores
```

各方式について連続音評価も実行します。

```powershell
sound-detector evaluate-continuous --manifest .\my_manifest.jsonl --model yamnet_transfer --head .\artifacts\yamnet-transfer\head_calibrated.npz --config .\artifacts\yamnet-transfer\config_calibrated.json --output-dir .\artifacts\yamnet-transfer
```

CPU/GPU別RTFは、それぞれ `--inference-device cpu` / `--inference-device cuda` を付け（YAMNetはTensorFlowの利用可能deviceを自動判定）、run directoryを分けて比較します。test結果を見て閾値を再調整してはいけません。

## ライブ推論

```powershell
sound-detector live --model yamnet_transfer --head .\artifacts\yamnet-transfer\head_calibrated.npz --config .\artifacts\yamnet-transfer\config_calibrated.json --event-log .\artifacts\live\events.jsonl --raw-log .\artifacts\live\raw_scores.jsonl
```

`time | top candidates | score | threshold | pending/detected/suppressed` を表示します。音声保存が必要な場合だけ `--record-wav .\recordings\session.wav` を付けます。JSONLはローカル出力のみです。

確定イベントだけを静かに監視する場合は `--events-only` を付けます。このモードでは候補、pending、suppressed、raw scoreをコンソールへ表示せず、時系列判定でイベントが確定した時だけ1イベント1行を表示します。`--event-log` と `--raw-log` の保存動作は従来どおりです。Logicool BRIOがWindowsの既定入力になっている場合、`--device` は不要です。

cmd.exeでの実行例:

```bat
cd /d C:\GitHub\embodied-YAMNet\daily_sound_event_detector
.venv\Scripts\activate.bat
sound-detector live --model yamnet_transfer --head .\artifacts\yamnet-transfer\head_calibrated.npz --config .\artifacts\yamnet-transfer\config_calibrated.json --event-log .\artifacts\live\events.jsonl --raw-log .\artifacts\live\raw_scores.jsonl --events-only
```

従来の250 msごとの詳細表示を使う場合は `--events-only` を外します。

### YAMNet標準版の診断用プロファイル

`configs/default.json` の閾値は未校正の転移学習モデル向け初期値であり、YAMNet標準出力には高すぎます。実環境データを収集する初期段階では `configs/yamnet_baseline_live.json` を使用できます。

```bat
sound-detector live --model yamnet_standard --config .\configs\yamnet_baseline_live.json --event-log .\artifacts\live\events.jsonl --raw-log .\artifacts\live\raw_scores.jsonl --events-only
```

このプロファイルは少量のライブログから作った診断用設定です。検出しやすさと引き換えに誤検出が増える可能性があります。採用閾値はラベル付きvalidation setで `calibrate` を実行して決め、test setやライブ試行結果に合わせて調整しないでください。

## 比較レポート

`evaluate` と `evaluate-continuous` を同じrun directoryへ出力した後に実行します。

```powershell
sound-detector export-report --run-dir .\artifacts\yamnet-standard --run-dir .\artifacts\yamnet-transfer --run-dir .\artifacts\beats-transfer --output .\reports\model-comparison.html
```

選定順は、FP/hour ≤ 0.5、各クラスprecision ≥ 0.95、その中でmacro recall最大です。合格モデルがなければ `selected: null` とし、未達を隠しません。

## テスト

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m unittest discover -s tests -v
```

テストは決定的に生成する小型PCM WAVとmock extractorだけを使い、マイク、ネットワーク、TensorFlow、PyTorch、GPUを要求しません。resampling、mono、ring buffer、window、label、threshold、hysteresis、cooldown、merge、schema、config、WAV integration、決定性、リーク検査を網羅します。

## 合否を出すまでに残る実験作業

このリポジトリにはユーザーの実録音、VocalSound/FSD50K、checkpointが提供されていないため、実モデルの学習済みhead、PR曲線、FP/hour、実測精度は現時点では生成できません。コードのmock実測をモデル性能として扱わないでください。上記データをmanifestへ登録し、3方式の全コマンドを完走した時点で採用判断が可能になります。
