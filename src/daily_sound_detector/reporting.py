from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from . import LABELS


def _load_if(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _svg_pr(csv_path: Path, output: Path, label: str) -> None:
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    points = " ".join(f"{40 + float(r['recall']) * 420:.1f},{270 - float(r['precision']) * 240:.1f}" for r in rows)
    output.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="500" height="310" viewBox="0 0 500 310">'
        f'<rect width="500" height="310" fill="white"/><path d="M40 30V270H460" fill="none" stroke="#555"/>'
        f'<polyline points="{points}" fill="none" stroke="#146c94" stroke-width="2"/>'
        f'<text x="250" y="302" text-anchor="middle">Recall</text><text x="14" y="155" transform="rotate(-90 14 155)" text-anchor="middle">Precision</text>'
        f'<text x="250" y="20" text-anchor="middle">PR: {html.escape(label)}</text></svg>', encoding="utf-8")


def export_report(run_dirs: list[str | Path], output: str | Path) -> dict:
    runs = []
    for value in run_dirs:
        directory = Path(value)
        metrics, continuous = _load_if(directory / "metrics.json"), _load_if(directory / "continuous_metrics.json")
        if metrics:
            row = {"directory": str(directory), "metrics": metrics, "continuous": continuous}
            fp_h = continuous.get("false_positives_per_hour") if continuous else None
            precision_ok = all(v["precision"] >= 0.95 for v in metrics["per_class"].values())
            row["passes"] = fp_h is not None and fp_h <= 0.5 and precision_ok
            runs.append(row)
            for label in LABELS:
                csv_path = directory / f"pr_curve_{label}.csv"
                if csv_path.is_file(): _svg_pr(csv_path, directory / f"pr_curve_{label}.svg", label)
    eligible = [r for r in runs if r["passes"]]
    selected = max(eligible, key=lambda r: r["metrics"]["macro_recall"], default=None)
    reason = "highest macro recall among models meeting FP/hour and precision constraints" if selected else "no model meets the deployment constraints; no production model selected"
    rows = []
    for run in runs:
        m, c = run["metrics"], run["continuous"] or {}
        rows.append(f"<tr><td>{html.escape(m['model']['architecture'])}</td><td>{m['macro_f1']:.3f}</td><td>{m['macro_recall']:.3f}</td>"
                    f"<td>{c.get('false_positives_per_hour', 'not measured')}</td><td>{c.get('latency_ms', {}).get('p95', 'not measured')}</td><td>{'PASS' if run['passes'] else 'FAIL'}</td></tr>")
    weak = []
    if selected:
        weak = [label for label, values in selected["metrics"]["per_class"].items() if values["precision"] < .95 or values["recall"] < .80]
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Sound Detector Report</title>
<style>body{{font:15px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#17202a}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd;padding:8px;text-align:left}}th{{background:#eef4f7}}code{{background:#f5f5f5;padding:2px 4px}}</style>
<h1>Daily Sound Event Detector — Model Comparison</h1>
<p><strong>Selection:</strong> {html.escape(selected['metrics']['model']['architecture']) if selected else 'none'}</p>
<p><strong>Reason:</strong> {html.escape(reason)}</p>
<table><thead><tr><th>Model</th><th>Macro F1</th><th>Macro recall</th><th>FP/hour</th><th>p95 latency ms</th><th>Gate</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Acceptance targets</h2><p>Per-class precision ≥ 0.95, recall ≥ 0.80, macro F1 ≥ 0.87, total false positives ≤ 0.5/hour, p95 latency ≤ 1500 ms.</p>
<h2>Weak classes</h2><p>{html.escape(', '.join(weak)) if weak else ('None in the selected model.' if selected else 'A model must pass the deployment constraints before weak-class acceptance can be assessed.')}</p>
<h2>Important limitation</h2><p>Thresholds must be fitted on validation data only. The test set is evaluation-only. Public-only results are not sufficient for deployment; evaluate microphone-, room-, TTS-, and filler-specific long negative recordings.</p></html>"""
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(body, encoding="utf-8")
    result = {"models": len(runs), "selected": selected["metrics"]["model"] if selected else None, "reason": reason, "report": str(target)}
    target.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

