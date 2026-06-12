"""
score.py  --  metrics for the ICU-transfer benchmark (feeds the 10-minute readout).

Reads results.csv and reports, per model:
  * parse-success rate (format adherence)
  * overall accuracy and BALANCED accuracy (the cohort is not 50/50)
  * recall on C  = escalation sensitivity  (the safety-critical metric: did the
    model catch patients who truly deteriorated?)
  * recall on A  = specificity            (did it leave stable patients alone?)
  * precision on C and the {A,B,C} confusion matrix
  * escalation rate (fraction predicted C) and latency distribution

Writes metrics.json and metrics.md.
"""

from __future__ import annotations
import csv
import json
import math
import os
import statistics as st
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = ["A", "B", "C"]


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion k/n. Returns (lo, hi)."""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def model_metrics(rows):
    n = len(rows)
    parsed = sum(1 for r in rows if r["parse_ok"] == "True")
    valid = [r for r in rows if r["predicted_action"] in LABELS]
    acc = sum(1 for r in valid if r["predicted_action"] == r["ground_truth_action"]) / n if n else 0.0

    # per-class recall + Wilson 95% CI
    recall, ci = {}, {}
    for cls in ["C", "A"]:
        gt = [r for r in rows if r["ground_truth_action"] == cls]
        k = sum(1 for r in gt if r["predicted_action"] == cls)
        recall[cls] = (k / len(gt)) if gt else None
        ci[cls] = wilson(k, len(gt))
    bal = st.mean([v for v in recall.values() if v is not None]) if recall else 0.0

    # precision on C
    predC = [r for r in rows if r["predicted_action"] == "C"]
    precC = (sum(1 for r in predC if r["ground_truth_action"] == "C") / len(predC)) if predC else None

    # confusion matrix gt -> pred
    cm = {g: Counter() for g in LABELS}
    for r in rows:
        if r["ground_truth_action"] in LABELS:
            cm[r["ground_truth_action"]][r["predicted_action"] or "-"] += 1

    lat = [float(r["latency_s"]) for r in rows if r["latency_s"]]
    esc = sum(1 for r in rows if r["predicted_action"] == "C") / n if n else 0.0
    return {
        "n": n,
        "parse_rate": round(parsed / n, 3) if n else 0.0,
        "accuracy": round(acc, 3),
        "balanced_accuracy": round(bal, 3),
        "recall_C_escalation_sensitivity": None if recall["C"] is None else round(recall["C"], 3),
        "recall_C_ci95": ci["C"],
        "recall_A_specificity": None if recall["A"] is None else round(recall["A"], 3),
        "recall_A_ci95": ci["A"],
        "precision_C": None if precC is None else round(precC, 3),
        "escalation_rate": round(esc, 3),
        "latency_s_mean": round(st.mean(lat), 2) if lat else None,
        "latency_s_median": round(st.median(lat), 2) if lat else None,
        "latency_s_p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 2) if lat else None,
        "confusion_gt_to_pred": {g: dict(cm[g]) for g in LABELS},
    }


def main():
    rows = load(os.path.join(HERE, "results.csv"))
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    out = {m: model_metrics(rs) for m, rs in by_model.items()}
    with open(os.path.join(HERE, "metrics.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    lines = ["# Benchmark metrics", ""]
    for m, mm in out.items():
        lines += [f"## {m}", ""]
        lines += [f"- n = {mm['n']}",
                  f"- parse-success rate = {mm['parse_rate']:.1%}",
                  f"- accuracy = {mm['accuracy']:.1%}  |  balanced accuracy = {mm['balanced_accuracy']:.1%}",
                  f"- escalation sensitivity (recall on C) = {fmt(mm['recall_C_escalation_sensitivity'])}  (95% CI {ci_fmt(mm['recall_C_ci95'])})",
                  f"- specificity (recall on A) = {fmt(mm['recall_A_specificity'])}  (95% CI {ci_fmt(mm['recall_A_ci95'])})",
                  f"- precision on C = {fmt(mm['precision_C'])}",
                  f"- escalation rate (predicted C) = {mm['escalation_rate']:.1%}",
                  f"- latency: mean {mm['latency_s_mean']}s, median {mm['latency_s_median']}s, p95 {mm['latency_s_p95']}s",
                  "", "Confusion matrix (rows = ground truth, cols = predicted):", "",
                  "| gt\\pred | A | B | C | - |", "|---|---|---|---|---|"]
        for g in LABELS:
            c = mm["confusion_gt_to_pred"][g]
            lines.append(f"| {g} | {c.get('A',0)} | {c.get('B',0)} | {c.get('C',0)} | {c.get('-',0)} |")
        lines.append("")
    md = "\n".join(lines)
    with open(os.path.join(HERE, "metrics.md"), "w") as fh:
        fh.write(md)
    print(md)


def fmt(x):
    return "n/a" if x is None else f"{x:.1%}"


def ci_fmt(c):
    return "n/a" if not c or c[0] is None else f"{c[0]:.0%}-{c[1]:.0%}"


if __name__ == "__main__":
    main()
