"""
analyze.py  --  cross-model comparison for the readout narrative.

Merges the per-model results_<m>.csv files into results.csv, then prints:
  * a per-model metric line (sensitivity / specificity / accuracy / parse / latency)
  * how each model handled the two showcase patients (10029 septic, 10120 liver failure)
  * "rescued" deteriorations: true-C cases the local model missed (A) but a closed model caught (C)
"""
from __future__ import annotations
import csv, glob, json, math, os, statistics as st
from collections import defaultdict
from math import comb


def mcnemar_exact(b, c):
    """Two-sided exact McNemar (binomial) p-value for discordant pair counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * p)

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = ["A", "B", "C"]


def merge():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "results_*.csv"))):
        rows += list(csv.DictReader(open(f)))
    # de-dup on (patient_id, model) keeping the last
    seen = {}
    for r in rows:
        seen[(r["patient_id"], r["model"])] = r
    merged = list(seen.values())
    fields = list(merged[0].keys())
    with open(os.path.join(HERE, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(merged)
    return merged


def metric_line(rows):
    n = len(rows)
    parsed = sum(1 for r in rows if r["parse_ok"] == "True") / n
    acc = sum(1 for r in rows if r["predicted_action"] == r["ground_truth_action"]) / n
    def rec(c):
        g = [r for r in rows if r["ground_truth_action"] == c]
        return (sum(1 for r in g if r["predicted_action"] == c) / len(g)) if g else float("nan")
    lat = [float(r["latency_s"]) for r in rows if r["latency_s"]]
    esc = sum(1 for r in rows if r["predicted_action"] == "C") / n
    return dict(n=n, parse=parsed, acc=acc, sensC=rec("C"), specA=rec("A"),
                esc=esc, lat=st.median(lat))


def main():
    merged = merge()
    by_model = defaultdict(list)
    for r in merged:
        by_model[r["model"]].append(r)
    cohort = {r["patient_id"]: r for r in (json.loads(l) for l in open(os.path.join(HERE, "cohort.jsonl")))}

    print("MODEL".ljust(34), "n   parse  acc   sensC  specA  esc%   medLat")
    order = sorted(by_model, key=lambda m: (MODELS_ORDER.get(m, 9)))
    for m in order:
        x = metric_line(by_model[m])
        print(m.ljust(34),
              f"{x['n']:<3} {x['parse']:.0%}  {x['acc']:.0%}   {x['sensC']:.0%}    {x['specA']:.0%}    {x['esc']:.0%}    {x['lat']:.2f}s")

    print("\nSHOWCASE PATIENTS")
    for pid, tag in [("10029_132349", "septic shock (true C)"), ("10120_193924", "liver failure (true C)")]:
        print(f"\n  {pid} -- {tag}")
        for m in order:
            r = next((r for r in by_model[m] if r["patient_id"] == pid), None)
            if r:
                print(f"    {m:32} -> {r['predicted_action'] or '-'}  | {r['rationale'][:120]}")

    # rescued deteriorations + McNemar significance on the paired escalation decision (true-C only)
    local = "ollama:llama3.1:8b"
    print("\nESCALATION SENSITIVITY: paired comparison vs the local model (true-C patients)")
    if local in by_model:
        local_pred = {r["patient_id"]: r["predicted_action"] for r in by_model[local]}
        for m in order:
            if m == local:
                continue
            closed_pred = {r["patient_id"]: r["predicted_action"] for r in by_model[m]}
            trueC = [r["patient_id"] for r in by_model[m] if r["ground_truth_action"] == "C"]
            # b = local caught, closed missed ; c = local missed, closed caught (rescued)
            b = sum(1 for p in trueC if local_pred.get(p) == "C" and closed_pred.get(p) != "C")
            c = sum(1 for p in trueC if local_pred.get(p) != "C" and closed_pred.get(p) == "C")
            pval = mcnemar_exact(b, c)
            sig = "SIGNIFICANT" if pval < 0.05 else "not significant"
            print(f"  {m:32} rescued={c}  regressed={b}  McNemar p={pval:.4f}  ({sig} at 0.05)")


MODELS_ORDER = {"ollama:llama3.1:8b": 0, "closed:claude-sonnet-4-6": 1, "closed:claude-opus-4-8": 2}

if __name__ == "__main__":
    main()
