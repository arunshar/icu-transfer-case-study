"""
reps.py  --  multi-seed robustness for the closed models.

The local model (Ollama, temperature 0) is bit-deterministic: one run suffices.
The closed reasoning models are NOT deterministic at temperature 0, so we run each
K times and report mean [min, max] for sensitivity, specificity, accuracy, B-rate,
and the McNemar contrast vs the local model. This answers the "single seed, no
variance" critique and quantifies the reproducibility gap between open and closed.

Usage: REPS=5 python reps.py
"""
from __future__ import annotations
import csv, json, math, os, statistics as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import comb

import prompts, benchmark

HERE = os.path.dirname(os.path.abspath(__file__))
K = int(os.environ.get("REPS", "5"))
WORKERS = int(os.environ.get("WORKERS", "6"))
cohort = [json.loads(l) for l in open(os.path.join(HERE, "cohort.jsonl"))]
gt = {p["patient_id"]: p["ground_truth"] for p in cohort}
trueC = [p["patient_id"] for p in cohort if p["ground_truth"] == "C"]
trueA = [p["patient_id"] for p in cohort if p["ground_truth"] == "A"]


def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) * 0.5 ** n)


def run_once(fn):
    preds = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(benchmark.run_one, fn, p["narrative"]): p["patient_id"] for p in cohort}
        for f in as_completed(futs):
            preds[futs[f]] = f.result()["predicted_action"]
    return preds


def metrics(preds, local_preds):
    acc = sum(1 for pid in gt if preds.get(pid) == gt[pid]) / len(gt)
    sensC = sum(1 for pid in trueC if preds.get(pid) == "C") / len(trueC)
    specA = sum(1 for pid in trueA if preds.get(pid) == "A") / len(trueA)
    brate = sum(1 for pid in gt if preds.get(pid) == "B") / len(gt)
    rescued = sum(1 for pid in trueC if preds.get(pid) == "C" and local_preds.get(pid) != "C")
    regressed = sum(1 for pid in trueC if preds.get(pid) != "C" and local_preds.get(pid) == "C")
    return {"acc": acc, "sensC": sensC, "specA": specA, "brate": brate,
            "rescued": rescued, "regressed": regressed, "mcnemar_p": mcnemar_exact(regressed, rescued)}


def agg(key, reps):
    vals = [r[key] for r in reps]
    return {"mean": round(st.mean(vals), 3), "min": round(min(vals), 3), "max": round(max(vals), 3),
            "sd": round(st.pstdev(vals), 3)}


def main():
    # local model: deterministic, reuse the scored results.csv if present, else run once
    local_preds = {}
    rc = os.path.join(HERE, "results.csv")
    if os.path.exists(rc):
        for r in csv.DictReader(open(rc)):
            if r["model"].startswith("ollama"):
                local_preds[r["patient_id"]] = r["predicted_action"]
    if not local_preds:
        local_preds = run_once(benchmark.MODELS["ollama"]["fn"])
    local_m = metrics(local_preds, local_preds)
    print(f"LOCAL (deterministic): sensC {local_m['sensC']:.0%}  specA {local_m['specA']:.0%}  acc {local_m['acc']:.0%}")

    out = {"K": K, "local": local_m, "closed": {}}
    for name in ["sonnet", "opus"]:
        fn = benchmark.MODELS[name]["fn"]
        reps = []
        for k in range(K):
            preds = run_once(fn)
            m = metrics(preds, local_preds)
            reps.append(m)
            print(f"  {name} rep {k+1}/{K}: sensC {m['sensC']:.0%}  specA {m['specA']:.0%}  acc {m['acc']:.0%}  "
                  f"rescued {m['rescued']}/reg {m['regressed']} p={m['mcnemar_p']:.4f}")
        out["closed"][name] = {key: agg(key, reps) for key in ["acc", "sensC", "specA", "brate", "rescued", "regressed", "mcnemar_p"]}

    with open(os.path.join(HERE, "reps_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    # markdown
    L = ["# Multi-seed robustness (closed models, K reps)", "",
         f"Local llama3.1:8b is temperature-0 deterministic (one run). Closed models run K={K} times each.", "",
         "| model | sensitivity (recall C) | specificity (recall A) | accuracy | B-rate | rescued vs local | McNemar p |",
         "|---|---|---|---|---|---|---|",
         f"| llama3.1:8b (det.) | {local_m['sensC']:.0%} | {local_m['specA']:.0%} | {local_m['acc']:.0%} | {local_m['brate']:.0%} | - | - |"]
    for name, lbl in [("sonnet", "Claude Sonnet 4.6"), ("opus", "Claude Opus 4.8")]:
        c = out["closed"][name]
        def cell(k, pct=True):
            a = c[k]
            return f"{a['mean']:.0%} [{a['min']:.0%}-{a['max']:.0%}]" if pct else f"{a['mean']:.1f} [{a['min']:.0f}-{a['max']:.0f}]"
        L.append(f"| {lbl} | {cell('sensC')} | {cell('specA')} | {cell('acc')} | {cell('brate')} | "
                 f"{cell('rescued', False)} | {c['mcnemar_p']['mean']:.4f} [{c['mcnemar_p']['min']:.4f}-{c['mcnemar_p']['max']:.4f}] |")
    L += ["", f"Reproducibility gap: the local model is exactly reproducible; the closed models drift across "
              f"identical temperature-0 runs, which matters for a regulated, auditable evaluation."]
    open(os.path.join(HERE, "reps_summary.md"), "w").write("\n".join(L))
    print("\nwrote reps_summary.md / .json")


if __name__ == "__main__":
    main()
