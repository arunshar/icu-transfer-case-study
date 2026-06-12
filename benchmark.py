"""
benchmark.py  --  Deliverable 2 for the ICU-transfer clinical-agent case study.

Reads cohort.jsonl and submits each patient narrative to one or more model
endpoints under STRICTLY STATELESS, single-turn conditions (a fresh message list
per patient, no conversation history, no session state, temperature 0). Captures
the categorical decision (A/B/C), wall-clock latency, and a parse-success flag.

Models:
  * open-source : local Ollama  (default llama3.1:8b) -- runs fully offline, so
                  patient text never leaves the machine. This is the cleanest
                  "zero-retention adhering to MIMIC constraints" configuration.
  * closed      : Anthropic Claude (default claude-3-5-sonnet-20241022), enabled
                  only when ANTHROPIC_API_KEY is set. Anthropic offers
                  zero-data-retention; we additionally never persist prompts.

Usage:
  python benchmark.py --models ollama                 # open-source only
  python benchmark.py --models ollama,claude          # both (needs the key)
  python benchmark.py --models ollama --limit 3       # quick smoke test
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import time

import prompts

HERE = os.path.dirname(os.path.abspath(__file__))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
# Default closed model. claude-3-5-sonnet was retired by mid-2026; use a current id.
CLOSED_MODEL = os.environ.get("CLOSED_MODEL", "claude-sonnet-4-6")


# ---------------------------------------------------------------- model clients
def call_ollama(system: str, user: str) -> str:
    import requests
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": 0, "num_predict": 200},
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


_anthropic_client = None


def make_claude(model_id: str):
    """Return a stateless single-turn caller for a given Claude model id."""
    def _call(system: str, user: str) -> str:
        global _anthropic_client
        import anthropic
        if _anthropic_client is None:
            _anthropic_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        kwargs = dict(model=model_id, max_tokens=1024, system=system,
                      messages=[{"role": "user", "content": user}])
        try:                                  # temperature 0 for determinism ...
            msg = _anthropic_client.messages.create(temperature=0, **kwargs)
        except anthropic.BadRequestError:     # ... unless the model rejects it (thinking models)
            msg = _anthropic_client.messages.create(**kwargs)
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return _call


MODELS = {
    "ollama": {"label": f"ollama:{OLLAMA_MODEL}", "fn": call_ollama, "kind": "open-source"},
    "sonnet": {"label": "closed:claude-sonnet-4-6", "fn": make_claude("claude-sonnet-4-6"), "kind": "closed"},
    "opus":   {"label": "closed:claude-opus-4-8",  "fn": make_claude("claude-opus-4-8"),  "kind": "closed"},
    # generic alias honoring CLOSED_MODEL (defaults to Sonnet 4.6)
    "claude": {"label": f"closed:{CLOSED_MODEL}",  "fn": make_claude(CLOSED_MODEL),        "kind": "closed"},
}


# ------------------------------------------------------------------- benchmark
def run_one(fn, narrative):
    """One stateless call + optional single format-repair. Returns a result dict."""
    system = prompts.SYSTEM_PROMPT
    user = prompts.build_user_prompt(narrative)
    t0 = time.perf_counter()
    repaired = False
    error = ""
    try:
        raw = fn(system, user)
        action, rationale, parse_ok = prompts.parse_action(raw)
        if not parse_ok:
            # one repair attempt: same decision, just fix the format (still stateless)
            repaired = True
            raw2 = fn(system, user + "\n\nIMPORTANT: reply with ONLY the JSON object, nothing else.")
            a2, r2, ok2 = prompts.parse_action(raw2)
            if ok2 or a2 is not None:
                action, rationale, parse_ok, raw = a2, r2, ok2, raw2
    except Exception as e:                       # network / API / model error
        action, rationale, parse_ok, raw = None, "", False, ""
        error = f"{type(e).__name__}: {e}"[:200]
    latency = round(time.perf_counter() - t0, 3)
    return {"predicted_action": action, "rationale": rationale, "parse_ok": parse_ok,
            "repaired": repaired, "latency_s": latency, "raw": (raw or "")[:400], "error": error}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=os.path.join(HERE, "cohort.jsonl"))
    ap.add_argument("--models", default="ollama", help="comma list: ollama,claude")
    ap.add_argument("--limit", type=int, default=0, help="only first N patients (smoke test)")
    ap.add_argument("--out", default=os.path.join(HERE, "results.csv"))
    args = ap.parse_args()

    cohort = [json.loads(l) for l in open(args.cohort)]
    if args.limit:
        cohort = cohort[: args.limit]

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in requested if m not in MODELS]
    if unknown:
        raise SystemExit(f"Unknown models {unknown}. Choose from {list(MODELS)}.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        closed = [m for m in requested if MODELS[m]["kind"] == "closed"]
        if closed:
            print(f"!! ANTHROPIC_API_KEY not set -- skipping closed models {closed}.")
            requested = [m for m in requested if m not in closed]
    if not requested:
        raise SystemExit("No runnable models. Set ANTHROPIC_API_KEY or use --models ollama.")

    rows = []
    for m in requested:
        spec = MODELS[m]
        print(f"\n=== running {spec['label']}  ({spec['kind']}) on {len(cohort)} patients ===")
        for i, p in enumerate(cohort, 1):
            res = run_one(spec["fn"], p["narrative"])
            rows.append({
                "patient_id": p["patient_id"],
                "model": spec["label"],
                "model_kind": spec["kind"],
                "predicted_action": res["predicted_action"] or "",
                "ground_truth_action": p["ground_truth"],
                "latency_s": res["latency_s"],
                "parse_ok": res["parse_ok"],
                "repaired": res["repaired"],
                "correct": int(res["predicted_action"] == p["ground_truth"]),
                "rationale": (res["rationale"] or "").replace("\n", " ")[:240],
                "error": res["error"],
            })
            mark = "OK " if res["parse_ok"] else "(?)"
            print(f"  [{i:>3}/{len(cohort)}] {p['patient_id']:>14}  gt={p['ground_truth']}  "
                  f"pred={res['predicted_action'] or '-'} {mark} {res['latency_s']:.2f}s"
                  + (f"  ERR {res['error']}" if res['error'] else ""))

    fields = ["patient_id", "model", "model_kind", "predicted_action", "ground_truth_action",
              "latency_s", "parse_ok", "repaired", "correct", "rationale", "error"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    # per-model breakouts
    for m in requested:
        label = MODELS[m]["label"]
        sub = [r for r in rows if r["model"] == label]
        path = os.path.join(HERE, f"results_{m}.csv")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(sub)
    print(f"\nwrote {len(rows)} rows -> {args.out} (+ per-model files)")


if __name__ == "__main__":
    main()
