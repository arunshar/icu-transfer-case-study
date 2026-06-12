"""
extract.py  --  Deliverable 1 for the ICU-transfer clinical-agent case study.

Builds a retrospective cohort from the MIMIC-III demo (v1.4), synthesizes a
pre-event clinical narrative for the 12-hour window ending 1 hour before the
decision point, and writes cohort.jsonl (one patient-window per line, temporally
ordered, with no post-event data).

Design decisions (defended in READOUT.md):
  * Target event = a ward -> ICU *transfer* (TRANSFERS.eventtype == 'transfer'
    with curr_careunit in an ICU unit and prev_careunit not an ICU unit). This
    deliberately EXCLUDES eventtype == 'admit' rows whose curr_careunit is an
    ICU, because those are direct ICU admissions, not floor deteriorations.
  * Transfer timestamp = the ICU segment INTIME.
  * Observation window = [event_time - 13h, event_time - 1h]  (12h, ending 1h
    before transfer). Temporal integrity is asserted: nothing at or after
    event_time - 1h enters any prompt.
  * Because the demo is ICU-centric and every subject is an ICU patient, we
    cannot draw negatives from "patients never in the ICU". Instead a CONTROL
    window is a 12h ward window after which the patient was NOT escalated to the
    ICU within the next 12h (they stayed on the floor / were discharged).
    Ground truth = A. This yields a balanced benchmark so accuracy is meaningful
    rather than degenerate recall on an all-positive cohort.

Usage:
  python extract.py --download      # fetch the 8 demo CSVs into data/ (optional)
  python extract.py                 # build cohort.jsonl from data/
"""

from __future__ import annotations
import argparse
import json
import os
import sqlite3
import sys
from datetime import timedelta

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TABLES = ["ADMISSIONS", "PATIENTS", "TRANSFERS", "ICUSTAYS",
          "CHARTEVENTS", "LABEVENTS", "D_ITEMS", "D_LABITEMS"]
BASE_URL = "https://physionet.org/files/mimiciii-demo/1.4"

ICU_UNITS = {"MICU", "SICU", "CCU", "CSRU", "TSICU", "NICU"}

WINDOW_HOURS = 12          # observation window length
LEAD_HOURS = 1             # gap between window end and the decision point
TARGET_TOTAL = 100         # desired total cohort size (positives + controls)

# ----------------------------------------------------------------------------
# Concept dictionaries (CareVue + MetaVision itemids).  Vitals are charted in
# CHARTEVENTS; labs in LABEVENTS.  Reference ranges drive HIGH/LOW flags.
# ----------------------------------------------------------------------------
VITALS = {
    "Heart rate (bpm)":            {"ids": [211, 220045],                          "low": 60,  "high": 100},
    "Systolic BP (mmHg)":          {"ids": [51, 442, 455, 6701, 220050, 220179],  "low": 90,  "high": 140},
    "Diastolic BP (mmHg)":         {"ids": [8368, 8440, 8441, 220051, 220180],    "low": 60,  "high": 90},
    "Mean arterial pressure (mmHg)":{"ids": [52, 456, 6702, 220052, 220181],      "low": 65,  "high": 100},
    "Respiratory rate (/min)":     {"ids": [618, 615, 220210, 224690],            "low": 12,  "high": 20},
    "SpO2 (%)":                    {"ids": [646, 220277],                          "low": 92,  "high": 100},
    "Temperature (F)":             {"ids": [678, 223761],                          "low": 97.0,"high": 100.4},
    "Temperature C->F":            {"ids": [676, 223762],                          "low": 97.0,"high": 100.4, "to_f": True},
    "GCS total":                   {"ids": [198],                                  "low": 15,  "high": 15},
}

LABS = {
    "WBC (K/uL)":        {"ids": [51300, 51301],  "low": 4.0,  "high": 11.0},
    "Hemoglobin (g/dL)": {"ids": [51222, 50811],  "low": 12.0, "high": 17.0},
    "Platelets (K/uL)":  {"ids": [51265],         "low": 150,  "high": 400},
    "Sodium (mEq/L)":    {"ids": [50983, 50824],  "low": 135,  "high": 145},
    "Potassium (mEq/L)": {"ids": [50971, 50822],  "low": 3.5,  "high": 5.1},
    "Bicarbonate (mEq/L)":{"ids": [50882, 50803], "low": 22,   "high": 29},
    "BUN (mg/dL)":       {"ids": [51006],         "low": 7,    "high": 20},
    "Creatinine (mg/dL)":{"ids": [50912],         "low": 0.6,  "high": 1.3},
    "Glucose (mg/dL)":   {"ids": [50931, 50809],  "low": 70,   "high": 180},
    "Lactate (mmol/L)":  {"ids": [50813],         "low": 0.5,  "high": 2.0},
    "Chloride (mEq/L)":  {"ids": [50902, 50806],  "low": 98,   "high": 108},
    "Anion gap":         {"ids": [50868],         "low": 8,    "high": 12},
    "INR":               {"ids": [51237],         "low": 0.9,  "high": 1.2},
}


def download():
    import requests
    os.makedirs(DATA, exist_ok=True)
    for t in TABLES:
        url = f"{BASE_URL}/{t}.csv"
        dst = os.path.join(DATA, f"{t}.csv")
        print(f"downloading {t}.csv ...", end=" ", flush=True)
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        with open(dst, "wb") as fh:
            fh.write(r.content)
        print(f"ok ({len(r.content):,} bytes)")


def load_tables():
    t = {}
    for name in TABLES:
        df = pd.read_csv(os.path.join(DATA, f"{name}.csv"), low_memory=False)
        df.columns = [c.lower() for c in df.columns]
        t[name] = df
    for col in ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"]:
        if col in t["ADMISSIONS"]:
            t["ADMISSIONS"][col] = pd.to_datetime(t["ADMISSIONS"][col], errors="coerce")
    for col in ["intime", "outtime"]:
        t["TRANSFERS"][col] = pd.to_datetime(t["TRANSFERS"][col], errors="coerce")
        t["ICUSTAYS"][col] = pd.to_datetime(t["ICUSTAYS"][col], errors="coerce")
    t["PATIENTS"]["dob"] = pd.to_datetime(t["PATIENTS"]["dob"], errors="coerce")
    t["CHARTEVENTS"]["charttime"] = pd.to_datetime(t["CHARTEVENTS"]["charttime"], errors="coerce")
    t["LABEVENTS"]["charttime"] = pd.to_datetime(t["LABEVENTS"]["charttime"], errors="coerce")
    return t


def build_sqlite(t):
    path = os.path.join(DATA, "mimic_demo.sqlite")
    con = sqlite3.connect(path)
    for name, df in t.items():
        df.to_sql(name.lower(), con, if_exists="replace", index=False)
    con.commit()
    con.close()
    print(f"built SQLite DB at {path}")
    return path


def is_icu(x):
    return isinstance(x, str) and x in ICU_UNITS


def age_years(admittime, dob):
    if pd.isna(admittime) or pd.isna(dob):
        return None
    yrs = (admittime - dob).days / 365.25
    if yrs > 89:          # MIMIC shifts DOB for >89 -> report as 90 ("89+")
        return 90
    return int(round(yrs))


# ----------------------------------------------------------------------------
# Cohort construction
# ----------------------------------------------------------------------------
def find_positives(transfers):
    """First ward -> ICU transfer per hospital admission."""
    tr = transfers
    mask = (tr.eventtype == "transfer") & tr.curr_careunit.apply(is_icu) & (~tr.prev_careunit.apply(is_icu))
    pos = tr[mask].sort_values(["hadm_id", "intime"]).groupby("hadm_id", as_index=False).first()
    out = []
    for _, r in pos.iterrows():
        out.append({"subject_id": int(r.subject_id), "hadm_id": int(r.hadm_id),
                    "event_time": r.intime, "event_kind": "icu_transfer"})
    return out


def icu_intimes_for_hadm(transfers):
    """Map hadm_id -> sorted list of ICU-arrival timestamps (for the stable-future check)."""
    tr = transfers
    icu_rows = tr[tr.curr_careunit.apply(is_icu)]
    m = {}
    for _, r in icu_rows.iterrows():
        m.setdefault(int(r.hadm_id), []).append(r.intime)
    for k in m:
        m[k] = sorted(x for x in m[k] if pd.notna(x))
    return m


def find_negatives(transfers, admissions, labs, positive_hadms, n_target):
    """
    Control windows: a 12h ward window whose end (decision point - 1h) is followed
    by >= 12h with NO ICU transfer (patient stayed on the floor or was discharged).
    Anchored on real lab timestamps so the window has data. Ground truth = A.
    """
    icu_map = icu_intimes_for_hadm(transfers)
    expire = admissions.set_index("hadm_id")["hospital_expire_flag"].to_dict()
    # ward segments = transfer/admit rows whose curr_careunit is NaN (general ward)
    ward = transfers[(transfers.curr_careunit.isna()) & (transfers.eventtype != "discharge")].copy()
    labs_by_hadm = {h: g.sort_values("charttime") for h, g in labs.groupby("hadm_id")}

    negs, seen_hadm = [], set()
    # controls must come from admissions NOT used as positives, one window per admission
    order = list(ward.sort_values(["hadm_id", "intime"]).itertuples(index=False))
    for r in order:
        if len(negs) >= n_target:                 # stop before processing once full
            return negs
        hadm = int(r.hadm_id)
        if hadm in seen_hadm or hadm in positive_hadms:
            continue
        if expire.get(hadm, 0) == 1:              # exclude in-hospital deaths from controls
            continue
        seg_in, seg_out = r.intime, r.outtime
        if pd.isna(seg_in) or pd.isna(seg_out):
            continue
        g = labs_by_hadm.get(hadm)
        if g is None or g.empty:
            continue
        # candidate window ends = lab timestamps >= WINDOW_HOURS into the segment
        # and >= LEAD_HOURS before segment end
        cand = g[(g.charttime >= seg_in + timedelta(hours=WINDOW_HOURS)) &
                 (g.charttime <= seg_out - timedelta(hours=LEAD_HOURS))]
        chosen = None
        for w_end in cand.charttime.unique():
            w_end = pd.Timestamp(w_end)
            w0 = w_end - timedelta(hours=WINDOW_HOURS)
            n_lab = int(((g.charttime >= w0) & (g.charttime < w_end)).sum())
            if n_lab < 5:
                continue
            decision = w_end + timedelta(hours=LEAD_HOURS)
            # stable future: no ICU arrival within the next 12h after the window
            future_icu = [t for t in icu_map.get(hadm, []) if t > w_end]
            if future_icu and (min(future_icu) - w_end) <= timedelta(hours=WINDOW_HOURS):
                continue
            chosen = (w0, w_end, decision)
            break
        if chosen:
            w0, w_end, decision = chosen
            negs.append({"subject_id": int(r.subject_id), "hadm_id": hadm,
                         "event_time": decision, "event_kind": "stable_floor"})
            seen_hadm.add(hadm)
        if len(negs) >= n_target:
            return negs

    # --- top-up: patients DISCHARGED ALIVE from a ward with no ICU in the last 24h.
    # The 12h window ends 1h before discharge. A textbook "stayed stable on the floor"
    # control; decision point = hospital discharge.
    disch = admissions.set_index("hadm_id")["dischtime"].to_dict()
    last_unit = {}
    for hadm, g in transfers[transfers.eventtype != "discharge"].groupby("hadm_id"):
        g = g.dropna(subset=["intime"]).sort_values("intime")
        if len(g):
            last_unit[int(hadm)] = g.iloc[-1].curr_careunit
    for hadm in sorted(set(int(h) for h in transfers.hadm_id.unique())):
        if len(negs) >= n_target:
            break
        if hadm in seen_hadm or hadm in positive_hadms or expire.get(hadm, 0) == 1:
            continue
        if not pd.isna(last_unit.get(hadm, "x")):     # last location must be a ward (NaN)
            continue
        dt = disch.get(hadm)
        if pd.isna(dt):
            continue
        if any((dt - timedelta(hours=24)) <= t <= dt for t in icu_map.get(hadm, [])):
            continue                                   # ICU contact in last 24h -> not a clean control
        g = labs_by_hadm.get(hadm)
        if g is None or g.empty:
            continue
        w1 = dt - timedelta(hours=LEAD_HOURS)
        w0 = w1 - timedelta(hours=WINDOW_HOURS)
        if int(((g.charttime >= w0) & (g.charttime < w1)).sum()) < 3:
            continue
        negs.append({"subject_id": int(g.iloc[0].subject_id), "hadm_id": hadm,
                     "event_time": dt, "event_kind": "discharged_floor"})
        seen_hadm.add(hadm)
    return negs


# ----------------------------------------------------------------------------
# Window data + narrative
# ----------------------------------------------------------------------------
def trend(first, last, lo, hi):
    if first is None or last is None:
        return ""
    span = max(1e-6, (hi - lo) * 0.15)
    if last - first > span:
        return "rising"
    if first - last > span:
        return "falling"
    return "stable"


def flag(val, lo, hi):
    if val is None:
        return ""
    if val > hi:
        return "HIGH"
    if val < lo:
        return "LOW"
    return "normal"


def summarize_series(rows, lo, hi, to_f=False):
    """rows: DataFrame with charttime, valuenum (sorted). Returns dict or None."""
    rows = rows.dropna(subset=["valuenum"]).sort_values("charttime")
    if rows.empty:
        return None
    vals = rows.valuenum.astype(float).tolist()
    if to_f:
        vals = [v * 9 / 5 + 32 for v in vals]
    first, last = vals[0], vals[-1]
    return {"first": round(first, 1), "last": round(last, 1),
            "min": round(min(vals), 1), "max": round(max(vals), 1),
            "n": len(vals), "trend": trend(first, last, lo, hi),
            "flag": flag(last, lo, hi)}


def pull_window(charts_h, labs_h, w0, w1):
    """Return structured vital and lab summaries within [w0, w1)."""
    vitals = {}
    cw = charts_h[(charts_h.charttime >= w0) & (charts_h.charttime < w1)]
    for name, cfg in VITALS.items():
        sub = cw[cw.itemid.isin(cfg["ids"])]
        s = summarize_series(sub, cfg["low"], cfg["high"], cfg.get("to_f", False))
        if s:
            key = "Temperature (F)" if name == "Temperature C->F" else name
            if key not in vitals:        # prefer the first (e.g., native F over converted)
                vitals[key] = s
    labs_out = {}
    lw = labs_h[(labs_h.charttime >= w0) & (labs_h.charttime < w1)]
    for name, cfg in LABS.items():
        sub = lw[lw.itemid.isin(cfg["ids"])]
        s = summarize_series(sub, cfg["low"], cfg["high"])
        if s:
            labs_out[name] = s
    return vitals, labs_out


def synthesize_narrative(ctx, vitals, labs):
    lines = []
    loc = "general ward"
    # NOTE: time-since-admission is deliberately omitted from the narrative. An earlier
    # version printed "admitted about N hours ago" vs "hospital day N", which leaked the
    # class (controls skew late-stay / discharge-anchored), letting a model shortcut the
    # label from phrasing alone. The decision must rest on physiology, not stay length.
    age = ctx.get("age")
    age_str = f"{age}-year-old" if age and age < 90 else ("89+ year-old" if age else "adult")
    dx = (ctx.get("diagnosis") or "").strip().title() or "an unspecified condition"
    atype = (ctx.get("admission_type") or "").strip().title()
    lines.append(f"{age_str} {ctx.get('gender','patient')} on the {loc}. "
                 f"Admission type: {atype or 'unknown'}. Admitting problem: {dx}.")

    lines.append("")
    lines.append("VITAL SIGNS over the last 12 hours:")
    if not vitals:
        lines.append("  No nursing-charted vital signs are documented for this window; "
                     "assessment rests on the laboratory trend below.")
    else:
        for name, s in vitals.items():
            tr = f", {s['trend']}" if s["trend"] else ""
            fl = f" [{s['flag']}]" if s["flag"] not in ("", "normal") else ""
            lines.append(f"  {name}: latest {s['last']}{fl} (range {s['min']}-{s['max']} over {s['n']} readings{tr}).")

    lines.append("")
    lines.append("LABORATORY RESULTS over the last 12 hours (latest value, with trend):")
    if not labs:
        lines.append("  No laboratory results available in this window.")
    else:
        # surface abnormal labs first
        items = sorted(labs.items(), key=lambda kv: 0 if kv[1]["flag"] in ("HIGH", "LOW") else 1)
        for name, s in items:
            tr = f", {s['trend']} from {s['first']}" if s["trend"] else ""
            fl = f" [{s['flag']}]" if s["flag"] not in ("", "normal") else ""
            lines.append(f"  {name}: {s['last']}{fl}{tr}.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="download MIMIC-III demo CSVs first")
    ap.add_argument("--out", default=os.path.join(HERE, "cohort.jsonl"))
    args = ap.parse_args()

    if args.download:
        download()

    missing = [t for t in TABLES if not os.path.exists(os.path.join(DATA, f"{t}.csv"))]
    if missing:
        sys.exit(f"Missing CSVs: {missing}. Run with --download.")

    t = load_tables()
    build_sqlite(t)

    adm, pat = t["ADMISSIONS"], t["PATIENTS"]
    charts, labs = t["CHARTEVENTS"], t["LABEVENTS"]
    adm_ix = adm.set_index("hadm_id")
    pat_ix = pat.set_index("subject_id")

    positives = find_positives(t["TRANSFERS"])
    positive_hadms = {p["hadm_id"] for p in positives}
    n_neg_target = max(0, TARGET_TOTAL - len(positives))
    negatives = find_negatives(t["TRANSFERS"], adm, labs, positive_hadms, n_neg_target)

    charts_by_hadm = {h: g for h, g in charts.groupby("hadm_id")}
    labs_by_hadm = {h: g for h, g in labs.groupby("hadm_id")}

    cohort = []
    leak_violations = 0
    excluded_forbidden = 0
    for rec, gt in [(p, "C") for p in positives] + [(n, "A") for n in negatives]:
        hadm = rec["hadm_id"]
        event_time = rec["event_time"]
        w1 = event_time - timedelta(hours=LEAD_HOURS)   # window end = 1h before decision
        w0 = w1 - timedelta(hours=WINDOW_HOURS)
        admittime = adm_ix.loc[hadm, "admittime"] if hadm in adm_ix.index else pd.NaT
        if pd.notna(admittime):
            w0 = max(w0, admittime)                     # clip to hospital admission
        charts_h = charts_by_hadm.get(hadm, charts.iloc[0:0])
        labs_h = labs_by_hadm.get(hadm, labs.iloc[0:0])

        vitals, labvals = pull_window(charts_h, labs_h, w0, w1)
        if not vitals and not labvals:
            continue                                    # no observable signal -> drop

        # Temporal integrity. By construction the window is [w0, w1) with
        # w1 = event_time - LEAD_HOURS, and every chart/lab query is filtered to
        # charttime < w1, so no observation at or after the cutoff can enter a prompt.
        # We (1) assert that construction invariant as a regression guard (it fires
        # if the window math is ever broken or the window collapses after clipping to
        # admission), and (2) count the observations in the forbidden zone
        # [cutoff, event] that the filter correctly excluded, which proves the filter
        # is doing real work rather than the integrity being vacuous.
        cutoff = w1
        if not (w1 == event_time - timedelta(hours=LEAD_HOURS) and w0 < w1):
            leak_violations += 1
            continue
        excluded_forbidden += int(
            ((charts_h.charttime >= cutoff) & (charts_h.charttime <= event_time)).sum()
            + ((labs_h.charttime >= cutoff) & (labs_h.charttime <= event_time)).sum())

        srow = adm_ix.loc[hadm] if hadm in adm_ix.index else None
        sub = rec["subject_id"]
        dob = pat_ix.loc[sub, "dob"] if sub in pat_ix.index else pd.NaT
        gender = pat_ix.loc[sub, "gender"] if sub in pat_ix.index else ""
        gender = {"M": "male", "F": "female"}.get(gender, "patient")
        ctx = {
            "age": age_years(admittime, dob),
            "gender": gender,
            "diagnosis": srow["diagnosis"] if srow is not None else "",
            "admission_type": srow["admission_type"] if srow is not None else "",
            "hours_since_admit": ((w1 - admittime).total_seconds() / 3600) if pd.notna(admittime) else None,
        }
        narrative = synthesize_narrative(ctx, vitals, labvals)

        cohort.append({
            "patient_id": f"{sub}_{hadm}",
            "subject_id": sub,
            "hadm_id": hadm,
            "ground_truth": gt,                # C = escalate, A = continue floor monitoring
            "event_kind": rec["event_kind"],
            "window_start": w0.isoformat(),
            "window_end": w1.isoformat(),
            "decision_time": event_time.isoformat(),
            "n_vitals": len(vitals),
            "n_labs": len(labvals),
            "narrative": narrative,
            "structured_features": {"vitals": vitals, "labs": labvals},
        })

    # temporal ordering by window_end
    cohort.sort(key=lambda r: r["window_end"])

    with open(args.out, "w") as fh:
        for r in cohort:
            fh.write(json.dumps(r) + "\n")

    n_c = sum(1 for r in cohort if r["ground_truth"] == "C")
    n_a = sum(1 for r in cohort if r["ground_truth"] == "A")
    print("\n================  COHORT SUMMARY  ================")
    print(f"  positives (C, ward->ICU escalation): {n_c}")
    print(f"  controls  (A, stable on floor):      {n_a}")
    print(f"  total windows:                       {len(cohort)}")
    print(f"  window-invariant violations dropped: {leak_violations}")
    print(f"  forbidden-zone observations correctly excluded by the filter: {excluded_forbidden}")
    print(f"  mean labs/window:  {sum(r['n_labs'] for r in cohort)/max(1,len(cohort)):.1f}")
    print(f"  mean vitals/window:{sum(r['n_vitals'] for r in cohort)/max(1,len(cohort)):.1f}")
    print(f"  windows with >=1 charted vital: {sum(1 for r in cohort if r['n_vitals']>0)}")
    print(f"  wrote -> {args.out}")
    print("==================================================")


if __name__ == "__main__":
    main()
