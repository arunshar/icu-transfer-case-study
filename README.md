# ICU-Transfer Prediction Case Study

Can a stateless, zero-retention LLM read a general-ward patient's last 12 hours and
recommend escalation **before** an ICU transfer? This repository contains the deliverables
for that case study, built on the public MIMIC-III demo.

## Deliverables map

| # | deliverable | file(s) |
|---|---|---|
| 1 | cohort construction + 12h pre-transfer window, outputs a JSONL with no post-event data | `extract.py` (and `pipeline.sql`, the SQL companion); output is `cohort.jsonl` |
| 2 | stateless, single-turn inference capturing the A/B/C decision | `benchmark.py` (with `prompts.py`, the system prompt and output-contract parser); output is `results.csv` |
| 3 | one row per patient (id, predicted action, ground-truth action, latency, parse flag) | `results.csv` |
| 4 | 10-minute readout (accuracy, one correct case, one failure, temporal handling, dataset and limitations) | `READOUT.md` and `ICU_Transfer_Readout.pptx` |

`score.py` computes the accuracy, escalation sensitivity, specificity, precision, and
confusion matrices reported in the readout, so the headline numbers are reproducible from
`results.csv`.

## Headline result

73 windows (41 ward-to-ICU escalations labeled C, 32 matched stable-floor controls
labeled A), three models, identical prompt, all parsing 100 percent. The local
llama3.1:8b "wins" raw accuracy (51 percent) by answering "continue" for 65 of 73, a
silent under-triager with 12 percent escalation sensitivity. Claude Sonnet 4.6 reaches 46
percent sensitivity and catches the deteriorations the local model misses. The lesson is
methodological: on an act-and-coordinate task, accuracy ranks the least useful model
first, so fix the metric, the operating point, the action space, and the label before
ranking models. Full discussion, including the temporal-integrity guard, the
length-of-stay leak that was found and removed, and the dataset limitations, is in
`READOUT.md`.

## Run it

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. data + cohort (downloads the 8 MIMIC-III demo CSVs, builds SQLite, writes cohort.jsonl)
python extract.py --download

# 2. open-source model, fully offline (zero data egress)
ollama pull llama3.1:8b
python benchmark.py --models ollama

# 2b. (optional) add the closed models under zero-data-retention
export ANTHROPIC_API_KEY=...
python benchmark.py --models ollama,sonnet,opus

# 3. metrics that back the readout
python score.py
```

## Data and license

Built on the openly licensed [MIMIC-III Clinical Database Demo v1.4](https://physionet.org/content/mimiciii-demo/1.4/)
(100 de-identified, date-shifted patients, no credentialing). No patient data is committed
to this repository; `extract.py --download` fetches it on demand.
