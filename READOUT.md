# 10-Minute Readout: Evaluating Clinical AI Agents for ICU-Transfer Prediction

**Task.** Can a stateless, zero-retention LLM read a general-ward patient's last 12
hours and recommend escalation **before** an ICU transfer? One single-turn call per
patient, choose exactly one: **A** continue floor monitoring, **B** routine morning
labs, **C** escalate to ICU.

**Setup.** From the 100-patient MIMIC-III demo I construct 73 evaluable decision windows
(41 ward to ICU escalations, 32 matched stable-floor controls). Three models, identical
prompt: local **llama3.1:8b** (Ollama, fully offline, the strongest zero-retention guarantee),
plus **Claude Sonnet 4.6** and **Claude Opus 4.8** under zero-data-retention. Stateless,
temperature 0, strict JSON contract. I report Wilson 95% CIs and an exact binomial McNemar test.

---

## 1. Headline: one statistically solid result, wrapped in a benchmark-design lesson

Closed-model figures are mean [min-max] over **5 independent seeds**; the local model
is temperature-0 deterministic (one run, with Wilson 95% CIs).

| model | accuracy | escalation sensitivity (recall C) | specificity (recall A) | "order labs" (B) rate | median latency |
|---|---|---|---|---|---|
| llama3.1:8b (local, deterministic) | **51%** | 12% (5/41), CI 5-26% | 100%, CI 89-100% | 4% | 1.5s |
| Claude Sonnet 4.6 | 43% [38-47] | **46% [42-49]** | 39% [34-44] | 25% [18-32] | 2.3s |
| Claude Opus 4.8 | 31% [30-33] | 18% [17-20] | 49% [47-53] | **55% [52-59]** | 1.9s |

All three models parsed 100% of the time, so format adherence was trivial. Everything
interesting is below the surface:

- **The robust result (defend this with a p-value), and it held across all 5 seeds:**
  Sonnet 4.6 caught **12 to 15 of the deteriorations the local model missed and regressed
  on zero in every seed** (exact binomial McNemar, **p ≤ 0.0005** every run). Escalation
  sensitivity rose from 12% to 46%. On the metric that matters for a safety action,
  capability helped, and the paired test is unambiguous and reproducible.
- **The weakest model "wins" on accuracy.** llama3.1:8b answers "A, continue" for 65 of
  73 patients. Defaulting to the always-continue (negative) class buys 51% accuracy on
  this balanced cohort, and would score ~95% at the real ~1-5% deterioration prevalence,
  while catching only 12% of real deteriorations (CI 5-26%). High accuracy, dangerous
  model. The accuracy ranking (51 / 43 / 31) is inverted relative to usefulness and is
  within noise; I do not defend it as a powered comparison.
- **The frontier model refuses to guess.** Opus 4.8 picks **B ("order more labs") ~55%
  of the time** rather than commit. Because the ground truth is binary by construction,
  B is **unscoreable**, so its accuracy collapses to ~31%. Its sensitivity edge over the
  local model is **not** significant in any seed (McNemar p 0.45 to 0.69), so I make no "Opus is better"
  claim; the honest read is that it defers, and a binary rubric cannot reward deferral.

**The lesson:** choose the metric and the operating point *before* you rank models.
Accuracy ranks the silent under-triager first; escalation-sensitivity-at-a-fixed-alarm-
budget ranks the useful model first. This is an **L3 act-and-coordinate** task, exactly
where maturity-aware benchmarking says final-answer accuracy misleads.

---

## 1a. False-positive cost and operating-point economics

Escalating a stable patient to the ICU is itself costly (a bed, a transfer, alarm
fatigue), so escalation sensitivity has to be read alongside its false-alarm cost.
Precision on C (of the patients a model escalates, how many truly needed it) and the
false-escalation count make that explicit:

| model | escalation sensitivity (recall C) | false escalations of 32 controls | precision on C (PPV) |
|---|---|---|---|
| llama3.1:8b (local) | 12% (5/41) | 0 | 100% |
| Claude Sonnet 4.6 | 46% | 13 | 59% |
| Claude Opus 4.8 | 18% | 2 | 78% |

No model occupies the corner a deployer wants (catch them all, alarm on none): the local
model never false-alarms but misses 88% of deteriorations, while Sonnet catches the most
but false-alarms on 41% of stable patients.

The picture sharpens at real ward prevalence. The balanced 41/32 cohort is ~56% positive;
real ward deterioration is ~1 to 5%, where precision collapses because almost everyone is
truly stable. Holding Sonnet at sensitivity 46% and specificity 39%:

| true prevalence | true deteriorations caught / 1000 | false escalations / 1000 | precision (PPV) |
|---|---|---|---|
| 1% | ~5 | ~604 | 0.8% |
| 3% | ~14 | ~592 | 2.3% |
| 5% | ~23 | ~580 | 3.8% |

At 3% prevalence Sonnet's operating point spends roughly 592 false alarms to find about
14 real deteriorations, a ~2.3% PPV, and would flag ~60% of the whole ward. So
specificity, not sensitivity, is the deployment bottleneck, and the right metric is
precision-recall on C at a fixed alarm budget evaluated at true prevalence, never raw
accuracy on the balanced cohort. The acceptance threshold is a clinical cost decision
(the cost of a missed deterioration versus a false escalation) and must be set before a
model is chosen.

---

## 2. The methodological controls I applied (this is the real work)

- **Cohort construction from the 100-patient demo:** the brief's "100 patients" is the
  size of the MIMIC-III demo database, which I use in full. It yields 55 ward to ICU
  transfers, 41 with usable pre-event signal (14 dropped for too-rapid, empty windows),
  which I pair with 32 matched stable-floor controls (25 lab-anchored + 7 discharged
  alive), for 73 evaluable windows. I build a balanced set rather than an all-positive
  one on purpose: the brief states the label is "transferred or not," and with no controls
  accuracy, specificity, and the false-positive analysis below are all undefined (a model
  that always escalates would score 100%). The balance is what makes the metrics mean
  something.
- **Temporal integrity guaranteed by construction:** every window ends 1h before
  transfer and every chart/lab query is filtered to before that cutoff, so nothing
  at/after it can enter a prompt. I assert the window-positioning invariant for all 73
  windows (**0 violations**) and confirm the filter discarded **943 forbidden-zone
  observations** it would otherwise have seen (proving the guard is not vacuous).
- **I found and removed a templating leak.** An earlier narrative phrased short stays as
  "admitted N hours ago" and long stays as "hospital day N", which correlated with the
  label (controls skew late-stay), letting a model shortcut ~73% accuracy from phrasing
  alone. I stripped stay-length text entirely and **re-ran all three models**; the
  numbers above are post-fix.
- **Statistics:** Wilson 95% CIs on every recall, and an exact paired **McNemar test**
  for the sensitivity contrasts, because at n=73 a bare ranking would be noise.
- **Multi-seed robustness:** the closed reasoning models are **not deterministic at
  temperature 0**, so I ran each 5 times and report mean [range]; the local model is
  bit-identical across runs. The Sonnet result is significant in all 5 seeds; the
  reproducibility gap (open deterministic vs closed drifting) is itself a finding for a
  regulated, auditable evaluation.
- **Evaluator bias avoided by construction:** the label is the deterministic
  ground-truth transfer event and scoring is rule-based code, so there is no
  LLM-as-judge anywhere, and the benchmark is structurally immune to self-preference
  bias (no model grades itself or a sibling). Contamination is handled separately below;
  self-preference simply cannot arise here.
- **Disclosed residual confounds** (below), rather than hide them.

---

## 3. One correct prediction (all three agree)

**Patient 10029 (true C).** 78-year-old, HR 128, MAP 52 (shock), RR 30, temp 104F, WBC
16, lactate 3.9. All three escalated. Sonnet: *"Septic shock: MAP 52, HR 128, RR 30,
temp 104, lactate 3.9, AKI (Cr 3.4), leukocytosis, immediate ICU escalation required."*
When deterioration is unambiguous and vitals are present, every model gets it; the task
only discriminates in the gray zone.

---

## 4. One failure (and the capability gap that closed it)

**Patient 10120 (true C).** 27-year-old in liver failure, **no charted vitals**, lactate
**15.1**, creatinine **10.7**, glucose 444, anion gap 32, platelets 74.

| model | call | reasoning |
|---|---|---|
| llama3.1:8b | **A** (miss) | *"Stable laboratory trend, no acute changes in vital signs; continued monitoring is prudent."* |
| Sonnet 4.6 | **C** | *"Lactate 15.1 with high anion gap, glucose 444 rising sharply, severe AKI (Cr 10.7): critical metabolic instability."* |
| Opus 4.8 | **C** | *"Lactate 15 with anion gap 32, severe renal failure, glucose 444: profound metabolic derangement."* |

The local model anchored on the **absence of vitals** as if it were reassurance and
conflated a **stable trend with not-critical**. A lactate of 15 is a dying patient
regardless of trend. Both Claude models read the absolute lab severity and escalated.
This single case is where the capability gap was decisive (it is an anecdote, n=1; the
powered version of this claim is the McNemar result above), and it is the patient who
matters most. *(No format violations existed: 100% parse on all three models. The failures are
clinical, not formatting.)*

**Failure-mode taxonomy (generalizing this case).** The 10120 miss is one instance of a
broader set of transformer failure modes this benchmark surfaced:

1. **Shortcut learning:** the length-of-stay templating leak gave a ~73% accuracy
   shortcut from phrasing alone; transformers latch onto the easiest predictive signal,
   not the causal one. Found and removed; all models re-run.
2. **Majority-class collapse:** the local model answered "continue" on 65 of 73 windows
   (12% sensitivity); minimizing average loss on a rare, high-cost event teaches the
   model to do nothing.
3. **Surface pattern-matching over absolute severity:** in 10120 the local model treated
   absent vitals and a stable trend as reassurance; a lactate of 15 is a dying patient
   regardless of trend.
4. **Weak numeric and threshold reasoning:** escalation is quantitative (compare values
   to thresholds, integrate a trajectory), where tokenization and compositional logic are
   transformer weak spots.
5. **Closed-model nondeterminism:** at temperature 0 the closed reasoning models drift
   across runs (Sonnet specificity ranged 34 to 44% over 5 seeds), so a single number is
   not reproducible.

---

## 5. How the models handled the temporal data

- All three **used the trend signal** (rationales cite rising/falling/stable), so the
  12h framing landed.
- The local model **over-weighted trend stability over absolute severity** and treated
  **missing vitals as evidence against escalation**, which is backwards.
- Opus treated ambiguity as a reason to **gather more data (B)** rather than commit.
- **No temporal leakage:** windows end 1h pre-transfer; stateless calls see only
  pre-decision data.

---

## 6. Nature of the dataset and limitations

- **MIMIC-III demo:** 100 de-identified ICU patients, openly licensed, date-shifted.
- **ICU-centric charting is the key limitation.** Vitals are recorded mainly in the ICU,
  so the pre-transfer **ward window is lab-dominant**: only 5 of 73 windows had any
  charted vital. This is not really an early-warning feed; a real version needs dense
  ward vitals (eICU or the MIMIC-IV ED module). Residual confound disclosed: those 5
  vital-bearing windows are all positives, an artifact of ICU charting. In principle a
  model could shortcut "vitals present, so escalate" and inflate sensitivity on those 5
  cases; in practice the local model ignored even the vitals it saw, so it did not
  exploit the shortcut. The clean fix is a vitals-symmetric, lab-only ablation.
- **ICU transfer is a noisy proxy for deterioration.** It bundles elective/planned
  post-op ICU admissions: **7 of 41 positives are "Elective" admissions** (e.g., a
  same-day-surgery renal-cancer case), which are not floor deteriorations. The label is
  "got an ICU bed," not "needed one"; it also misses patients capped at comfort care or
  blocked by bed availability. A physiology-based severity label (a NEWS/qSOFA threshold)
  would be cleaner.
- **The unscoreable B option is itself a finding.** Because the label is binary, "order
  morning labs" can never score correct, which penalizes the calibrated "I need more
  information" behavior. A better benchmark makes B a first-class abstain category and
  scores calibration / selective prediction separately.
- **Cohort funnel (reconciling the "100"):** the 100 is the demo database size, used in
  full; it gives 55 ward to ICU transfers, 41 with usable pre-event signal (14 dropped for
  too-rapid, empty windows), plus 32 controls (25 lab-anchored stable windows + 7
  discharge-anchored), for 73 evaluable windows. I should report specificity with the
  discharge subset removed as a sensitivity analysis, and apply one symmetric inclusion
  rule to both arms.
- **Prevalence:** the balanced 41/32 cohort does not reflect the real ~1-5% ward-
  deterioration rate, so PPV and alarm volume at deployment would differ; operating
  points must be set at true prevalence.
- **Contamination:** MIMIC text may be in the closed models' pretraining, so the
  open-vs-closed gap is not a clean capability test without perturbation/counterfactual
  probes. n=73, single site, one cohort.
- **Closed-model nondeterminism:** at temperature 0 the closed reasoning models still
  drift across identical runs (Sonnet specificity ranged 34-44% over 5 seeds), so single
  numbers are unstable. I report 5-seed mean [range]; a regulated deployment would need
  version-pinning and seed-level reproducibility the closed endpoints do not currently
  guarantee.
- **Brief inconsistency handled:** Deliverable 1 said "MIMIC-IV"; the dataset and link
  are the **MIMIC-III demo**, which is what I built on.

---

## 7. What I would change (v2), and how it maps to your team's thesis

1. **Report the right metric:** escalation sensitivity at a fixed alarm budget
   (precision-recall on C) at real prevalence, never raw accuracy.
2. **Fix the action space:** make B a first-class abstain/gather-more-data category and
   score calibration separately (the uncertainty pillar your own "Measuring What Matters"
   found ~96% of the 53 surveyed medical benchmarks skip).
3. **Fix the label:** physiology-based deterioration label, not the transfer event;
   exclude elective/post-op ICU admits.
4. **Guideline-augmented prompting** (NEWS/qSOFA in context), structurally the same move
   as the DRL paper's Differential Reasoning Knowledge Base (DR-KB) of retrieved reasoning
   patches, to lift sensitivity.
5. **Robustness + contamination probes;** add dense ward vitals (eICU / MIMIC-IV ED);
   add latency/cost logging, a de-id/BAA boundary, and drift monitoring for deployment.
6. **Evaluator-bias controls for rationale and abstain scoring:** v1 is judge-free by
   construction, but the moment v2 scores free-text rationales or a first-class abstain
   option, an LLM judge re-introduces self-preference risk. The controls are a
   cross-family judge (never grade a model with itself or a sibling), a diverse jury with
   majority vote, blinding the judge to the source model, and clinician adjudication of a
   sampled subset, anchored wherever possible to a deterministic ground-truth label.

---

## 8. Bottom line

Every model parsed 100% of the time, so the difficulty was entirely clinical, and
reliability did not climb monotonically with capability. One result holds up under a
significance test in all 5 seeds: the mid-tier closed model caught significantly more
real deteriorations than the local model (McNemar p ≤ 0.0005, sensitivity 12% to 46%),
though it traded away specificity (100% to 39%, far more false alarms). The frontier
model mostly deferred to "order more labs," which a binary rubric cannot reward, and
the weakest model posted the highest accuracy by silently doing nothing.

So the production decision is about failure mode, not peak accuracy: choose the local
model's silent under-triage with zero data egress, Sonnet's higher sensitivity at the
cost of alarms, or Opus's information-seeking that a binary rubric cannot score. Picking
that operating point requires designing the evaluation, the action space, the label, and
the acceptance thresholds before you ever rank a model, which is what maturity-aware
benchmarking is for.
