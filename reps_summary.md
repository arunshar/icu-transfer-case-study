# Multi-seed robustness (closed models, K reps)

Local llama3.1:8b is temperature-0 deterministic (one run). Closed models run K=5 times each.

| model | sensitivity (recall C) | specificity (recall A) | accuracy | B-rate | rescued vs local | McNemar p |
|---|---|---|---|---|---|---|
| llama3.1:8b (det.) | 12% | 100% | 51% | 4% | - | - |
| Claude Sonnet 4.6 | 46% [42%-49%] | 39% [34%-44%] | 43% [38%-47%] | 25% [18%-32%] | 13.8 [12-15] | 0.0000 [0.0000-0.0000] |
| Claude Opus 4.8 | 18% [17%-20%] | 49% [47%-53%] | 31% [30%-33%] | 55% [52%-59%] | 4.2 [4-5] | 0.6410 [0.4530-0.6880] |

Reproducibility gap: the local model is exactly reproducible; the closed models drift across identical temperature-0 runs, which matters for a regulated, auditable evaluation.