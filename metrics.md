# Benchmark metrics

## ollama:llama3.1:8b

- n = 73
- parse-success rate = 100.0%
- accuracy = 50.7%  |  balanced accuracy = 56.1%
- escalation sensitivity (recall on C) = 12.2%  (95% CI 5%-26%)
- specificity (recall on A) = 100.0%  (95% CI 89%-100%)
- precision on C = 100.0%
- escalation rate (predicted C) = 6.8%
- latency: mean 1.66s, median 1.55s, p95 2.1s

Confusion matrix (rows = ground truth, cols = predicted):

| gt\pred | A | B | C | - |
|---|---|---|---|---|
| A | 32 | 0 | 0 | 0 |
| B | 0 | 0 | 0 | 0 |
| C | 33 | 3 | 5 | 0 |

## closed:claude-opus-4-8

- n = 73
- parse-success rate = 100.0%
- accuracy = 30.1%  |  balanced accuracy = 32.0%
- escalation sensitivity (recall on C) = 17.1%  (95% CI 8%-31%)
- specificity (recall on A) = 46.9%  (95% CI 31%-64%)
- precision on C = 77.8%
- escalation rate (predicted C) = 12.3%
- latency: mean 2.1s, median 1.94s, p95 3.13s

Confusion matrix (rows = ground truth, cols = predicted):

| gt\pred | A | B | C | - |
|---|---|---|---|---|
| A | 15 | 15 | 2 | 0 |
| B | 0 | 0 | 0 | 0 |
| C | 9 | 25 | 7 | 0 |

## closed:claude-sonnet-4-6

- n = 73
- parse-success rate = 100.0%
- accuracy = 41.1%  |  balanced accuracy = 40.4%
- escalation sensitivity (recall on C) = 46.3%  (95% CI 32%-61%)
- specificity (recall on A) = 34.4%  (95% CI 20%-52%)
- precision on C = 59.4%
- escalation rate (predicted C) = 43.8%
- latency: mean 2.46s, median 2.33s, p95 2.94s

Confusion matrix (rows = ground truth, cols = predicted):

| gt\pred | A | B | C | - |
|---|---|---|---|---|
| A | 11 | 8 | 13 | 0 |
| B | 0 | 0 | 0 | 0 |
| C | 13 | 9 | 19 | 0 |
