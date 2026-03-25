# Priority SFT v2 — Dataset Statistics

Generated: 2026-03-25

## 1. Train/Dev composition

| File | Mode | Count | Source |
|------|------|------:|--------|
| train_v2.jsonl | conflict | 10,000 | IH-Challenge defenses (grader-filtered) |
| train_v2.jsonl | compatible | 15,596 | Alpaca + GPT-4o-mini (Gemini 3 Flash judge-filtered) |
| train_v2.jsonl | knowledge | 10,000 | SQuAD train |
| train_v2.jsonl | neutral | 10,000 | Alpaca combined |
| dev_v2.jsonl | conflict | 900 | IH-Challenge (50/type, 18 types) |
| dev_v2.jsonl | knowledge | 500 | SQuAD validation (zero train overlap) |
| dev_v2.jsonl | neutral | 500 | Alpaca dev (zero train overlap) |
| **Train total** | | **45,596** | |
| **Dev total** | | **1,900** | |

## 2. Optional add-ons

| File | Count | Source |
|------|------:|--------|
| conflict_composite.jsonl | 4,701 | IH-Challenge multi-constraint tasks, 1 sample per task_id |

## 3. Train — length statistics (characters)

| Mode | Field | p10 | p25 | p50 | p75 | p90 | mean |
|------|-------|----:|----:|----:|----:|----:|-----:|
| conflict | context | 145 | 233 | 674 | 1017 | 1271 | 677 |
| conflict | prompt | 76 | 122 | 234 | 392 | 515 | 275 |
| conflict | response | 4 | 8 | 31 | 193 | 465 | 152 |
| compatible | context | 37 | 46 | 57 | 69 | 84 | 59 |
| compatible | prompt | 57 | 67 | 84 | 114 | 156 | 100 |
| compatible | response | 77 | 138 | 272 | 502 | 711 | 351 |
| knowledge | context | 449 | 560 | 692 | 894 | 1142 | 753 |
| knowledge | prompt | 99 | 133 | 155 | 173 | 191 | 151 |
| knowledge | response | 4 | 7 | 14 | 23 | 42 | 20 |
| neutral | context | 38 | 47 | 57 | 70 | 85 | 60 |
| neutral | prompt | 16 | 34 | 34 | 36 | 75 | 46 |
| neutral | response | 21 | 47 | 151 | 401 | 590 | 254 |

## 4. Dev — length statistics (characters)

| Mode | Field | p10 | p25 | p50 | p75 | p90 | mean |
|------|-------|----:|----:|----:|----:|----:|-----:|
| conflict | context | 60 | 121 | 224 | 353 | 778 | 328 |
| conflict | prompt | 103 | 153 | 254 | 384 | 474 | 277 |
| conflict | response | 8 | 131 | 203 | 374 | 667 | 293 |
| knowledge | context | 450 | 566 | 689 | 877 | 1085 | 752 |
| knowledge | prompt | 154 | 163 | 176 | 191 | 208 | 180 |
| knowledge | response | 4 | 8 | 15 | 23 | 41 | 20 |
| neutral | context | 37 | 46 | 55 | 69 | 84 | 59 |
| neutral | prompt | 20 | 34 | 34 | 38 | 76 | 46 |
| neutral | response | 25 | 52 | 186 | 437 | 626 | 278 |

## 5. Conflict train — by task_type

| Task type | Count | % | Avg ctx (ch) | Avg prompt (ch) | Avg resp (ch) |
|-----------|------:|--:|-------------:|----------------:|--------------:|
| blue_team_auto | 5,538 | 55.4% | 1039 | 255 | 20 |
| nsfw_check | 431 | 4.3% | 258 | 341 | 4 |
| json_format | 422 | 4.2% | 117 | 258 | 385 |
| hidden_word | 409 | 4.1% | 189 | 272 | 442 |
| advertise | 396 | 4.0% | 323 | 266 | 542 |
| sentiment | 384 | 3.8% | 240 | 307 | 7 |
| partial_password | 366 | 3.7% | 121 | 285 | 225 |
| partial_pin | 363 | 3.6% | 238 | 255 | 208 |
| no_PII | 354 | 3.5% | 368 | 363 | 324 |
| letters_only | 353 | 3.5% | 232 | 268 | 555 |
| secret_pin | 343 | 3.4% | 116 | 291 | 249 |
| hide_system | 339 | 3.4% | 337 | 298 | 342 |
| say_banned_words | 302 | 3.0% | 204 | 411 | 593 |

## 6. Conflict train — by attack difficulty level

Levels from IH-Challenge attack generation: L1 = direct attack, L2 = tricky/indirect, L3 = sophisticated (base64, social engineering, fake system messages).

| Level | Count | % |
|------:|------:|--:|
| L1 | 3,688 | 36.9% |
| L2 | 3,129 | 31.3% |
| L3 | 3,183 | 31.8% |

## 7. Conflict train — by IH-Challenge split

IH-Challenge dataset is organized into splits by task complexity: single-constraint (1 atomic rule), input-conditioned (parsing/format tasks), anti-overrefusal (benign requests that look adversarial — model should NOT refuse).

| Split | Count | % |
|-------|------:|--:|
| input-conditioned | 5,538 | 55.4% |
| single-constraint | 3,198 | 32.0% |
| anti-overrefusal | 1,264 | 12.6% |

## 8. Compatible train — by variant

Variant A: extra constraint on existing task. Variant B: second independent task.

| Variant | Count | % | Avg ctx (ch) | Avg prompt (ch) | Avg resp (ch) |
|---------|------:|--:|-------------:|----------------:|--------------:|
| A | 7,594 | 48.7% | 60 | 91 | 261 |
| B | 8,002 | 51.3% | 59 | 109 | 436 |

## 9. Conflict dev — by task_type

- **holdout**: task types NOT in train — used to test generalization to unseen constraint types
- **in_distribution**: task types present in train (but different task_ids with zero overlap)
- **GPT-4o-mini passed**: how many of the 50 dev samples had a correct defense from GPT-4o-mini (0 = all levels failed for these task_ids, meaning these are harder tasks)

| Task type | Count | dev_split | GPT-4o-mini passed |
|-----------|------:|-----------|-------------------:|
| advertise | 50 | in_distribution | 39 |
| avoid_competitors | 50 | holdout | 32 |
| blue_team_auto | 50 | in_distribution | 0 |
| composite | 50 | in_distribution | 0 |
| disclaimer | 50 | holdout | 49 |
| hidden_word | 50 | in_distribution | 45 |
| hide_system | 50 | in_distribution | 0 |
| json_format | 50 | in_distribution | 47 |
| letters_only | 50 | in_distribution | 21 |
| no_CC | 50 | holdout | 26 |
| no_PII | 50 | in_distribution | 0 |
| nsfw_check | 50 | in_distribution | 49 |
| partial_password | 50 | in_distribution | 0 |
| partial_pin | 50 | in_distribution | 0 |
| password | 50 | holdout | 19 |
| say_banned_words | 50 | in_distribution | 0 |
| secret_pin | 50 | in_distribution | 0 |
| sentiment | 50 | in_distribution | 34 |

## 10. Conflict dev — by level and split

| Level | Count | % |
|------:|------:|--:|
| L1 | 318 | 35.3% |
| L2 | 289 | 32.1% |
| L3 | 293 | 32.6% |

| Split | Count | % |
|-------|------:|--:|
| anti-overrefusal | 579 | 64.3% |
| single-constraint | 221 | 24.6% |
| multi-constraint | 50 | 5.6% |
| input-conditioned | 50 | 5.6% |

## 11. Lexical overlap (context ∩ response / response words)

Fraction of response words that also appear in the context. High overlap = response is extracted from context (expected for knowledge mode). Low overlap = response is generated independently (expected for conflict/neutral).

| Mode | p10 | p25 | p50 | p75 | p90 | mean |
|------|----:|----:|----:|----:|----:|-----:|
| conflict | 0.00 | 0.00 | 0.00 | 0.16 | 0.33 | 0.13 |
| compatible | 0.02 | 0.04 | 0.08 | 0.14 | 0.24 | 0.11 |
| knowledge | 0.00 | 0.67 | 1.00 | 1.00 | 1.00 | 0.79 |
| neutral | 0.00 | 0.03 | 0.08 | 0.17 | 0.35 | 0.14 |

