# LOPA prototype run report

> **Read this first.** All results come from *synthetic* referrals with rule-generated labels and *scripted* fault injection. They show how the safeguards behave in controlled tests, not clinical performance. Annual volume, initiating rates and tolerable frequencies are **placeholders** (`config.py`) until Phase 1 data and the expert panel provide real values, so the absolute gap ratios are illustrative; the measured PFDs and the rankings are the substantive output.

## 1. Machine learning component (Table 6 targets)
Selected model: `rf-threshold-821ee4c1`

| Measure | Value | Target | Status |
|---|---|---|---|
| urgent_recall | 0.933 | >= 0.9 | PASS |
| macro_f1 | 0.832 | >= 0.8 | PASS |
| ece | 0.0363 | <= 0.05 | PASS |
| robustness_drop_pts | 5.56 | <= 10.0 | PASS |
| drift_detected_within_one_window | 1 | >= 1.0 | PASS |
| latency_s | 0.0191 | <= 1.0 | PASS |

Robustness stress levels (recall drop, points): eps_0.1=5.56, eps_0.2=26.7, eps_0.3=48.9; malformed inputs rejected by PL1: 1.
Drift monitor: detected 1 of injected shifts, false-alarm rate 0.00833.

## 2. Measured PFDs (condition C1, estimation seeds)

| Scenario | Layer | failures/demands | PFD | 95% Clopper-Pearson | note |
|---|---|---|---|---|---|
| S1 Invalid referral data | PL1 | 727/5000 | 0.145 | [0.136, 0.155] |  |
| S1 Invalid referral data | PL4 | 1419/5000 | 0.284 | [0.271, 0.297] |  |
| S2 Unauthorized operation | PL2 | 708/5000 | 0.142 | [0.132, 0.152] |  |
| S3 Manipulated clinical input | PL1 | 4587/5000 | 0.917 | [0.909, 0.925] |  |
| S3 Manipulated clinical input | PL3 | 5000/5000 | 1 | [0.999, 1] |  |
| S3 Manipulated clinical input | PL4 | 2792/5000 | 0.558 | [0.545, 0.572] |  |
| S4 Incorrect AI suggestion | PL3 | 666/5000 | 0.133 | [0.124, 0.143] |  |
| S4 Incorrect AI suggestion | PL4 | 1248/5000 | 0.25 | [0.238, 0.262] |  |
| S5 Model drift | PL3 | 1911/5000 | 0.382 | [0.369, 0.396] |  |
| S5 Model drift | PL4 | 1236/5000 | 0.247 | [0.235, 0.259] |  |
| S9 Common-cause failure | PL3 | 646/5000 | 0.129 | [0.12, 0.139] |  |
| S9 Common-cause failure | PL4 | 1207/5000 | 0.241 | [0.23, 0.254] |  |
| S10 Combined control failures | PL1 | 4583/5000 | 0.917 | [0.909, 0.924] |  |
| S10 Combined control failures | PL3 | 5000/5000 | 1 | [0.999, 1] |  |
| S10 Combined control failures | PL4 | 2767/5000 | 0.553 | [0.539, 0.567] |  |

## 3. LOPA result per scenario

| Scenario | f_init /yr | PFD product | f_mit /yr | f_tol /yr | gap ratio | MC 95% gap | tolerable? |
|---|---|---|---|---|---|---|---|
| S1 | 600 | 0.0413 | 24.8 | 0.001 | 2.48e+04 | [2.29e+04, 2.68e+04] | no |
| S2 | 24 | 0.142 | 3.4 | 0.0001 | 3.4e+04 | [3.17e+04, 3.64e+04] | no |
| S3 | 12 | 0.512 | 6.15 | 0.0001 | 6.15e+04 | [5.99e+04, 6.31e+04] | no |
| S4 | 360 | 0.0332 | 12 | 0.001 | 1.2e+04 | [1.1e+04, 1.3e+04] | no |
| S5 | 120 | 0.0945 | 11.3 | 0.001 | 1.13e+04 | [1.07e+04, 1.2e+04] | no |
| S9 | 360 | 0.0312 | 11.2 | 0.0001 | 1.12e+05 | [1.03e+05, 1.22e+05] | no |
| S10 | 12 | 0.507 | 6.09 | 0.0001 | 6.09e+04 | [5.93e+04, 6.25e+04] | no |

Ranking by gap: S9 > S3 > S10 > S2 > S1 > S4 > S5. With tolerable frequency x0.1 the ranking is unchanged; with x10 it is unchanged.

## 4. Validation questions

**EQ1 - predicted vs observed harm (separate validation seeds)**

| Scenario | predicted | observed | observed 95% CP | observed inside predicted 95%? | per-seed ranges overlap? | C0 baseline |
|---|---|---|---|---|---|---|
| S1 | 0.0413 | 0.0828 | [0.0753, 0.0908] | NO - not supported | no (0.0325-0.0462 vs 0.078-0.088) | 1 |
| S2 | 0.142 | 0.155 | [0.145, 0.165] | NO - not supported | yes (0.125-0.154 vs 0.142-0.176) | 1 |
| S3 | 0.512 | 0.498 | [0.484, 0.512] | NO - not supported | yes (0.498-0.537 vs 0.477-0.527) | 1 |
| S4 | 0.0332 | 0.0338 | [0.029, 0.0392] | yes | yes (0.0286-0.0382 vs 0.025-0.039) | 1 |
| S5 | 0.0945 | 0.0956 | [0.0876, 0.104] | yes | yes (0.0892-0.107 vs 0.086-0.109) | 1 |
| S9 | 0.0312 | 0.0328 | [0.028, 0.0381] | yes | yes (0.0272-0.0335 vs 0.023-0.038) | 1 |
| S10 | 0.507 | 0.493 | [0.479, 0.507] | NO - not supported | yes (0.49-0.523 vs 0.47-0.511) | 1 |

A *NO* above means the product-of-PFDs prediction is not supported for that scenario: the layers are not independent (for S1, PL1 and PL4 both miss the subtle faults, so their failures are positively correlated) or the demand mix varies between seeds. Dependent layers need conditional PFDs, not the simple product.

**EQ2 - independence (S9).** Independent product of PL3 and PL4 PFDs = 0.0312; observed joint failure with the shared model-output service down in 5% of demands = 0.0836; ratio = 2.68. S10: harm rises from 0.493 (all layers) to 1 when PL1 and PL4 are disabled.

**EQ3 - layer contribution (harm rate, lower is better)**

| Scenario | all credited | layer removed | + proposed rule cross-check (PLX) |
|---|---|---|---|
| S1 | 0.0828 | -PL1: 0.301, -PL4: 0.144 | n/a |
| S2 | 0.155 | n/a | n/a |
| S3 | 0.498 | -PL1: 0.544, -PL3: 0.498, -PL4: 0.912 | 0.495 |
| S4 | 0.0338 | -PL3: 0.259, -PL4: 0.131 | 0.0112 |
| S5 | 0.0956 | -PL3: 0.252, -PL4: 0.376 | 0.043 |
| S9 | 0.0328 | -PL3: 0.249, -PL4: 0.127 | 0.0134 |
| S10 | 0.493 | -PL1: 0.545, -PL3: 0.493, -PL4: 0.908 | 0.49 |

## 5. Not credited: recovery and accountability measures

- **S6 EHR outage**: {"recovery": {"n": 5000, "mean_recovery_h": 2.6117, "p95_recovery_h": 5.461, "share_beyond_window": 0.0002, "manual_routing_share": 0.7808}, "no_fallback": {"n": 5000, "mean_recovery_h": 9.9419, "p95_recovery_h": 31.6512, "share_beyond_window": 0.0052, "manual_routing_share": 0.0}}
- **S7 Missing acknowledgment**: {"recovery": {"n": 5000, "mean_retries": 1.565, "delay_alert_rate": 0.2298, "escalation_rate": 0.0688, "unresolved_or_late_share": 0.0052}}
- **S8 Audit malfunction (PL5)**: detection rate 0.979 of 5000 injected malfunctions; undetected 106; by kind {"alter": {"detected": 1712, "n": 1712}, "remove_tail": {"detected": 1017, "n": 1017}, "alter_rehash": {"detected": 419, "n": 525}, "remove_middle": {"detected": 1016, "n": 1016}, "interrupt": {"detected": 730, "n": 730}}; mean reconstruction 0.00316 ms. No credit by default: a log documents a wrong referral but does not prevent it.

## 6. Layer exclusions recorded per scenario

- S1: PL2: access control does not inspect data content; PL3: model sees data after validation; not independent of it; PL5: log documents the error but does not prevent it
- S2: PL1: valid content; nothing to reject; PL3: recommendation looks normal; PL4: reviewer cannot see who is acting; PL5: accountability only; discovers after harm
- S3: PL2: tamperer holds valid access; PL5: accountability only
- S4: PL1: input is valid; fault is in the output; PL2: no access violation; PL5: accountability only
- S5: PL1: drifted inputs are in range; PL2: no access violation; PL5: accountability only
- S9: PL1: input valid; PL2: no access violation; PL5: accountability only
- S10: PL2: no access violation; PL5: accountability only
