# Paper ↔ code map

How each part of the manuscript (Chapter 3) is implemented. Everything marked **ASSUMED** is a proposed
value from the paper or a placeholder to be replaced with Phase 1 / ORDI / partner data.

| Paper section | Implemented in | Notes |
|---|---|---|
| 3.5.2 Prototype architecture, Table 5 | `pipeline.py`, `schema.py` | entry → validation → recommendation → monitoring → review → dispatch → audit |
| PL1 Input validation | `validation.py` | missing / range / malformed / contradictory checks |
| PL2 Access controls | `access.py` | role-based permissions, session and MFA checks |
| PL3 Model monitoring | `monitor.py` | PSI drift on rolling window, confidence/OOD check per recommendation |
| PL4 Human oversight | `oversight.py` | simulated reviewer, interception rate 0.5–0.9, reduced when the reviewer sees only model output |
| PL5 Audit mechanisms | `audit.py` | hash-chained, checkpointed log; no credit by default |
| 3.5.3 ML component | `model.py`, `synthetic.py`, `model_eval.py` | logistic regression baseline + random forest; imbalance: none / class weight / SMOTE / threshold |
| Synthetic data + 200-case reference set | `synthetic.py` | training labeller and an *independent* reference labeller (stand-in for clinician adjudication) |
| Table 6 targets | `model_eval.py` | urgent recall ≥ 0.90, macro-F1 ≥ 0.80, ECE ≤ 0.05, robustness drop ≤ 10 pts, drift ≤ 1 window, ≤ 1 s |
| 3.6.2 Scenarios and layers, Table 7 | `scenarios.py` (`CREDITED`, `EXCLUSIONS`) | exclusions are recorded with reasons |
| 3.6.3 Treatment of dependence | `harness.py` (C5), `report.py` (EQ2) | S9 shared model-output service takes PL3 + PL4 down together |
| 3.6.4 Calculation, Eq. 2–3 | `lopa.py` | f_mit = f_init × ΠPFD, gap = f_mit / f_tol |
| 3.6.5 Fault-injection, Table 8, C0–C5 | `scenarios.py`, `harness.py`, `faults.py` | 1,000 demands × 5 seeds; separate validation seeds |
| Eq. 4, Clopper–Pearson, rule of three, Jeffreys, Monte Carlo | `lopa.py` | zero failures → bounded at 3/n, never reported as 0 |
| 3.8.3 Sensitivity (tolerable × 0.1 / × 10) | `lopa.sensitivity` | |
| 3.8.4 EQ1–EQ3 validation | `report.py` | EQ1 uses separate seeds so it is not circular |
| 3.5.5 Workflow replay against Phase 1 baseline | **not built yet** | needs the Phase 1 observation data |
| 3.7 Delphi expert panel | **not code** | the report's tables are what the panel reviews |

## Known gaps / next steps
- `config.py`: annual volume, initiating rates, tolerable frequencies, common-cause rate are placeholders.
- Real clinician-adjudicated reference set and real blinded reviewers replace the simulated ones.
- The referral form follows a generalized baseline; align it with the PH eReferral Implementation Guide / partner form.
- No web UI yet; the prototype is a library plus CLI. A referral-entry screen is the natural next module.
- S6/S7 are simple stochastic simulations (outside the LOPA equation, as in the paper).
