# LOPA – ORDI Commission Thesis

Prototype for the thesis *A Layer of Protection Framework for Managing Technical and Cybersecurity Risks in
AI-Based Healthcare Referral*. It implements Phases 3–4 of the methodology: a minimum viable AI-assisted referral
workflow, five candidate protection layers, a fault-injection harness (scenarios S1–S10, conditions C0–C5) and the
LOPA calculation. See [`docs/PAPER_MAP.md`](docs/PAPER_MAP.md) for how every part maps to the manuscript.

> **Status: first pass.** Synthetic data only, scripted attacks only, placeholder LOPA inputs (`lopa_referral/config.py`).
> Results show how the safeguards behave in controlled tests, **not** clinical performance.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests                   # unit tests
python -m lopa_referral demo                           # one valid and one invalid referral through the workflow
python -m lopa_referral eval-model                     # ML measures vs the Table 6 targets
python -m lopa_referral run --demands 1000 --seeds 5   # full study run -> outputs/REPORT.md, outputs/results.json
python -m lopa_referral run --demands 200 --seeds 2    # quick smoke run (~1.5 min)
python -m lopa_referral serve --open                   # web interface at http://127.0.0.1:8000
```

## Web interface
`python -m lopa_referral serve --open` starts a local, dependency-free web app (standard library only). Sign in with a demo role, then:
**New referral** (live validation, plain-language errors) → **Review queue** (entered details beside the AI suggestion; confirm needs a source-check tick, override needs a reason; monitor-flagged cases go to manual triage) →
**Tracker** (send, confirm receipt, history) → **Audit log** (integrity check) → **Model monitor** (drift demo for the ML-engineer role) → **Risk analysis** (results of the latest `run`).
Each role only sees what it may do, which is the access-control layer in action. Data is synthetic and in memory only; accounts are demo accounts (real authentication is future work).

## What is in the box
| Module | Role |
|---|---|
| `schema.py`, `synthetic.py` | referral form fields, synthetic data, independent 200-case reference set |
| `validation.py` (PL1), `access.py` (PL2), `monitor.py` (PL3), `oversight.py` (PL4), `audit.py` (PL5) | the five layers |
| `model.py`, `model_eval.py` | specialty + urgency models, imbalance handling, Table 6 evaluation |
| `pipeline.py` | the referral workflow with each layer switchable |
| `scenarios.py`, `faults.py`, `harness.py` | S1–S10 test scenarios, C0–C5 conditions, PFD measurement |
| `webapp.py`, `web/index.html` | web interface over the same layers |
| `lopa.py`, `report.py` | f_mit, gap ratio, Clopper–Pearson, rule of three, Jeffreys Monte Carlo, EQ1–EQ3 |

## Honest limitations
- Labels are rule-generated; a model trained on them can only learn the rules. The 200-case reference set is labelled by a
  *separate* rule set as a stand-in; real clinician adjudication must replace it before reporting.
- Fault mixes, common-cause rate, annual volume and tolerable frequencies are **assumptions** for the expert panel to review.
- With 1,000 demands, a layer with zero observed failures is only bounded below ~3×10⁻³; it is never reported as 0.
- Absolute LOPA gap ratios are driven by placeholder inputs; the measured PFDs and rankings are the substantive output.
