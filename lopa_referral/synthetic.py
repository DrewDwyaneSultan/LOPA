"""Synthetic referral data (Section 3.5.3 "Data" and "Labels and reference standard").

* Features are generated from a documented schema (schema.py) with correlated
  clinical structure plus noise, so results are not trivially high.
* Training labels come from clinician-style referral criteria (``label_training``).
* The 200-case reference set is labelled by a *separate* rule set
  (``label_reference``) standing in for the independently adjudicated set.
  Real clinician adjudication must replace this before results are reported.
* Field distributions are placeholders until de-identified aggregate statistics
  from the partner are provided; this is a stated limitation in the paper.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .schema import SPECIALTIES, URGENCY, REASONS, FLAG_FIELDS, VITALS, FEATURES, NUMERIC_RANGES

REASON_PRIOR = np.array([0.13, 0.12, 0.10, 0.14, 0.10, 0.12, 0.17, 0.12])
REASON_TO_SPECIALTY = {   # primary mapping; overlap/noise added below
    "chest_pain": "cardiology", "breathing_difficulty": "pulmonology",
    "neuro_symptoms": "neurology", "abdominal_pain": "general_surgery",
    "pregnancy_related": "obstetrics_gynecology", "child_illness": "pediatrics",
    "chronic_disease_review": "internal_medicine", "injury": "orthopedics"}

def _clip(df):
    for c, (lo, hi) in NUMERIC_RANGES.items():
        df[c] = df[c].clip(lo, hi)
    return df

def generate(n: int, seed: int = 0, shift: dict | None = None, label_noise: float = 0.04) -> pd.DataFrame:
    """Generate ``n`` referrals. ``shift`` injects drift (used by S5 / drift tests):
    {"hr": +x, "sbp": +x, ..., "emergent_boost": p, "age": +x}."""
    rng = np.random.default_rng(seed)
    shift = shift or {}
    reason_idx = rng.choice(len(REASONS), size=n, p=REASON_PRIOR)
    reason = np.array(REASONS)[reason_idx]
    sex = rng.integers(0, 2, n)
    age = rng.gamma(4.0, 11.0, n) + shift.get("age", 0)
    age = np.where(reason == "child_illness", rng.uniform(0, 14, n), age)
    age = np.where(reason == "pregnancy_related", rng.uniform(16, 44, n), age)
    sex = np.where(reason == "pregnancy_related", 1, sex)
    # latent severity drives vitals; this creates overlap between urgency classes
    sev = rng.beta(1.2, 4.0, n)
    boost = shift.get("emergent_boost", 0.0)
    if boost:
        pick = rng.random(n) < boost
        sev = np.where(pick, rng.beta(5, 1.5, n), sev)
    df = pd.DataFrame({
        "age": age, "sex": sex, "reason": reason,
        "hr": 78 + 55 * sev + rng.normal(0, 11, n) + shift.get("hr", 0),
        "sbp": 124 - 45 * sev + rng.normal(0, 14, n) + shift.get("sbp", 0),
        "spo2": 98 - 11 * sev + rng.normal(0, 2.0, n) + shift.get("spo2", 0),
        "temp": 36.8 + 1.8 * sev + rng.normal(0, 0.5, n) + shift.get("temp", 0),
        "rr": 16 + 14 * sev + rng.normal(0, 3, n) + shift.get("rr", 0),
    })
    p = lambda base, k: np.clip(base + k * sev, 0, 0.95)
    df["chest_pain"] = (rng.random(n) < np.where(reason == "chest_pain", 1.0, 0.04)).astype(int)
    df["dyspnea"] = (rng.random(n) < np.where(reason == "breathing_difficulty", p(0.7, 0.25), p(0.05, 0.3))).astype(int)
    df["neuro_deficit"] = (rng.random(n) < np.where(reason == "neuro_symptoms", p(0.35, 0.5), 0.01)).astype(int)
    df["bleeding"] = (rng.random(n) < np.where(np.isin(reason, ["injury", "pregnancy_related", "abdominal_pain"]), p(0.06, 0.4), 0.01)).astype(int)
    df["trauma"] = (rng.random(n) < np.where(reason == "injury", 1.0, 0.02)).astype(int)
    df["diabetes"] = (rng.random(n) < np.clip(0.04 + age / 400, 0, 0.5)).astype(int)
    df["hypertension"] = (rng.random(n) < np.clip(0.03 + age / 250, 0, 0.7)).astype(int)
    df["cardiac_hx"] = (rng.random(n) < np.clip(0.01 + age / 500, 0, 0.4)).astype(int)
    df["pregnant"] = ((reason == "pregnancy_related") & (rng.random(n) < 0.9)).astype(int)
    df["immunocompromised"] = (rng.random(n) < 0.03).astype(int)
    df = _clip(df)
    df["urgency"] = label_training(df, rng, label_noise)
    df["specialty"] = label_specialty(df, rng, label_noise)
    return df

def _score(df, t):
    """Early-warning-style score; ``t`` holds thresholds (differs per labeller)."""
    s = np.zeros(len(df))
    s += (df.spo2 < t["spo2_hi"]) * 1 + (df.spo2 < t["spo2_lo"]) * 2
    s += (df.sbp < t["sbp_hi"]) * 1 + (df.sbp < t["sbp_lo"]) * 2
    s += (df.hr > t["hr_hi"]) * 1 + (df.hr > t["hr_lo"]) * 2
    s += (df.rr > t["rr_hi"]) * 1 + (df.rr > t["rr_lo"]) * 2
    s += (df.temp > t["temp_hi"]) * 1
    s += df.neuro_deficit * 2 + df.bleeding * 1 + df.chest_pain * 1 + df.dyspnea * 1
    s += (df.immunocompromised & (df.temp > 38.0)) * 1
    return s

TRAIN_T = dict(spo2_hi=94, spo2_lo=90, sbp_hi=105, sbp_lo=90, hr_hi=105, hr_lo=130, rr_hi=22, rr_lo=28, temp_hi=38.5)
REF_T = dict(spo2_hi=95, spo2_lo=91, sbp_hi=108, sbp_lo=92, hr_hi=100, hr_lo=125, rr_hi=21, rr_lo=26, temp_hi=38.3)

def _urgency_from_score(s):
    return np.where(s >= 7, 2, np.where(s >= 4, 1, 0))

def label_training(df, rng=None, noise=0.04):
    y = _urgency_from_score(_score(df, TRAIN_T))
    if rng is not None and noise:
        flip = rng.random(len(df)) < noise
        y = np.where(flip, np.clip(y + rng.choice([-1, 1], len(df)), 0, 2), y)
    return y

def label_reference(df, rng=None):
    """Independent labeller for the reference set (stand-in for clinician adjudication)."""
    y = _urgency_from_score(_score(df, REF_T))
    # an adjudicator would also upgrade acute neuro deficits with abnormal vitals
    y = np.where((df.neuro_deficit == 1) & (df.sbp > 170), np.maximum(y, 1), y)
    if rng is not None:
        flip = rng.random(len(df)) < 0.02
        y = np.where(flip, np.clip(y + rng.choice([-1, 1], len(df)), 0, 2), y)
    return y

def label_specialty(df, rng=None, noise=0.04):
    spec = df.reason.map(REASON_TO_SPECIALTY).to_numpy().astype(object)
    spec = np.where((df.reason == "chest_pain") & (df.dyspnea == 1) & (df.spo2 < 92), "pulmonology", spec)
    spec = np.where((df.reason == "abdominal_pain") & (df.age > 60) & (df.hypertension == 1), "internal_medicine", spec)
    spec = np.where((df.reason == "injury") & (df.bleeding == 1) & (df.sbp < 100), "general_surgery", spec)
    spec = np.where((df.reason == "chest_pain") & (df.age < 15), "pediatrics", spec)
    if rng is not None and noise:
        flip = rng.random(len(df)) < noise
        spec = np.where(flip, rng.choice(SPECIALTIES, len(df)), spec)
    return spec

def reference_set(n: int = 200, seed: int = 9001) -> pd.DataFrame:
    """200-case reference set, enriched with urgent/emergent cases so recall is measurable."""
    rng = np.random.default_rng(seed)
    df = generate(n * 4, seed=seed, label_noise=0.0)
    df["urgency"] = label_reference(df, rng)
    df["specialty"] = label_specialty(df, rng, noise=0.02)
    pos = df[df.urgency > 0]
    neg = df[df.urgency == 0]
    take_pos = min(len(pos), int(n * 0.45))
    out = pd.concat([pos.sample(take_pos, random_state=seed), neg.sample(n - take_pos, random_state=seed)])
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)

def to_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    for r in REASONS:
        X[f"reason_{r}"] = (X["reason"] == r).astype(int)
    return X[FEATURES].astype(float)
