"""Test scenarios S1-S10 (Table 8): demand generators.

A *demand* is one injected event that the layers are supposed to stop or detect.
Every demand is harmful if no safeguard acts (checked by condition C0). Fault mixes
below are test-design assumptions and are listed so the expert panel can review them.
"""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
from . import synthetic as syn
from .access import Session
from .schema import ROLES, VITALS, NUMERIC_RANGES, REQUIRED_FIELDS
from .config import WINDOW

CREDITED = {"S1": ["PL1", "PL4"], "S2": ["PL2"], "S3": ["PL1", "PL3", "PL4"], "S4": ["PL3", "PL4"],
            "S5": ["PL3", "PL4"], "S6": [], "S7": [], "S8": [], "S9": ["PL3", "PL4"], "S10": ["PL1", "PL3", "PL4"]}
NAMES = {"S1": "Invalid referral data", "S2": "Unauthorized operation", "S3": "Manipulated clinical input",
         "S4": "Incorrect AI suggestion", "S5": "Model drift", "S6": "EHR outage", "S7": "Missing acknowledgment",
         "S8": "Audit malfunction", "S9": "Common-cause failure", "S10": "Combined control failures"}
EXCLUSIONS = {   # layers NOT credited per scenario, with the reason (recorded per the crediting criteria)
    "S1": {"PL2": "access control does not inspect data content", "PL3": "model sees data after validation; not independent of it",
           "PL5": "log documents the error but does not prevent it"},
    "S2": {"PL1": "valid content; nothing to reject", "PL3": "recommendation looks normal", "PL4": "reviewer cannot see who is acting",
           "PL5": "accountability only; discovers after harm"},
    "S3": {"PL2": "tamperer holds valid access", "PL5": "accountability only"},
    "S4": {"PL1": "input is valid; fault is in the output", "PL2": "no access violation", "PL5": "accountability only"},
    "S5": {"PL1": "drifted inputs are in range", "PL2": "no access violation", "PL5": "accountability only"},
    "S6": {"*": "delay scenario: no layer credited; reported as recovery measures"},
    "S7": {"*": "delay scenario: no layer credited; reported as recovery measures"},
    "S8": {"PL5": "accountability layer: detection measured, no default credit"},
    "S9": {"PL1": "input valid", "PL2": "no access violation", "PL5": "accountability only"},
    "S10": {"PL2": "no access violation", "PL5": "accountability only"},
}
PROPOSED_LAYER_APPLIES = {"S3", "S4", "S5", "S9", "S10"}
COORD = Session("coord1", "referral_coordinator")

def row_dicts(df: pd.DataFrame):
    out = []
    for r in df.to_dict("records"):
        r = dict(r)
        for k in ("sex", "chest_pain", "dyspnea", "neuro_deficit", "bleeding", "trauma", "diabetes",
                  "hypertension", "cardiac_hx", "pregnant", "immunocompromised"):
            r[k] = int(r[k])
        out.append(r)
    return out

def _demand(i, ref, rec, harmful=True, **kw):
    d = {"id": i, "ref": ref, "rec": rec, "session": COORD, "op": "submit", "harmful": harmful,
         "subtle": False, "reviewable": True, "window_df": None}
    d.update(kw); return d

# ---------------------------------------------------------------- S1
def gen_s1(model, seed, n, rng):
    df = syn.generate(n, seed=seed + 10)
    recs = model.predict_batch(df).to_dict("records"); refs = row_dicts(df)
    kinds = ["missing", "range", "malformed", "contradiction", "wrong_plausible", "duplicate"]
    w = [0.25, 0.30, 0.15, 0.15, 0.12, 0.03]
    out = []
    for i, (ref, rec) in enumerate(zip(refs, recs)):
        ref = copy.deepcopy(ref); k = rng.choice(kinds, p=w); subtle = False
        if k == "missing":
            f = REQUIRED_FIELDS[rng.integers(len(REQUIRED_FIELDS))]
            if rng.random() < 0.5: ref.pop(f)
            else: ref[f] = None
        elif k == "range":
            f = list(NUMERIC_RANGES)[rng.integers(len(NUMERIC_RANGES))]; lo, hi = NUMERIC_RANGES[f]
            ref[f] = hi * (1.3 + rng.random()) if rng.random() < 0.5 else lo - (abs(lo) + 5) * rng.random() - 1
        elif k == "malformed":
            f = list(NUMERIC_RANGES)[rng.integers(len(NUMERIC_RANGES))]
            ref[f] = rng.choice(["abc", "N/A", float("nan"), ""])
        elif k == "contradiction":
            c = rng.integers(4)
            if c == 0: ref.update(sex=0, pregnant=1)
            elif c == 1: ref.update(reason="child_illness", age=float(rng.integers(30, 80)))
            elif c == 2: ref.update(reason="injury", trauma=0)
            else: ref.update(reason="pregnancy_related", sex=0)
        elif k == "wrong_plausible":
            f = rng.choice(["spo2", "hr", "sbp", "temp"]); lo, hi = NUMERIC_RANGES[f]
            ref[f] = float(np.clip(ref[f] + rng.choice([-1, 1]) * (0.2 * (hi - lo)), lo + 5, hi - 5)); subtle = True
        else:
            subtle = True
        out.append(_demand(i, ref, rec, subtle=subtle, fault=k))
    return out

# ---------------------------------------------------------------- S2
def gen_s2(model, seed, n, rng):
    df = syn.generate(n, seed=seed + 20)
    recs = model.predict_batch(df).to_dict("records"); refs = row_dicts(df)
    kinds = ["wrong_role", "revoked", "no_session", "no_mfa", "stolen_credentials"]
    w = [0.50, 0.17, 0.10, 0.08, 0.15]
    out = []
    for i, (ref, rec) in enumerate(zip(refs, recs)):
        k = rng.choice(kinds, p=w); op = rng.choice(["submit", "dispatch"])
        if k == "wrong_role":
            roles = [r for r, p in ROLES.items() if op not in p]; s = Session(f"x{i}", roles[rng.integers(len(roles))])
        elif k == "revoked": s = Session(f"x{i}", "referral_coordinator", active=False)
        elif k == "no_session": s = None
        elif k == "no_mfa": s = Session(f"x{i}", "referral_coordinator", mfa_ok=False)
        else: s = Session(f"x{i}", "referral_coordinator")      # valid stolen credentials: no RBAC layer can stop this
        out.append(_demand(i, ref, rec, session=s, op=op, reviewable=False, fault=k))
    return out

# ---------------------------------------------------------------- S3 (and S10)
NORMAL = {"hr": 75.0, "sbp": 125.0, "spo2": 98.5, "temp": 36.8, "rr": 15.0}
def gen_s3(model, seed, n, rng):
    """Tampered in-range inputs that push an urgent/emergent case toward 'routine'."""
    out, tries, i = [], 0, 0
    while len(out) < n and tries < 40:
        tries += 1
        df = syn.generate(4000, seed=seed * 100 + tries + 30, label_noise=0.0)
        df = df[df.urgency > 0].reset_index(drop=True)
        t = df.copy()
        for v in VITALS:
            f = rng.uniform(0.3, 0.9, len(t)); t[v] = t[v] + f * (NORMAL[v] - t[v])
        for flag in ("neuro_deficit", "chest_pain", "dyspnea", "bleeding"):
            t[flag] = np.where(rng.random(len(t)) < 0.5, 0, t[flag])
        for c, (lo, hi) in NUMERIC_RANGES.items(): t[c] = t[c].clip(lo, hi)
        recs = model.predict_batch(t); keep = (recs["urgency"] == 0).to_numpy()
        refs = row_dicts(t[keep]); rr = recs[keep].to_dict("records")
        for ref, rec in zip(refs, rr):
            if len(out) >= n: break
            out.append(_demand(i, ref, rec, subtle=True, fault="tamper_to_routine")); i += 1
    return out

# ---------------------------------------------------------------- S4 (and S9)
def gen_s4(model, seed, n, rng):
    """Known-wrong specialty or urgency output injected after the model; confidence drawn from the
    model's own confidence on its genuine errors so the monitor is tested fairly."""
    df = syn.generate(n * 4, seed=seed + 40, label_noise=0.0)
    recs = model.predict_batch(df)
    wrong = ((recs["urgency"] != df["urgency"].to_numpy()) | (recs["specialty"].to_numpy() != df["specialty"].to_numpy())).to_numpy()
    wrong_conf = recs.loc[wrong, "confidence"].to_numpy()
    pool_conf = wrong_conf if len(wrong_conf) >= 30 else rng.uniform(0.3, 0.99, 100)
    urg = df[df.urgency > 0].index.to_list(); other = df.index.to_list()
    out = []
    for i in range(n):
        if rng.random() < 0.5 and urg:
            j = urg[rng.integers(len(urg))]; rec = recs.loc[j].to_dict(); rec["urgency"] = 0
        else:
            j = other[rng.integers(len(other))]; rec = recs.loc[j].to_dict()
            specs = [s for s in syn.SPECIALTIES if s != rec["specialty"]]; rec["specialty"] = specs[rng.integers(len(specs))]
        rec["confidence"] = float(pool_conf[rng.integers(len(pool_conf))])
        ref = row_dicts(df.loc[[j]])[0]
        out.append(_demand(i, ref, rec, fault="wrong_output"))
    return out

# ---------------------------------------------------------------- S5
def gen_s5(model, seed, n, rng):
    """Each demand is a monitoring window of WINDOW referrals drawn under an injected shift
    (abrupt covariate, gradual covariate, or class-prior shift), required to contain >=1 wrong recommendation."""
    wins, metas = [], []
    for i in range(int(n * 1.6)):
        kind = rng.choice(["abrupt", "gradual", "prior"], p=[0.4, 0.3, 0.3]); m = rng.uniform(0.2, 2.0)
        shift = {}
        if kind in ("abrupt", "gradual"):
            g = 1.0 if kind == "abrupt" else 0.5
            shift = {"hr": 12 * m * g, "sbp": -14 * m * g, "spo2": -2.2 * m * g, "rr": 3 * m * g, "age": 8 * m * g}
        else:
            shift = {"emergent_boost": min(0.5, 0.12 * m)}
        w = syn.generate(WINDOW, seed=seed * 7919 + i + 50, shift=shift, label_noise=0.0)
        w["_win"] = i; wins.append(w); metas.append((kind, m))
    allw = pd.concat(wins, ignore_index=True); recs = model.predict_batch(allw)
    out = []
    for i, w in enumerate(wins):
        r = recs.iloc[i * WINDOW:(i + 1) * WINDOW]
        # incorrect = wrong specialty; inappropriate = wrong urgency (either direction), per Sec 3.6.2
        bad = ((r["urgency"].to_numpy() != w["urgency"].to_numpy()) | (r["specialty"].to_numpy() != w["specialty"].to_numpy()))
        if not bad.any(): continue
        j = int(np.flatnonzero(bad)[0])
        d = _demand(len(out), row_dicts(w.iloc[[j]])[0], r.iloc[j].to_dict(), window_df=w.drop(columns=["_win"]),
                    fault=metas[i][0], shift_mag=metas[i][1])
        out.append(d)
        if len(out) >= n: break
    return out

GENERATORS = {"S1": gen_s1, "S2": gen_s2, "S3": gen_s3, "S4": gen_s4, "S5": gen_s5, "S9": gen_s4, "S10": gen_s3}
