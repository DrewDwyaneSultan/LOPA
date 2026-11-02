"""Fault-injection harness (Section 3.6.5): scenarios S1-S10 x conditions C0-C5.

C0 no safeguards | C1 each credited layer alone | C2 all credited layers |
C3 each layer removed in turn | C4 proposed layer (PLX) added | C5 shared / combined failures (S9, S10).
Each (scenario, condition) runs ``demands`` demands in each random seed; seeds are separate for
estimation (SEEDS_ESTIMATE) and validation (SEEDS_VALIDATE) so the EQ1 comparison is not circular.
"""
from __future__ import annotations
import json, time, zlib
import numpy as np
from .config import (DEMANDS_PER_LAYER, SEEDS_ESTIMATE, SEEDS_VALIDATE, REVIEWER_INTERCEPT,
                     COMMON_CAUSE_RATE, PSI_THRESHOLD, WINDOW)
from .monitor import ModelMonitor
from .oversight import SimulatedReviewer
from .pipeline import ReferralSystem
from . import scenarios as sc, faults, synthetic as syn

PIPELINE_SCENARIOS = ["S1", "S2", "S3", "S4", "S5", "S9", "S10"]

def conditions(sid):
    cred = sc.CREDITED[sid]; c = {"C0": {"active": []}}
    for L in cred: c[f"C1:{L}"] = {"active": [L]}
    c["C2"] = {"active": cred}
    if len(cred) > 1:
        for L in cred: c[f"C3:-{L}"] = {"active": [x for x in cred if x != L]}
    if sid in sc.PROPOSED_LAYER_APPLIES: c["C4:+PLX"] = {"active": cred + ["PLX"]}
    if sid == "S9": c["C5:common_cause"] = {"active": cred, "common_cause": True}
    if sid == "S10": c["C5:PL1+PL4_down"] = {"active": cred, "disabled": ["PL1", "PL4"]}
    return c

def _run_config(demands, cfg, monitor, seed_tag, reviewer_rate, cc_rate):
    rng = np.random.default_rng(zlib.crc32(seed_tag.encode()))
    sys_ = ReferralSystem(cfg["active"], SimulatedReviewer(reviewer_rate), monitor, rng)
    harms, cls, by = 0, {}, {}
    for d in demands:
        shared = bool(cfg.get("common_cause") and d.get("u_cc", rng.random()) < cc_rate)
        o = sys_.process(d, disabled=cfg.get("disabled", ()), shared_down=shared)
        harms += int(o["harm"])
        if o["cls"]: cls[o["cls"]] = cls.get(o["cls"], 0) + 1
        if o["by"]: by[o["by"]] = by.get(o["by"], 0) + 1
    return {"n": len(demands), "harms": harms, "classes": cls, "stopped_by": by}

def run_scenario(sid, model, monitor, seeds, demands=DEMANDS_PER_LAYER, reviewer_rate=REVIEWER_INTERCEPT,
                 cc_rate=COMMON_CAUSE_RATE, verbose=False):
    conds = conditions(sid); res = {c: {"per_seed": []} for c in conds}
    for seed in seeds:
        gen_rng = np.random.default_rng(zlib.crc32(f"{sid}-{seed}".encode()))
        d = sc.GENERATORS[sid](model, seed, demands, gen_rng)
        crn = np.random.default_rng(zlib.crc32(f"crn-{sid}-{seed}".encode()))
        for x in d: x["u_rev"] = float(crn.random()); x["u_cc"] = float(crn.random())     # common random numbers
        for c, cfg in conds.items():
            res[c]["per_seed"].append({"seed": seed, **_run_config(d, cfg, monitor, f"{sid}-{c}-{seed}", reviewer_rate, cc_rate)})
        if verbose: print(f"  {sid} seed {seed} done ({len(d)} demands)", flush=True)
    for c in conds:
        ps = res[c]["per_seed"]; n = sum(p["n"] for p in ps); k = sum(p["harms"] for p in ps)
        res[c].update({"n": n, "harms": k, "pfd": k / n if n else None,
                       "seed_pfd_min": min(p["harms"] / p["n"] for p in ps), "seed_pfd_max": max(p["harms"] / p["n"] for p in ps),
                       "classes": _merge([p["classes"] for p in ps]), "stopped_by": _merge([p["stopped_by"] for p in ps])})
    return res

def _merge(ds):
    out = {}
    for d in ds:
        for k, v in d.items(): out[k] = out.get(k, 0) + v
    return out

def make_monitor():
    return ModelMonitor(syn.generate(6000, seed=777), window=WINDOW, psi_threshold=PSI_THRESHOLD)

def run_all(model, seeds, demands=DEMANDS_PER_LAYER, scenarios=None, verbose=True, **kw):
    mon = make_monitor(); out = {}
    for sid in (scenarios or sc.CREDITED):
        t0 = time.time()
        if sid in PIPELINE_SCENARIOS:
            out[sid] = run_scenario(sid, model, mon, seeds, demands, verbose=False, **kw)
        elif sid == "S6":
            rng = np.random.default_rng(6); out[sid] = {"recovery": faults.run_s6(demands * len(seeds), rng),
                                                      "no_fallback": faults.run_s6(demands * len(seeds), np.random.default_rng(6), False)}
        elif sid == "S7":
            out[sid] = {"recovery": faults.run_s7(demands * len(seeds), np.random.default_rng(7))}
        elif sid == "S8":
            out[sid] = {"C0": faults.run_s8(demands * len(seeds), np.random.default_rng(8), active=False),
                        "C1:PL5": faults.run_s8(demands * len(seeds), np.random.default_rng(8), active=True)}
        if verbose: print(f"{sid} {sc.NAMES[sid]}: {time.time()-t0:.1f}s", flush=True)
    return out
