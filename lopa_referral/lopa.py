"""LOPA calculation (Sections 3.6.4-3.8.4).

    f_mit = f_init x PFD_1 x ... x PFD_n          (Eq. 2)
    gap   = f_mit / f_tol                           (Eq. 3; > 1 means the risk is not tolerable)
    PFD   = failures / demands                      (Eq. 4)

Intervals: Clopper-Pearson 95%. With zero failures the upper bound is ~3/n (rule of three), so
with 1,000 demands only "below ~3e-3" is claimed; that bound (not 0) is used in the calculation.
Uncertainty: Beta(k+0.5, n-k+0.5) Jeffreys draws, 10,000 Monte Carlo samples.
Sensitivity: tolerable frequency x0.1 and x10, report whether the gap ranking changes.
"""
from __future__ import annotations
import numpy as np
from scipy import stats
from . import config as cfg

def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)

def rule_of_three(n: int) -> float:
    return 3.0 / n

def pfd_estimate(k: int, n: int):
    lo, hi = clopper_pearson(k, n)
    bounded = (k == 0)
    return {"k": k, "n": n, "pfd_obs": k / n, "pfd_used": rule_of_three(n) if bounded else k / n,
            "ci_lo": lo, "ci_hi": hi, "bounded_not_demonstrated": bounded}

def jeffreys_draws(k, n, size, rng):
    return rng.beta(k + 0.5, n - k + 0.5, size)

def scenario_lopa(sid, pfds: dict, mc_draws=cfg.MC_DRAWS, seed=0, v=None, f_tol=None):
    """pfds: layer -> {k, n}. Returns the LOPA record for one scenario."""
    v = v if v is not None else cfg.ANNUAL_REFERRALS
    f_init = v * cfg.INITIATING_RATE[sid]
    f_tol = f_tol if f_tol is not None else cfg.TOLERABLE[cfg.CONSEQUENCE[sid]]
    est = {L: pfd_estimate(x["k"], x["n"]) for L, x in pfds.items()}
    prod = float(np.prod([e["pfd_used"] for e in est.values()])) if est else 1.0
    f_mit = f_init * prod
    rng = np.random.default_rng(seed)
    if est:
        draws = np.prod([jeffreys_draws(x["k"], x["n"], mc_draws, rng) for x in pfds.values()], axis=0) * f_init
    else:
        draws = np.full(mc_draws, f_init)
    p5 = np.quantile(draws, [0.025, 0.5, 0.975])
    return {"scenario": sid, "f_init": f_init, "f_tol": f_tol, "pfds": est, "prod_pfd": prod, "f_mit": f_mit,
            "gap_ratio": f_mit / f_tol, "tolerable": f_mit <= f_tol,
            "mc_f_mit": {"p2.5": float(p5[0]), "p50": float(p5[1]), "p97.5": float(p5[2])},
            "mc_gap": {"p2.5": float(p5[0] / f_tol), "p50": float(p5[1] / f_tol), "p97.5": float(p5[2] / f_tol)},
            "prob_not_tolerable": float(np.mean(draws > f_tol)),
            "credit_applied": sorted(pfds)}

def ranking(records, key="gap_ratio"):
    return [r["scenario"] for r in sorted(records, key=lambda r: -r[key])]

def sensitivity(records_inputs):
    """records_inputs: {sid: pfds}. Re-run with f_tol x0.1 / x10; report ranking changes."""
    base = {sid: scenario_lopa(sid, p) for sid, p in records_inputs.items()}
    out = {"base_ranking": ranking(base.values())}
    for name, mult in (("f_tol_x0.1", 0.1), ("f_tol_x10", 10.0)):
        alt = {sid: scenario_lopa(sid, p, f_tol=base[sid]["f_tol"] * mult) for sid, p in records_inputs.items()}
        out[name] = {"ranking": ranking(alt.values()), "changed": ranking(alt.values()) != out["base_ranking"],
                     "gaps": {s: a["gap_ratio"] for s, a in alt.items()}}
    return out
