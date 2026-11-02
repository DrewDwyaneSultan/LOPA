"""Validation (EQ1-EQ3) and report writer (Section 3.8.4)."""
from __future__ import annotations
import json, time
import numpy as np
from . import lopa, scenarios as sc, config as cfg
from .harness import PIPELINE_SCENARIOS

def _pool(res, cond):
    return {"k": res[cond]["harms"], "n": res[cond]["n"]}

def build(model_eval_out, est, val, seed=0):
    """est / val: harness outputs on estimation / validation seeds."""
    rng = np.random.default_rng(seed)
    out = {"model": model_eval_out, "scenarios": {}, "EQ1": {}, "EQ2": {}, "EQ3": {}, "recovery": {}, "audit": {}}
    inputs = {}
    for sid in PIPELINE_SCENARIOS:
        cred = sc.CREDITED[sid]
        pfds = {L: _pool(est[sid], f"C1:{L}") for L in cred}
        inputs[sid] = pfds
        rec = lopa.scenario_lopa(sid, pfds, seed=seed)
        rec["name"] = sc.NAMES[sid]; rec["exclusions"] = sc.EXCLUSIONS.get(sid, {})
        out["scenarios"][sid] = rec
        # EQ1: predicted (from estimation seeds) vs observed (C2, separate validation seeds)
        obs = _pool(val[sid], "C2"); lo, hi = lopa.clopper_pearson(obs["k"], obs["n"])
        draws = np.prod([lopa.jeffreys_draws(p["k"], p["n"], cfg.MC_DRAWS, rng) for p in pfds.values()], axis=0)
        plo, phi = np.quantile(draws, [0.025, 0.975])
        pred = float(np.prod([lopa.pfd_estimate(p["k"], p["n"])["pfd_obs"] for p in pfds.values()]))
        # seed-to-seed spread: product of per-seed PFDs (estimation) vs per-seed observed harm (validation)
        per = [[(x["harms"] / x["n"]) for x in est[sid][f"C1:{L}"]["per_seed"]] for L in cred]
        pseed = [float(np.prod(v)) for v in zip(*per)]
        oseed = [x["harms"] / x["n"] for x in val[sid]["C2"]["per_seed"]]
        overlap = bool(min(pseed) <= max(oseed) and min(oseed) <= max(pseed))
        out["EQ1"][sid] = {"predicted_seed_range": [min(pseed), max(pseed)], "observed_seed_range": [min(oseed), max(oseed)],
                           "seed_ranges_overlap": overlap, "predicted": pred, "predicted_95": [float(plo), float(phi)],
                           "observed": obs["k"] / obs["n"], "observed_95_CP": [lo, hi],
                           "observed_inside_predicted_interval": bool(plo <= obs["k"] / obs["n"] <= phi),
                           "predicted_inside_observed_interval": bool(lo <= pred <= hi),
                           "C0_baseline": val[sid]["C0"]["harms"] / val[sid]["C0"]["n"]}
    out["sensitivity"] = lopa.sensitivity(inputs)
    # EQ2: independence in S9 (shared model-output service) and the S10 combined failure
    p3, p4 = (lopa.pfd_estimate(**_pool(est["S9"], f"C1:{L}"))["pfd_obs"] for L in ("PL3", "PL4"))
    j = val["S9"]["C5:common_cause"]; jo = j["harms"] / j["n"]; ind = p3 * p4
    out["EQ2"]["S9"] = {"independent_product": ind, "observed_joint": jo, "ratio": (jo / ind) if ind else None,
                        "common_cause_rate_assumed": cfg.COMMON_CAUSE_RATE}
    r10 = val["S10"]["C5:PL1+PL4_down"]; c2 = val["S10"]["C2"]
    out["EQ2"]["S10"] = {"harm_with_PL1_PL4_down": r10["harms"] / r10["n"], "harm_all_layers": c2["harms"] / c2["n"],
                         "note": "residual harm when two related safeguards are disabled; LOPA credit for them should be removed"}
    # EQ3: layer contribution (validation seeds): removal (C3) and addition (C4) effect on harm
    for sid in PIPELINE_SCENARIOS:
        base = val[sid]["C2"]["harms"] / val[sid]["C2"]["n"]; row = {"all_credited": base}
        for c, r in val[sid].items():
            if c.startswith("C3:"): row[f"removed_{c[3:]}"] = r["harms"] / r["n"]
            if c.startswith("C4:"): row["with_PLX_added"] = r["harms"] / r["n"]
        out["EQ3"][sid] = row
    out["EQ3"]["ranking_by_gap"] = lopa.ranking(out["scenarios"].values())
    for sid in ("S6", "S7"): out["recovery"][sid] = est[sid]
    out["audit"] = est["S8"]
    return out

def _f(x, p=3):
    return "n/a" if x is None else (f"{x:.{p}g}" if isinstance(x, float) else str(x))

def markdown(o) -> str:
    L = ["# LOPA prototype run report", "",
         "> **Read this first.** All results come from *synthetic* referrals with rule-generated labels and *scripted* fault injection. "
         "They show how the safeguards behave in controlled tests, not clinical performance. Annual volume, initiating rates and "
         "tolerable frequencies are **placeholders** (`config.py`) until Phase 1 data and the expert panel provide real values, so the absolute "
         "gap ratios are illustrative; the measured PFDs and the rankings are the substantive output.", ""]
    m = o["model"]
    L += ["## 1. Machine learning component (Table 6 targets)", f"Selected model: `{m['selected']}`", "",
          "| Measure | Value | Target | Status |", "|---|---|---|---|"]
    for k, v in m["targets"].items(): L.append(f"| {k} | {_f(v['value'])} | {v['target']} | {v['status']} |")
    rb = m["robustness"]
    L += ["", f"Robustness stress levels (recall drop, points): " + ", ".join(f"{k}={_f(v['drop_pts'],3)}" for k, v in rb.items() if k.startswith('eps')) +
          f"; malformed inputs rejected by PL1: {_f(rb['malformed_rejected_by_PL1'])}.",
          f"Drift monitor: detected {_f(m['drift']['share_detected'])} of injected shifts, false-alarm rate {_f(m['drift']['false_alarm_rate'])}.", ""]
    L += ["## 2. Measured PFDs (condition C1, estimation seeds)", "", "| Scenario | Layer | failures/demands | PFD | 95% Clopper-Pearson | note |", "|---|---|---|---|---|---|"]
    for sid, r in o["scenarios"].items():
        for lay, e in r["pfds"].items():
            L.append(f"| {sid} {r['name']} | {lay} | {e['k']}/{e['n']} | {_f(e['pfd_obs'])} | [{_f(e['ci_lo'])}, {_f(e['ci_hi'])}] | "
                     f"{'bounded only (<~3/n), not demonstrated' if e['bounded_not_demonstrated'] else ''} |")
    L += ["", "## 3. LOPA result per scenario", "", "| Scenario | f_init /yr | PFD product | f_mit /yr | f_tol /yr | gap ratio | MC 95% gap | tolerable? |", "|---|---|---|---|---|---|---|---|"]
    for sid, r in o["scenarios"].items():
        L.append(f"| {sid} | {_f(r['f_init'])} | {_f(r['prod_pfd'])} | {_f(r['f_mit'])} | {_f(r['f_tol'])} | {_f(r['gap_ratio'])} | "
                 f"[{_f(r['mc_gap']['p2.5'])}, {_f(r['mc_gap']['p97.5'])}] | {'yes' if r['tolerable'] else 'no'} |")
    s = o["sensitivity"]
    L += ["", f"Ranking by gap: {' > '.join(s['base_ranking'])}. With tolerable frequency x0.1 the ranking "
          f"{'changes' if s['f_tol_x0.1']['changed'] else 'is unchanged'}; with x10 it {'changes' if s['f_tol_x10']['changed'] else 'is unchanged'}.", "",
          "## 4. Validation questions", "", "**EQ1 - predicted vs observed harm (separate validation seeds)**", "",
          "| Scenario | predicted | observed | observed 95% CP | observed inside predicted 95%? | per-seed ranges overlap? | C0 baseline |", "|---|---|---|---|---|---|---|"]
    for sid, e in o["EQ1"].items():
        L.append(f"| {sid} | {_f(e['predicted'])} | {_f(e['observed'])} | [{_f(e['observed_95_CP'][0])}, {_f(e['observed_95_CP'][1])}] | "
                 f"{'yes' if e['observed_inside_predicted_interval'] else 'NO - not supported'} | "
                 f"{'yes' if e['seed_ranges_overlap'] else 'no'} ({_f(e['predicted_seed_range'][0])}-{_f(e['predicted_seed_range'][1])} vs {_f(e['observed_seed_range'][0])}-{_f(e['observed_seed_range'][1])}) | {_f(e['C0_baseline'])} |")
    e2 = o["EQ2"]["S9"]
    L += ["", "A *NO* above means the product-of-PFDs prediction is not supported for that scenario: the layers are not independent "
          "(for S1, PL1 and PL4 both miss the subtle faults, so their failures are positively correlated) or the demand mix varies between seeds. "
          "Dependent layers need conditional PFDs, not the simple product.", "",
          f"**EQ2 - independence (S9).** Independent product of PL3 and PL4 PFDs = {_f(e2['independent_product'])}; observed joint failure with the "
          f"shared model-output service down in {e2['common_cause_rate_assumed']:.0%} of demands = {_f(e2['observed_joint'])}; ratio = {_f(e2['ratio'])}. "
          f"S10: harm rises from {_f(o['EQ2']['S10']['harm_all_layers'])} (all layers) to {_f(o['EQ2']['S10']['harm_with_PL1_PL4_down'])} when PL1 and PL4 are disabled.", "",
          "**EQ3 - layer contribution (harm rate, lower is better)**", "", "| Scenario | all credited | layer removed | + proposed rule cross-check (PLX) |", "|---|---|---|---|"]
    for sid in PIPELINE_SCENARIOS:
        r = o["EQ3"][sid]; rem = ", ".join(f"{k[8:]}: {_f(v)}" for k, v in r.items() if k.startswith("removed_")) or "n/a"
        L.append(f"| {sid} | {_f(r['all_credited'])} | {rem} | {_f(r.get('with_PLX_added'))} |")
    L += ["", "## 5. Not credited: recovery and accountability measures", ""]
    for sid in ("S6", "S7"):
        L.append(f"- **{sid} {sc.NAMES[sid]}**: " + json.dumps({k: {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()} for k, v in o['recovery'][sid].items()}))
    a = o["audit"]["C1:PL5"]
    L += [f"- **S8 Audit malfunction (PL5)**: detection rate {_f(a['detection_rate'])} of {a['n']} injected malfunctions; "
          f"undetected {a['undetected']}; by kind {json.dumps(a['by_kind'])}; mean reconstruction {_f(a['reconstruction_ms_mean'])} ms. "
          "No credit by default: a log documents a wrong referral but does not prevent it.", "",
          "## 6. Layer exclusions recorded per scenario", ""]
    for sid, r in o["scenarios"].items():
        L.append(f"- {sid}: " + "; ".join(f"{k}: {v}" for k, v in r["exclusions"].items()))
    return "\n".join(L) + "\n"
