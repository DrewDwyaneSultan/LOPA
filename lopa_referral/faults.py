"""Scenarios that are not run through the dispatch pipeline: S6, S7 (delay / recovery) and S8 (audit malfunction)."""
from __future__ import annotations
import time
import numpy as np
from .audit import AuditLog
from .config import DELAY_WINDOW_HOURS

URG = ["routine", "urgent", "emergent"]

def run_s6(n, rng, with_fallback=True):
    """EHR outage: recovery time and manual routing. Reported as recovery measures, not PFD."""
    rec, late, manual = [], 0, 0
    for _ in range(n):
        urg = rng.choice(URG, p=[0.8, 0.15, 0.05]); dur = rng.lognormal(np.log(6.0), 1.0)   # outage hours
        t = dur
        if with_fallback and rng.random() < 0.9:          # manual routing available (assumed 90%)
            t = min(dur, 0.25 + rng.uniform(0.5, 3.0)); manual += int(t < dur)
        rec.append(t); late += int(t > DELAY_WINDOW_HOURS[urg])
    rec = np.array(rec)
    return {"n": n, "mean_recovery_h": float(rec.mean()), "p95_recovery_h": float(np.quantile(rec, 0.95)),
            "share_beyond_window": late / n, "manual_routing_share": manual / n}

def run_s7(n, rng, p_ack=0.6, retries=3, retry_gap_h=2.0):
    """Missing acknowledgment from the receiving facility: retries, delay alerts, escalation."""
    tot_retry = alerts = esc = unresolved = 0
    for _ in range(n):
        urg = rng.choice(URG, p=[0.8, 0.15, 0.05]); t, acked, r = 0.0, False, 0
        while r < retries and not acked:
            t += retry_gap_h; r += 1; acked = rng.random() < p_ack
            if not acked and r >= 2: alerts += 1
        tot_retry += r
        if not acked:
            esc += 1; t += 4.0; acked = rng.random() < 0.9          # phone escalation succeeds 90% (assumed)
        unresolved += int((not acked) or t > DELAY_WINDOW_HOURS[urg])
    return {"n": n, "mean_retries": tot_retry / n, "delay_alert_rate": alerts / n,
            "escalation_rate": esc / n, "unresolved_or_late_share": unresolved / n}

S8_KINDS = ["alter", "alter_rehash", "remove_middle", "remove_tail", "interrupt"]
S8_W = [0.35, 0.10, 0.20, 0.20, 0.15]

def run_s8(n, rng, active=True, entries=60):
    """Audit malfunction: logs altered, removed or interrupted. Detection = verify() problems or a count
    mismatch against the application's own independent action counter (reconciliation)."""
    detected = 0; kinds = {}; recon_ms = []
    for _ in range(n):
        k = rng.choice(S8_KINDS, p=S8_W)
        if not active:
            kinds.setdefault(k, [0, 0])[1] += 1; continue
        log = AuditLog(25); performed = 0
        for i in range(entries):
            log.append("coord1", "dispatch", i // 3, {"i": i}); performed += 1
        if k == "alter":
            j = int(rng.integers(entries)); log.entries[j]["detail"] = {"i": -1}
        elif k == "alter_rehash":          # attacker rewrites the chain from j onward (consistent hashes)
            j = int(rng.integers(entries)); log.entries[j]["detail"] = {"i": -1}
            prev = log.entries[j - 1]["hash"] if j else "GENESIS"
            for e in log.entries[j:]:
                body = {x: e[x] for x in ("seq", "ts", "actor", "action", "ref", "detail")}
                e["prev"] = prev; e["hash"] = log._h(prev, body); prev = e["hash"]
        elif k == "remove_middle":
            del log.entries[int(rng.integers(1, entries - 1))]
        elif k == "remove_tail":
            cut = int(rng.integers(1, 11)); del log.entries[-cut:]
            if rng.random() < 0.1: log.expected_count -= cut        # attacker also resets the writers' counter
        else:                               # logging service interrupted: later actions are never logged
            cut = int(rng.integers(3, 15)); del log.entries[-cut:]; log.expected_count -= cut
        ok, problems = log.verify()
        found = (not ok) or (len(log.entries) != performed)         # reconciliation with the application counter
        t0 = time.perf_counter(); log.trail(entries // 6); recon_ms.append((time.perf_counter() - t0) * 1e3)
        detected += int(found); b = kinds.setdefault(k, [0, 0]); b[0] += int(found); b[1] += 1
    return {"n": n, "detected": detected, "detection_rate": detected / n,
            "undetected": n - detected, "by_kind": {k: {"detected": v[0], "n": v[1]} for k, v in kinds.items()},
            "reconstruction_ms_mean": float(np.mean(recon_ms)) if recon_ms else None}
