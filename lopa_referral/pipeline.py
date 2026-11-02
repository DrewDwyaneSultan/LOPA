"""Referral workflow under test (Table 5): entry -> validation -> recommendation ->
monitoring -> review -> dispatch -> audit, with each protection layer switchable.

Layers: PL1 input validation, PL2 access controls, PL3 model monitoring,
PL4 human oversight, PL5 audit. PLX is the *proposed* additional layer used in
condition C4: an independent rule-based triage cross-check that does not use the model.
"""
from __future__ import annotations
from .access import AccessControl, Session
from .audit import AuditLog
from .monitor import ModelMonitor
from .oversight import SimulatedReviewer
from .validation import validate
from .synthetic import REF_T

ALL_LAYERS = ("PL1", "PL2", "PL3", "PL4", "PL5")
PREVENTED, DETECTED, AFTER, UNDETECTED = "prevented", "detected_before_harm", "discovered_after_harm", "undetected_harm"

def rule_urgency(r: dict) -> int:
    """Independent clinical-rule triage (PLX). Scalar version of synthetic._score with the reference thresholds."""
    t = REF_T; s = 0
    s += (r["spo2"] < t["spo2_hi"]) + 2 * (r["spo2"] < t["spo2_lo"]) + (r["sbp"] < t["sbp_hi"]) + 2 * (r["sbp"] < t["sbp_lo"])
    s += (r["hr"] > t["hr_hi"]) + 2 * (r["hr"] > t["hr_lo"]) + (r["rr"] > t["rr_hi"]) + 2 * (r["rr"] > t["rr_lo"])
    s += (r["temp"] > t["temp_hi"]) + 2 * r["neuro_deficit"] + r["bleeding"] + r["chest_pain"] + r["dyspnea"]
    s += int(r["immunocompromised"] == 1 and r["temp"] > 38.0)
    return 2 if s >= 7 else 1 if s >= 4 else 0

class ReferralSystem:
    def __init__(self, active, reviewer: SimulatedReviewer, monitor: ModelMonitor | None, rng,
                 audit_checkpoint_every: int = 25):
        self.active = set(active)
        self.reviewer, self.monitor, self.rng = reviewer, monitor, rng
        self.access = AccessControl()
        self.audit = AuditLog(audit_checkpoint_every)

    def process(self, d: dict, disabled=(), shared_down: bool = False) -> dict:
        """Run one demand. ``d`` carries: ref, session, op, rec (model output), harmful, subtle,
        reviewable, window_df. ``disabled``: layers forced off (S10). ``shared_down``: common-cause
        failure of the model-output service, which takes PL3 and PL4 out together (S9)."""
        act = self.active - set(disabled)
        if shared_down:
            act -= {"PL3", "PL4"}
        rid = d.get("id", 0); rec = d["rec"]
        def log(action, **detail):
            if "PL5" in act:
                self.audit.append(d["session"].user if d.get("session") else "anon", action, rid, detail)
        # PL2 access control on the operation
        if "PL2" in act:
            ok, why = self.access.check(d.get("session"), d.get("op", "submit"))
            if not ok:
                log("denied", reason=why)
                return {"harm": False, "cls": PREVENTED, "by": "PL2"}
        log("submit")
        # PL1 input validation
        if "PL1" in act:
            ok, errs = validate(d["ref"])
            if not ok:
                log("rejected", errors=errs)
                return {"harm": False, "cls": PREVENTED, "by": "PL1"}
        log("recommend", specialty=rec["specialty"], urgency=int(rec["urgency"]), model=rec["model_version"])
        # PL3 model monitoring (window drift if a window is supplied, else per-recommendation checks)
        if "PL3" in act and self.monitor is not None:
            if d.get("window_df") is not None:
                flagged = self.monitor.window_drift(d["window_df"])[0]
            else:
                flagged = self.monitor.check_recommendation(d["ref"], float(rec["confidence"]))[0]
            if flagged and d["harmful"]:
                log("escalated_manual", by="PL3")
                return {"harm": False, "cls": DETECTED, "by": "PL3"}
        # PLX proposed layer: independent rule cross-check (only meaningful on valid-looking input)
        if "PLX" in act and d.get("reviewable", True):
            if (rule_urgency(d["ref"]) > 0) != (int(rec["urgency"]) > 0) and d["harmful"]:
                log("escalated_manual", by="PLX")
                return {"harm": False, "cls": DETECTED, "by": "PLX"}
        # PL4 human oversight
        if "PL4" in act and d.get("reviewable", True):
            if self.reviewer.intercepts(d["harmful"], self.rng, d.get("subtle", False), d.get("u_rev")):
                log("overridden", by="PL4")
                return {"harm": False, "cls": DETECTED, "by": "PL4"}
            log("confirmed")
        # dispatch
        log("dispatch")
        if d["harmful"]:
            return {"harm": True, "cls": AFTER if "PL5" in act else UNDETECTED, "by": None}
        return {"harm": False, "cls": None, "by": None}
