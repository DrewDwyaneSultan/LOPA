"""PL3 Model monitoring: input-drift (PSI), per-recommendation OOD/confidence checks,
override/agreement tracking and a reference-sample calibration check (Sec 3.5.3 "Drift monitoring").

Agreement is not correctness, so the monitor also compares a periodic sample of
recommendations with reference-standard labels.
"""
from __future__ import annotations
import numpy as np
from .schema import VITALS

def psi(expected, actual, bins: int = 5) -> float:
    e = np.asarray(expected, float); a = np.asarray(actual, float)
    edges = np.quantile(e, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    pe = np.histogram(e, edges)[0] / len(e); pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))

class ModelMonitor:
    def __init__(self, ref_df, window: int = 100, psi_threshold: float = 0.2,
                 conf_floor: float = 0.40, z_limit: float = 4.0):
        self.cols = VITALS + ["age"]
        self.ref = {c: ref_df[c].to_numpy() for c in self.cols}
        self.mu = {c: ref_df[c].mean() for c in self.cols}
        self.sd = {c: ref_df[c].std() + 1e-9 for c in self.cols}
        self.window, self.psi_threshold = window, psi_threshold
        self.conf_floor, self.z_limit = conf_floor, z_limit
    # --- window level (drift)
    def window_drift(self, win_df):
        scores = {c: psi(self.ref[c], win_df[c].to_numpy()) for c in self.cols}
        worst = max(scores, key=scores.get)
        return scores[worst] > self.psi_threshold, scores[worst], worst
    # --- per recommendation (out-of-distribution / low confidence => escalate)
    def check_recommendation(self, row: dict, confidence: float):
        z = max(abs((float(row[c]) - self.mu[c]) / self.sd[c]) for c in self.cols)
        if confidence < self.conf_floor:
            return True, "low_confidence"
        if z > self.z_limit:
            return True, "out_of_distribution"
        return False, "ok"
    @staticmethod
    def agreement_rate(model_out, reviewer_out):
        return float(np.mean(np.asarray(model_out) == np.asarray(reviewer_out)))
    @staticmethod
    def sample_accuracy(model_out, reference_labels):
        return float(np.mean(np.asarray(model_out) == np.asarray(reference_labels)))
