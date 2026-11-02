"""Machine learning component (Section 3.5.3).

Two supervised tasks on structured referral features:
  * receiving specialty (multi-class)
  * urgency (routine / urgent / emergent)
Logistic regression is the baseline, a random forest the comparison model.
Imbalance strategies compared: none, class weighting, SMOTE-style oversampling
(training split only), and decision-threshold tuning. The selected configuration
is the one with the best urgent-case recall at an acceptable false-alarm rate.
"""
from __future__ import annotations
import hashlib, json, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .schema import SPECIALTIES, URGENCY
from .synthetic import to_features

FALSE_ALARM_CAP = 0.25     # max share of routine cases flagged urgent/emergent

def smote(X: np.ndarray, y: np.ndarray, seed: int = 0, k: int = 5):
    """Minimal SMOTE (Chawla et al., 2002): interpolate minority samples toward neighbours."""
    rng = np.random.default_rng(seed)
    counts = np.bincount(y); target = counts.max()
    Xs, ys = [X], [y]
    for c in np.flatnonzero(counts):
        need = target - counts[c]
        if need <= 0 or counts[c] < 2:
            continue
        Xc = X[y == c]
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xc))).fit(Xc)
        idx = nn.kneighbors(Xc, return_distance=False)[:, 1:]
        base = rng.integers(0, len(Xc), need)
        neigh = idx[base, rng.integers(0, idx.shape[1], need)]
        lam = rng.random((need, 1))
        Xs.append(Xc[base] + lam * (Xc[neigh] - Xc[base])); ys.append(np.full(need, c))
    return np.vstack(Xs), np.concatenate(ys)

def make_estimator(kind: str, class_weight=None, seed: int = 0):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=class_weight))
    if kind == "rf":
        return RandomForestClassifier(n_estimators=150, min_samples_leaf=3, class_weight=class_weight,
                                      random_state=seed, n_jobs=1)
    raise ValueError(kind)

def nonroutine_decision(proba: np.ndarray, tau: float) -> np.ndarray:
    nonr = 1.0 - proba[:, 0]
    flagged = nonr >= tau
    return np.where(flagged, 1 + np.argmax(proba[:, 1:], axis=1), 0)

def tune_threshold(proba, y_true, cap=FALSE_ALARM_CAP):
    """Best urgent-case recall subject to false-alarm rate <= cap (validation data)."""
    best_tau, best_rec = 0.5, -1.0
    pos, neg = y_true > 0, y_true == 0
    for tau in np.linspace(0.05, 0.9, 86):
        pred = nonroutine_decision(proba, tau) > 0
        far = pred[neg].mean() if neg.any() else 0.0
        rec = pred[pos].mean() if pos.any() else 0.0
        if far <= cap and rec > best_rec:
            best_tau, best_rec = float(tau), float(rec)
    return best_tau

def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error (Guo et al., 2017), top-label, equal-width bins."""
    edges = np.linspace(0, 1, bins + 1); total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)

class ReferralModel:
    def __init__(self, kind="rf", strategy="class_weight", seed=0, calibrate=False):
        self.kind, self.strategy, self.seed, self.calibrate = kind, strategy, seed, calibrate
        self.iso = None
        self.tau = 0.5
        self.spec_clf = self.urg_clf = None
        self.version = None
        self.metrics: dict = {}
    # ---- training
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None):
        X = to_features(train).to_numpy()
        ys = train["specialty"].map({s: i for i, s in enumerate(SPECIALTIES)}).to_numpy()
        yu = train["urgency"].to_numpy().astype(int)
        cw = "balanced" if self.strategy in ("class_weight",) else None
        self.spec_clf = make_estimator(self.kind, None, self.seed).fit(X, ys)
        Xu, yuu = (smote(X, yu, self.seed) if self.strategy == "smote" else (X, yu))
        self.urg_clf = make_estimator(self.kind, cw, self.seed).fit(Xu, yuu)
        if self.calibrate and valid is not None:       # isotonic calibration of P(urgent or emergent), validation data
            pv = self.urg_clf.predict_proba(to_features(valid).to_numpy())
            self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(1 - pv[:, 0], (valid["urgency"].to_numpy() > 0).astype(int))
        if self.strategy == "threshold" and valid is not None:
            pv = self._urg_proba(to_features(valid).to_numpy())
            self.tau = tune_threshold(pv, valid["urgency"].to_numpy().astype(int))
        sig = json.dumps({"k": self.kind, "s": self.strategy, "seed": self.seed, "n": len(train),
                          "tau": self.tau, "data": hashlib.md5(pd.util.hash_pandas_object(train).values.tobytes()).hexdigest()})
        self.version = f"{self.kind}-{self.strategy}-" + hashlib.sha1(sig.encode()).hexdigest()[:8]
        return self
    def _urg_proba(self, X):
        p = self.urg_clf.predict_proba(X)
        if self.iso is None: return p
        nonr = np.clip(self.iso.predict(1 - p[:, 0]), 1e-4, 1 - 1e-4)
        rest = p[:, 1:] / np.clip(p[:, 1:].sum(axis=1, keepdims=True), 1e-9, None)
        return np.column_stack([1 - nonr, rest * nonr[:, None]])
    # ---- inference
    def predict_batch(self, df: pd.DataFrame):
        X = to_features(df).to_numpy()
        ps = self.spec_clf.predict_proba(X); pu = self._urg_proba(X)
        spec_idx = self.spec_clf.classes_[np.argmax(ps, axis=1)]
        urg = nonroutine_decision(pu, self.tau)
        urg_conf = np.where(urg == 0, pu[:, 0], 1 - pu[:, 0])      # confidence in the decided side
        conf = np.minimum(ps.max(axis=1), urg_conf)
        return pd.DataFrame({"specialty": [SPECIALTIES[i] for i in spec_idx], "urgency": urg,
                             "confidence": conf, "p_routine": pu[:, 0], "model_version": self.version})
    def recommend_one(self, row: dict):
        t0 = time.perf_counter()
        out = self.predict_batch(pd.DataFrame([row])).iloc[0].to_dict()
        out["latency_s"] = time.perf_counter() - t0
        return out

def train_and_select(train, valid, seed=0, kinds=("logreg", "rf"),
                     strategies=("none", "class_weight", "smote", "threshold")):
    """Compare model x imbalance strategy on validation; return (best model, table)."""
    rows, models = [], {}
    yv = valid["urgency"].to_numpy().astype(int)
    for k in kinds:
        for s in strategies:
            m = ReferralModel(k, s, seed).fit(train, valid)
            pred = m.predict_batch(valid)["urgency"].to_numpy()
            pos, neg = yv > 0, yv == 0
            rec, far = (pred[pos] > 0).mean(), (pred[neg] > 0).mean()
            rows.append({"model": k, "strategy": s, "urgent_recall": rec, "false_alarm": far,
                         "specialty_acc": (m.predict_batch(valid)["specialty"].to_numpy() == valid["specialty"].to_numpy()).mean()})
            models[(k, s)] = m
    tab = pd.DataFrame(rows)
    ok = tab[tab.false_alarm <= FALSE_ALARM_CAP]
    pick = (ok if len(ok) else tab).sort_values(["urgent_recall", "specialty_acc"], ascending=False).iloc[0]
    return models[(pick.model, pick.strategy)], tab
