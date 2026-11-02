"""Unit tests (stdlib unittest; also runs under pytest). Run: python -m unittest discover -s tests"""
import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from lopa_referral import synthetic as syn, lopa, validation, audit, access, monitor, oversight, schema
from lopa_referral.scenarios import row_dicts, COORD, CREDITED
from lopa_referral.pipeline import ReferralSystem, rule_urgency
from lopa_referral.model import smote, ece, tune_threshold, nonroutine_decision

def valid_ref():
    return row_dicts(syn.generate(1, seed=3))[0]

class TestValidation(unittest.TestCase):
    def test_valid_passes(self):
        for r in row_dicts(syn.generate(200, seed=1)): self.assertTrue(validation.validate(r)[0], r)
    def test_missing_range_malformed_contradiction(self):
        r = valid_ref(); r.pop("hr"); self.assertFalse(validation.validate(r)[0])
        r = valid_ref(); r["spo2"] = 140; self.assertIn("range:spo2", validation.validate(r)[1])
        r = valid_ref(); r["temp"] = "abc"; self.assertFalse(validation.validate(r)[0])
        r = valid_ref(); r.update(sex=0, pregnant=1); self.assertFalse(validation.validate(r)[0])

class TestAccess(unittest.TestCase):
    def test_rbac(self):
        ac = access.AccessControl()
        self.assertTrue(ac.check(COORD, "dispatch")[0])
        self.assertFalse(ac.check(access.Session("a", "auditor"), "dispatch")[0])
        self.assertFalse(ac.check(access.Session("a", "referral_coordinator", active=False), "submit")[0])
        self.assertFalse(ac.check(None, "submit")[0])
        self.assertFalse(ac.check(access.Session("a", "referral_coordinator", mfa_ok=False), "submit")[0])

class TestAudit(unittest.TestCase):
    def make(self, n=60):
        a = audit.AuditLog(25)
        for i in range(n): a.append("u", "dispatch", i // 3, {"i": i})
        return a
    def test_clean_verifies(self): self.assertTrue(self.make().verify()[0])
    def test_alteration_detected(self):
        a = self.make(); a.entries[10]["detail"] = {"i": -1}; self.assertFalse(a.verify()[0])
    def test_removal_detected(self):
        a = self.make(); del a.entries[20]; self.assertFalse(a.verify()[0])
    def test_trail(self): self.assertEqual(len(self.make().trail(2)), 3)

class TestLopa(unittest.TestCase):
    def test_clopper_pearson_zero_failures(self):
        lo, hi = lopa.clopper_pearson(0, 1000)
        self.assertEqual(lo, 0.0); self.assertAlmostEqual(hi, 0.00368, places=4)
        self.assertAlmostEqual(lopa.rule_of_three(1000), 3e-3)
    def test_zero_failure_uses_bound_not_zero(self):
        e = lopa.pfd_estimate(0, 1000); self.assertTrue(e["bounded_not_demonstrated"]); self.assertAlmostEqual(e["pfd_used"], 3e-3)
    def test_equation(self):
        r = lopa.scenario_lopa("S4", {"PL3": {"k": 100, "n": 1000}, "PL4": {"k": 200, "n": 1000}})
        self.assertAlmostEqual(r["f_mit"], r["f_init"] * 0.1 * 0.2); self.assertAlmostEqual(r["gap_ratio"], r["f_mit"] / r["f_tol"])
        self.assertLessEqual(r["mc_gap"]["p2.5"], r["mc_gap"]["p97.5"])

class TestMonitor(unittest.TestCase):
    def test_psi_detects_shift(self):
        ref = syn.generate(5000, 1); same = syn.generate(100, 2); shifted = syn.generate(100, 3, shift={"hr": 40, "sbp": -40})
        mon = monitor.ModelMonitor(ref)
        self.assertTrue(mon.window_drift(shifted)[0])
        self.assertFalse(mon.window_drift(same)[0])

class TestModelHelpers(unittest.TestCase):
    def test_smote_balances(self):
        X = np.random.default_rng(0).normal(size=(120, 4)); y = np.array([0] * 100 + [1] * 20)
        Xs, ys = smote(X, y); self.assertEqual(np.bincount(ys).tolist(), [100, 100])
    def test_ece_perfect(self): self.assertAlmostEqual(ece(np.ones(10), np.ones(10)), 0.0)
    def test_threshold_respects_cap(self):
        rng = np.random.default_rng(0); y = (rng.random(2000) < 0.2).astype(int)
        p1 = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.15, 2000), 0.01, 0.99)
        proba = np.column_stack([1 - p1, p1 / 2, p1 / 2]); tau = tune_threshold(proba, y)
        pred = nonroutine_decision(proba, tau) > 0; self.assertLessEqual(pred[y == 0].mean(), 0.25 + 1e-9)

class TestPipeline(unittest.TestCase):
    def mk(self, active, p=1.0):
        return ReferralSystem(active, oversight.SimulatedReviewer(p), None, np.random.default_rng(0))
    def demand(self, ref=None, harmful=True, **kw):
        rec = {"specialty": "cardiology", "urgency": 0, "confidence": 0.9, "model_version": "t"}
        d = {"id": 1, "ref": ref or valid_ref(), "rec": rec, "session": COORD, "op": "submit", "harmful": harmful,
             "subtle": False, "reviewable": True, "window_df": None}; d.update(kw); return d
    def test_pl1_blocks_invalid(self):
        r = valid_ref(); r["spo2"] = 140
        self.assertEqual(self.mk(["PL1"]).process(self.demand(r))["by"], "PL1")
    def test_unprotected_harm_reaches_dispatch(self):
        o = self.mk([]).process(self.demand()); self.assertTrue(o["harm"]); self.assertEqual(o["cls"], "undetected_harm")
    def test_audit_only_discovers_after_harm(self):
        o = self.mk(["PL5"]).process(self.demand()); self.assertTrue(o["harm"]); self.assertEqual(o["cls"], "discovered_after_harm")
    def test_pl4_intercepts_when_certain(self):
        o = self.mk(["PL4"], 1.0).process(self.demand()); self.assertFalse(o["harm"]); self.assertEqual(o["cls"], "detected_before_harm")
    def test_shared_dependency_disables_pl3_pl4(self):
        o = self.mk(["PL4"], 1.0).process(self.demand(), shared_down=True); self.assertTrue(o["harm"])
    def test_pl2_denies(self):
        o = self.mk(["PL2"]).process(self.demand(session=access.Session("x", "auditor"), op="dispatch")); self.assertEqual(o["by"], "PL2")
    def test_adding_layers_never_increases_harm(self):
        for u in (0.1, 0.5, 0.9):
            d = self.demand(u_rev=u); a = self.mk(["PL4"], 0.6).process(d)["harm"]; b = self.mk(["PL1", "PL4"], 0.6).process(d)["harm"]
            self.assertLessEqual(int(b), int(a))

class TestSchemaPaper(unittest.TestCase):
    def test_five_layers_and_scenarios(self):
        self.assertEqual(len(CREDITED), 10); self.assertTrue(all(L in ("PL1", "PL2", "PL3", "PL4", "PL5") for v in CREDITED.values() for L in v))
    def test_reference_set_size(self): self.assertEqual(len(syn.reference_set(200)), 200)
    def test_features_match_schema(self):
        self.assertEqual(list(syn.to_features(syn.generate(5, 1)).columns), schema.FEATURES)

if __name__ == "__main__":
    unittest.main()
