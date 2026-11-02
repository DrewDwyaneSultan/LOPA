"""API tests for the web interface (starts a real server on a free port)."""
import http.client, json, os, sys, threading, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from http.server import ThreadingHTTPServer
from lopa_referral import webapp

class TestWeb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        webapp.S = webapp.State()
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), webapp.Handler); cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
    @classmethod
    def tearDownClass(cls): cls.srv.shutdown()
    def call(self, method, path, body=None, token=""):
        c = http.client.HTTPConnection("127.0.0.1", self.port); h = {"Content-Type": "application/json", "X-Token": token}
        c.request(method, path, json.dumps(body) if body is not None else None, h); r = c.getresponse(); d = json.loads(r.read() or b"{}"); return r.status, d
    def login(self, role): return self.call("POST", "/api/login", {"role": role})[1]["token"]
    def sample(self):
        _, r = self.call("GET", "/api/sample"); return r
    def test_full_flow_and_roles(self):
        clin, coord, recv = self.login("referring_clinician"), self.login("referral_coordinator"), self.login("receiving_clinician")
        bad = self.sample(); bad["spo2"] = 140
        st, d = self.call("POST", "/api/referrals", {"ref": bad}, clin); self.assertEqual(st, 422); self.assertTrue(d["errors"])
        st, d = self.call("POST", "/api/referrals", {"ref": self.sample()}, clin); self.assertEqual(st, 200); rid = d["referral"]["id"]
        # clinician may not review or dispatch (PL2)
        self.assertEqual(self.call("POST", f"/api/referrals/{rid}/review", {"decision": "confirm"}, clin)[0], 403)
        self.assertEqual(self.call("POST", f"/api/referrals/{rid}/dispatch", {}, clin)[0], 403)
        if d["referral"]["status"] == "pending_review":      # PL4: confirm requires the source-check tick
            self.assertEqual(self.call("POST", f"/api/referrals/{rid}/review", {"decision": "confirm"}, coord)[0], 422)
            self.assertEqual(self.call("POST", f"/api/referrals/{rid}/review", {"decision": "confirm", "checked_source": True}, coord)[0], 200)
        else:
            self.assertEqual(self.call("POST", f"/api/referrals/{rid}/review", {"decision": "override", "specialty": "cardiology", "urgency": "urgent", "note": "manual triage"}, coord)[0], 200)
        self.assertEqual(self.call("POST", f"/api/referrals/{rid}/dispatch", {}, coord)[0], 200)
        self.assertEqual(self.call("POST", f"/api/referrals/{rid}/acknowledge", {}, recv)[0], 200)
    def test_audit_visible_only_to_auditor_and_chain_ok(self):
        self.assertEqual(self.call("GET", "/api/audit", token=self.login("referring_clinician"))[0], 403)
        aud = self.login("auditor"); st, v = self.call("GET", "/api/audit/verify", token=aud); self.assertEqual(st, 200); self.assertTrue(v["ok"])
    def test_no_session_denied(self): self.assertEqual(self.call("GET", "/api/referrals")[0], 403)
    def test_page_served(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port); c.request("GET", "/"); r = c.getresponse(); self.assertEqual(r.status, 200); self.assertIn(b"Referral Prototype", r.read())

if __name__ == "__main__": unittest.main()
