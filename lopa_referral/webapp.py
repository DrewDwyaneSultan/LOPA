"""Web interface for the referral prototype (standard library only, no extra installs).

    python -m lopa_referral serve            # then open http://127.0.0.1:8000

Screens follow the use-case flow of the paper: sign in -> referral entry (PL1 validation)
-> review (PL3 flag + PL4 confirm/override) -> tracker and dispatch (PL2 roles) -> audit log (PL5)
-> model monitoring (PL3) -> risk analysis (read-only LOPA results). Accounts are demo accounts:
real authentication is future work, and all data is synthetic and held in memory only.
"""
from __future__ import annotations
import json, os, re, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
from . import synthetic as syn
from .access import AccessControl, Session
from .audit import AuditLog
from .model import train_and_select
from .monitor import ModelMonitor
from .scenarios import row_dicts
from .schema import (ROLES, SPECIALTIES, URGENCY, REASONS, NUMERIC_RANGES, FLAG_FIELDS, VITALS)
from .validation import validate

WEB = Path(__file__).parent / "web"
ROOT = Path(__file__).resolve().parent.parent
LABELS = {"age": "Age", "sex": "Sex", "reason": "Reason for referral", "hr": "Heart rate", "sbp": "Systolic blood pressure",
          "spo2": "Oxygen saturation", "temp": "Temperature", "rr": "Respiratory rate", "chest_pain": "Chest pain",
          "dyspnea": "Shortness of breath", "neuro_deficit": "Neurological deficit", "bleeding": "Bleeding", "trauma": "Trauma / injury",
          "diabetes": "Diabetes", "hypertension": "Hypertension", "cardiac_hx": "Heart disease history", "pregnant": "Pregnant",
          "immunocompromised": "Weakened immune system"}
UNITS = {"age": "years", "hr": "beats/min", "sbp": "mmHg", "spo2": "%", "temp": "°C", "rr": "breaths/min"}
NORMAL = {"hr": "60–100", "sbp": "100–140", "spo2": "95–100", "temp": "36.1–37.5", "rr": "12–20"}
REASON_LABELS = {"chest_pain": "Chest pain", "breathing_difficulty": "Breathing difficulty", "neuro_symptoms": "Neurological symptoms",
                 "abdominal_pain": "Abdominal pain", "pregnancy_related": "Pregnancy-related", "child_illness": "Child illness",
                 "chronic_disease_review": "Chronic disease review", "injury": "Injury"}
INCONSISTENT = {"pregnant": "“Pregnant” does not match the sex or age entered.",
                "reason_sex": "A pregnancy-related referral needs the sex to be female.",
                "reason_age": "“Child illness” does not match an age over 18.",
                "reason_flag": "The reason for referral and the findings ticked below do not match (e.g. chest pain / injury).",
                "vitals": "Very low oxygen saturation with no breathing difficulty and a normal breathing rate looks inconsistent. Please re-check."}
USERS = {"referring_clinician": "Dr. Ana Reyes (referring clinician)", "referral_coordinator": "Mark Santos (referral coordinator)",
         "receiving_clinician": "Dr. Liza Cruz (receiving clinician)", "auditor": "Ben Ocampo (auditor)",
         "ml_engineer": "Kim Dela Cruz (ML engineer)", "admin": "Joy Tan (administrator)"}

def explain(err: str) -> str:
    kind, _, f = err.partition(":")
    if kind == "missing": return f"Please fill in: {LABELS.get(f, f)}."
    if kind == "range":
        lo, hi = NUMERIC_RANGES[f]; return f"{LABELS[f]} must be between {lo} and {hi} {UNITS.get(f, '')}.".replace("  ", " ")
    if kind == "malformed": return f"{LABELS.get(f, f)} is not a valid entry."
    if kind == "inconsistent": return INCONSISTENT.get(f, "These entries contradict each other.")
    return err

class State:
    def __init__(self):
        t0 = time.time()
        tr, va = syn.generate(8000, 1), syn.generate(3000, 2)
        self.model, _ = train_and_select(tr, va, kinds=("rf",), strategies=("threshold",))
        self.ref_df = syn.generate(6000, seed=777)
        self.monitor = ModelMonitor(self.ref_df)
        self.access = AccessControl(); self.audit = AuditLog(25)
        self.referrals: dict[str, dict] = {}; self.sessions: dict[str, Session] = {}
        self.window: list[dict] = []; self.lock = threading.Lock()
        self.agree = self.total_review = 0
        print(f"model ready ({self.model.version}) in {time.time()-t0:.1f}s", flush=True)

S: State | None = None

def now(): return time.strftime("%Y-%m-%d %H:%M:%S")

def clean(ref: dict) -> dict:
    out = {}
    for k, v in ref.items():
        if k in NUMERIC_RANGES or k in ("sex", *FLAG_FIELDS):
            if isinstance(v, str):
                v = v.strip()
                if v == "": v = None
                else:
                    try: v = float(v) if k in NUMERIC_RANGES else int(float(v))
                    except ValueError: pass
        out[k] = v
    for k in FLAG_FIELDS:
        if out.get(k) is True: out[k] = 1
        if out.get(k) is False: out[k] = 0
    return out

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    # ---- helpers
    def send(self, code, obj, ctype="application/json"):
        body = obj if isinstance(obj, (bytes, str)) else json.dumps(obj, default=str)
        body = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype + "; charset=utf-8" if ctype.startswith("text") or ctype == "application/json" else ctype)
        self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        try: return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError: return {}
    def session(self): return S.sessions.get(self.headers.get("X-Token", ""))
    def need(self, perm):
        s = self.session(); ok, why = S.access.check(s, perm)
        if not ok:
            with S.lock: S.audit.append(s.user if s else "anonymous", "denied", None, {"permission": perm, "reason": why})
            msg = {"no_active_session": "Please sign in first.", "missing_privilege": "Your role is not allowed to do this.",
                   "mfa_required": "Verification required."}[why]
            self.send(403, {"error": msg, "reason": why}); return None
        return s
    # ---- routes
    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"): return self.send(200, (WEB / "index.html").read_bytes(), "text/html")
        if p == "/api/meta": return self.send(200, self.meta())
        if p == "/api/me":
            s = self.session(); return self.send(200, {"user": s.user, "role": s.role, "perms": sorted(ROLES[s.role])} if s else {"user": None})
        if p == "/api/referrals":
            if not self.need("view_record"): return
            with S.lock: return self.send(200, sorted(S.referrals.values(), key=lambda r: r["id"], reverse=True))
        if p == "/api/audit":
            if not self.need("view_audit"): return
            with S.lock: return self.send(200, {"entries": S.audit.entries[-300:][::-1], "total": len(S.audit.entries)})
        if p == "/api/audit/verify":
            if not self.need("view_audit"): return
            with S.lock: ok, problems = S.audit.verify(); return self.send(200, {"ok": ok, "problems": problems, "entries": len(S.audit.entries)})
        if p == "/api/monitor": return self.send(200, self.monitor_status())
        if p == "/api/lopa": return self.send(200, self.lopa())
        if p == "/api/sample":
            d = syn.generate(1, seed=int(time.time()) % 100000); r = row_dicts(d)[0]
            for k in ("urgency", "specialty"): r.pop(k, None)
            return self.send(200, r)
        self.send(404, {"error": "not found"})
    def do_POST(self):
        p, b = self.path.split("?")[0], self.body()
        if p == "/api/login":
            role = b.get("role")
            if role not in USERS: return self.send(400, {"error": "unknown role"})
            tok = uuid.uuid4().hex; S.sessions[tok] = Session(USERS[role].split(" (")[0], role)
            with S.lock: S.audit.append(S.sessions[tok].user, "login", None, {"role": role})
            return self.send(200, {"token": tok, "user": S.sessions[tok].user, "role": role, "perms": sorted(ROLES[role])})
        if p == "/api/validate":
            ok, errs = validate(clean(b.get("ref", {}))); return self.send(200, {"ok": ok, "errors": [explain(e) for e in errs], "codes": errs})
        if p == "/api/referrals":
            s = self.need("submit")
            if not s: return
            return self.submit(s, clean(b.get("ref", {})))
        m = re.match(r"^/api/referrals/(R-\d+)/(review|dispatch|acknowledge)$", p)
        if m:
            perm = {"review": "review", "dispatch": "dispatch", "acknowledge": "acknowledge"}[m.group(2)]
            s = self.need(perm)
            if not s: return
            return getattr(self, m.group(2))(s, m.group(1), b)
        if p == "/api/simulate-drift":
            if not self.need("manage_model"): return
            rows = row_dicts(syn.generate(60, seed=int(time.time()) % 10000, shift={"hr": 24, "sbp": -28, "spo2": -4, "rr": 6, "age": 15}))
            with S.lock: S.window = (S.window + rows)[-100:]; S.audit.append("ml_engineer", "simulate_drift", None, {"rows": 60})
            return self.send(200, self.monitor_status())
        if p == "/api/reset-monitor":
            if not self.need("manage_model"): return
            with S.lock: S.window = []
            return self.send(200, self.monitor_status())
        self.send(404, {"error": "not found"})
    # ---- actions
    def submit(self, s, ref):
        ok, errs = validate(ref)                                    # PL1
        with S.lock:
            if not ok:
                S.audit.append(s.user, "rejected", None, {"errors": errs})
                return self.send(422, {"ok": False, "errors": [explain(e) for e in errs], "codes": errs})
            rid = f"R-{len(S.referrals)+1:04d}"
            rec = S.model.recommend_one(ref)
            flagged, why = S.monitor.check_recommendation(ref, float(rec["confidence"]))          # PL3
            rec["specialty_label"] = rec["specialty"].replace("_", " ").title(); rec["urgency_label"] = URGENCY[int(rec["urgency"])]
            r = {"id": rid, "created": now(), "by": s.user, "ref": ref, "ai": {k: (float(v) if isinstance(v, (np.floating,)) else (int(v) if isinstance(v, np.integer) else v)) for k, v in rec.items()},
                 "flagged": bool(flagged), "flag_reason": why if flagged else None,
                 "status": "needs_manual_triage" if flagged else "pending_review", "final": None, "history": [[now(), s.user, "Submitted"]]}
            S.referrals[rid] = r; S.window = (S.window + [ref])[-100:]
            S.audit.append(s.user, "submit", rid); S.audit.append("model", "recommend", rid, {"specialty": rec["specialty"], "urgency": int(rec["urgency"]), "model": rec["model_version"], "flagged": bool(flagged)})
            if flagged: S.audit.append("monitor", "escalated_manual", rid, {"why": why})
        self.send(200, {"ok": True, "referral": r})
    def review(self, s, rid, b):
        with S.lock:
            r = S.referrals.get(rid)
            if not r or r["status"] not in ("pending_review", "needs_manual_triage"): return self.send(409, {"error": "This referral is not waiting for review."})
            if b.get("decision") == "override" or r["status"] == "needs_manual_triage":
                sp, ur, note = b.get("specialty"), b.get("urgency"), (b.get("note") or "").strip()
                if sp not in SPECIALTIES or ur not in URGENCY: return self.send(422, {"error": "Choose a specialty and an urgency."})
                if len(note) < 5: return self.send(422, {"error": "Please add a short reason for your decision."})
                final = {"specialty": sp, "urgency": ur, "note": note, "decision": "override" if r["status"] == "pending_review" else "manual_triage"}
            else:
                final = {"specialty": r["ai"]["specialty"], "urgency": r["ai"]["urgency_label"], "note": (b.get("note") or "").strip(), "decision": "confirm"}
                if not b.get("checked_source"): return self.send(422, {"error": "Please tick that you checked the entered details before confirming."})
            same = final["specialty"] == r["ai"]["specialty"] and final["urgency"] == r["ai"]["urgency_label"]
            S.total_review += 1; S.agree += int(same)
            r["final"] = final; r["status"] = "reviewed"; r["history"].append([now(), s.user, {"confirm": "Confirmed AI suggestion", "override": "Overrode AI suggestion", "manual_triage": "Triaged manually"}[final["decision"]]])
            S.audit.append(s.user, {"confirm": "confirmed", "override": "overridden", "manual_triage": "manual_triage"}[final["decision"]], rid, {"ai": [r["ai"]["specialty"], r["ai"]["urgency_label"]], "final": [final["specialty"], final["urgency"]]})
        self.send(200, {"ok": True, "referral": r})
    def dispatch(self, s, rid, b):
        with S.lock:
            r = S.referrals.get(rid)
            if not r or r["status"] != "reviewed": return self.send(409, {"error": "Only reviewed referrals can be sent."})
            r["status"] = "dispatched"; r["history"].append([now(), s.user, f"Sent to {r['final']['specialty'].replace('_', ' ')}"]); S.audit.append(s.user, "dispatch", rid)
        self.send(200, {"ok": True, "referral": r})
    def acknowledge(self, s, rid, b):
        with S.lock:
            r = S.referrals.get(rid)
            if not r or r["status"] != "dispatched": return self.send(409, {"error": "Nothing to acknowledge yet."})
            r["status"] = "acknowledged"; r["history"].append([now(), s.user, "Receipt acknowledged"]); S.audit.append(s.user, "acknowledge", rid)
        self.send(200, {"ok": True, "referral": r})
    # ---- data
    def meta(self):
        return {"reasons": [{"id": r, "label": REASON_LABELS[r]} for r in REASONS], "specialties": SPECIALTIES, "urgency": URGENCY,
                "vitals": [{"id": v, "label": LABELS[v], "unit": UNITS[v], "min": NUMERIC_RANGES[v][0], "max": NUMERIC_RANGES[v][1], "normal": NORMAL[v]} for v in VITALS],
                "findings": [{"id": f, "label": LABELS[f]} for f in ("chest_pain", "dyspnea", "neuro_deficit", "bleeding", "trauma")],
                "history": [{"id": f, "label": LABELS[f]} for f in ("diabetes", "hypertension", "cardiac_hx", "pregnant", "immunocompromised")],
                "age": {"min": NUMERIC_RANGES["age"][0], "max": NUMERIC_RANGES["age"][1]},
                "users": [{"role": r, "name": n} for r, n in USERS.items()], "model": S.model.version}
    def monitor_status(self):
        with S.lock:
            win = list(S.window); n = len(win)
            out = {"window_size": n, "reviews": S.total_review, "agreement": (S.agree / S.total_review) if S.total_review else None,
                   "override_rate": (1 - S.agree / S.total_review) if S.total_review else None, "model": S.model.version, "psi": None, "status": "collecting"}
            if n >= 30:
                import pandas as pd
                drift, worst, var = S.monitor.window_drift(pd.DataFrame(win))
                out.update({"psi": worst, "worst": LABELS.get(var, var), "status": "drift" if drift else "stable", "threshold": S.monitor.psi_threshold})
        return out
    def lopa(self):
        for p in (ROOT / "outputs" / "results.json", ROOT / "docs" / "sample_run" / "results.json"):
            if p.exists():
                d = json.loads(p.read_text())["results"]
                return {"source": str(p.relative_to(ROOT)), "targets": d["model"]["targets"],
                        "scenarios": [{"id": k, "name": v["name"], "pfds": {l: {"pfd": e["pfd_obs"], "lo": e["ci_lo"], "hi": e["ci_hi"]} for l, e in v["pfds"].items()},
                                       "f_mit": v["f_mit"], "f_tol": v["f_tol"], "gap": v["gap_ratio"]} for k, v in d["scenarios"].items()],
                        "eq2": d["EQ2"]["S9"]}
        return {"error": "No results yet. Run: python -m lopa_referral run"}

def serve(host="127.0.0.1", port=8000, open_browser=False):
    global S
    S = State()
    srv = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"; print(f"Referral prototype running at {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        import webbrowser; webbrowser.open(url)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
