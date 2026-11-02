"""Command line: python -m lopa_referral {demo,eval-model,run} [options]"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
from . import synthetic as syn, config as cfg
from .model import train_and_select
from .model_eval import evaluate
from . import harness, report

def _default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.ndarray,)): return o.tolist()
    return str(o)

def cmd_demo(a):
    """Push a few referrals through the full workflow with all five layers active."""
    from .pipeline import ReferralSystem
    from .oversight import SimulatedReviewer
    from .scenarios import row_dicts, COORD
    tr, va = syn.generate(8000, 1), syn.generate(3000, 2)
    model, _ = train_and_select(tr, va, kinds=("rf",), strategies=("threshold",))
    df = syn.generate(3, seed=42); recs = model.predict_batch(df).to_dict("records"); refs = row_dicts(df)
    bad = dict(refs[0]); bad["spo2"] = 140                                   # out-of-range entry
    sys_ = ReferralSystem(["PL1", "PL2", "PL3", "PL4", "PL5"], SimulatedReviewer(0.75), harness.make_monitor(), np.random.default_rng(0))
    for name, ref, rec in (("valid referral", refs[1], recs[1]), ("invalid SpO2=140", bad, recs[0])):
        d = {"id": name, "ref": ref, "rec": rec, "session": COORD, "op": "submit", "harmful": name != "valid referral", "subtle": False, "reviewable": True, "window_df": None}
        print(name, "->", sys_.process(d), "| suggestion:", rec["specialty"], URG[int(rec["urgency"])], f"(conf {rec['confidence']:.2f})")
    print("audit chain ok:", sys_.audit.verify()[0], "| entries:", len(sys_.audit.entries))
URG = ["routine", "urgent", "emergent"]

def cmd_eval(a):
    _, out = evaluate()
    print(json.dumps(out["targets"], indent=1, default=_default))

def cmd_run(a):
    t0 = time.time(); os.makedirs(a.out, exist_ok=True)
    print("1/4 train + evaluate model", flush=True)
    model, mev = evaluate()
    # evaluate() selects its own model; reuse the same training recipe for the harness
    tr, va = syn.generate(10000, 1), syn.generate(3000, 2)
    model, _ = train_and_select(tr, va, seed=0)
    est_seeds = cfg.SEEDS_ESTIMATE[: a.seeds]; val_seeds = cfg.SEEDS_VALIDATE[: a.seeds]
    print(f"2/4 harness, estimation seeds {est_seeds}, {a.demands} demands", flush=True)
    est = harness.run_all(model, est_seeds, a.demands)
    print(f"3/4 harness, validation seeds {val_seeds}", flush=True)
    val = harness.run_all(model, val_seeds, a.demands, scenarios=harness.PIPELINE_SCENARIOS)
    print("4/4 LOPA + report", flush=True)
    res = report.build(mev, est, val)
    json.dump({"results": res, "estimation": est, "validation": val, "config": {"demands": a.demands, "seeds": a.seeds}},
              open(os.path.join(a.out, "results.json"), "w"), indent=1, default=_default)
    open(os.path.join(a.out, "REPORT.md"), "w").write(report.markdown(res))
    print(f"done in {time.time()-t0:.0f}s -> {a.out}/REPORT.md")

def cmd_report(a):
    """Rebuild REPORT.md from a saved results.json (no re-running of tests)."""
    d = json.load(open(os.path.join(a.out, "results.json")))
    res = report.build(d["results"]["model"], d["estimation"], d["validation"])
    open(os.path.join(a.out, "REPORT.md"), "w").write(report.markdown(res)); print("rebuilt", a.out)

def main():
    ap = argparse.ArgumentParser(prog="lopa_referral")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo").set_defaults(f=cmd_demo)
    sub.add_parser("eval-model").set_defaults(f=cmd_eval)
    r = sub.add_parser("run"); r.add_argument("--demands", type=int, default=cfg.DEMANDS_PER_LAYER)
    r.add_argument("--seeds", type=int, default=5); r.add_argument("--out", default="outputs"); r.set_defaults(f=cmd_run)
    rp = sub.add_parser("report"); rp.add_argument("--out", default="outputs"); rp.set_defaults(f=cmd_report)
    a = ap.parse_args(); a.f(a)

if __name__ == "__main__":
    main()
