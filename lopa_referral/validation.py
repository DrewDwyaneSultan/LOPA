"""PL1 Input validation (Table 5/7): field, range and consistency checks."""
from __future__ import annotations
import math
from .schema import REQUIRED_FIELDS, NUMERIC_RANGES, FLAG_FIELDS, REASONS, FLAG_FIELDS

def _num(v):
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None

def validate(ref: dict):
    """Return (ok, [errors]). Rejects missing, out-of-range, malformed and contradictory entries."""
    errs = []
    for f in REQUIRED_FIELDS:
        if f not in ref or ref[f] is None or (isinstance(ref[f], str) and not ref[f].strip()):
            errs.append(f"missing:{f}")
    if errs:
        return False, errs
    for f, (lo, hi) in NUMERIC_RANGES.items():
        v = _num(ref[f])
        if v is None:
            errs.append(f"malformed:{f}")
        elif not (lo <= v <= hi):
            errs.append(f"range:{f}")
    if ref["reason"] not in REASONS:
        errs.append("malformed:reason")
    if ref["sex"] not in (0, 1):
        errs.append("malformed:sex")
    for f in FLAG_FIELDS:
        if ref[f] not in (0, 1):
            errs.append(f"malformed:{f}")
    if errs:
        return False, errs
    age, sex = float(ref["age"]), ref["sex"]
    # consistency checks (contradictory entries)
    if ref["pregnant"] == 1 and (sex == 0 or age < 12 or age > 55):
        errs.append("inconsistent:pregnant")
    if ref["reason"] == "pregnancy_related" and sex == 0:
        errs.append("inconsistent:reason_sex")
    if ref["reason"] == "child_illness" and age > 18:
        errs.append("inconsistent:reason_age")
    if ref["reason"] == "chest_pain" and ref["chest_pain"] == 0:
        errs.append("inconsistent:reason_flag")
    if ref["reason"] == "injury" and ref["trauma"] == 0:
        errs.append("inconsistent:reason_flag")
    if float(ref["spo2"]) < 85 and ref["dyspnea"] == 0 and float(ref["rr"]) < 12:
        errs.append("inconsistent:vitals")
    return (not errs), errs
