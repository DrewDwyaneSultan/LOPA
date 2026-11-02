"""Referral form schema (Table 5: structured form with required fields).

Fields follow the kind of data a referral form carries (age, sex, reason
category, vital signs, key findings, comorbidity flags). The real partner
form is not yet available, so this is a generalized baseline that should be
replaced or confirmed against the PH eReferral Implementation Guide and the
partner's actual form once access is granted.
"""
from __future__ import annotations

SPECIALTIES = ["cardiology", "pulmonology", "neurology", "general_surgery",
               "obstetrics_gynecology", "pediatrics", "internal_medicine", "orthopedics"]
URGENCY = ["routine", "urgent", "emergent"]          # index = class id
REASONS = ["chest_pain", "breathing_difficulty", "neuro_symptoms", "abdominal_pain",
           "pregnancy_related", "child_illness", "chronic_disease_review", "injury"]

# numeric fields: (min plausible, max plausible) = hard validation range
NUMERIC_RANGES = {
    "age": (0, 110),
    "hr": (20, 220),        # heart rate, beats/min
    "sbp": (50, 260),       # systolic BP, mmHg
    "spo2": (50, 100),      # oxygen saturation, %
    "temp": (30.0, 43.0),   # deg C
    "rr": (4, 60),          # respiratory rate, breaths/min
}
FLAG_FIELDS = ["chest_pain", "dyspnea", "neuro_deficit", "bleeding", "trauma",
               "diabetes", "hypertension", "cardiac_hx", "pregnant", "immunocompromised"]
REQUIRED_FIELDS = ["age", "sex", "reason", "hr", "sbp", "spo2", "temp", "rr"] + FLAG_FIELDS
FEATURES = (["age", "sex", "hr", "sbp", "spo2", "temp", "rr"] + FLAG_FIELDS
            + [f"reason_{r}" for r in REASONS])
# vitals that tampering/drift tests are allowed to change
VITALS = ["hr", "sbp", "spo2", "temp", "rr"]

ROLES = {   # PL2 role-based access control (Table 5)
    "referring_clinician": {"submit", "view_record"},
    "referral_coordinator": {"submit", "review", "dispatch", "view_record"},
    "receiving_clinician": {"view_record", "acknowledge"},
    "auditor": {"view_audit", "view_record"},
    "ml_engineer": {"manage_model"},
    "admin": {"manage_roles"},
}
