"""Assumptions made explicit. Every value here is a PROPOSED value from the paper
(Sections 3.2-3.8) or a placeholder to be replaced with Phase 1 / partner data."""
DEMANDS_PER_LAYER = 1000          # Sec 3.6.5: 1,000 demands per layer
SEEDS_ESTIMATE = [0, 1, 2, 3, 4]  # 5 random seeds (estimation)
SEEDS_VALIDATE = [100, 101, 102, 103, 104]   # separate seeds for EQ1 validation (non-circular)
REVIEWER_INTERCEPT = 0.75         # simulated reviewer; sweep 0.5-0.9 (Sec 3.6.5)
COMMON_CAUSE_RATE = 0.05          # S9: share of demands where the shared model-output service is down (ASSUMED)
WINDOW = 100                      # monitoring window (referrals)
PSI_THRESHOLD = 0.2
DELAY_WINDOW_HOURS = {"emergent": 24, "urgent": 72, "routine": 168}   # Sec 3.6.2 (panel reviews in Round 1)
# LOPA inputs (Sec 3.6.4): V x p. PLACEHOLDERS until the partner provides referral volume and Phase 1 error rates.
ANNUAL_REFERRALS = 12000          # ASSUMED, vary in sensitivity analysis
INITIATING_RATE = {               # p: probability that a referral carries the initiating event (ASSUMED)
    "S1": 0.05, "S2": 0.002, "S3": 0.001, "S4": 0.03, "S5": 0.01, "S9": 0.03, "S10": 0.001,
}
# provisional tolerable frequencies per year by consequence class (Sec 3.6.4)
TOLERABLE = {"lower": 1e-2, "intermediate": 1e-3, "highest": 1e-4}
CONSEQUENCE = {"S1": "intermediate", "S2": "highest", "S3": "highest", "S4": "intermediate",
               "S5": "intermediate", "S9": "highest", "S10": "highest"}
MC_DRAWS = 10_000
