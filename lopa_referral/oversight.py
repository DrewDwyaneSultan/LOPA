"""PL4 Human oversight: a reviewer sees the suggestion and confirms or overrides.

Independence is limited: people over-rely on automated advice (automation bias),
so interception rate is a parameter (0.5-0.9 in the simulated-reviewer test, Sec 3.6.5)
and is reduced when the reviewer sees only the model output and not the source fields.
Real blinded reviewers (3 x 40 cases) replace the simulation if ethics allows.
"""
from __future__ import annotations

class SimulatedReviewer:
    def __init__(self, interception_rate: float = 0.75, sees_source_fields: bool = True,
                 model_only_penalty: float = 0.6):
        self.p = interception_rate
        self.sees_source = sees_source_fields
        self.penalty = model_only_penalty
    def effective_rate(self, subtle: bool = False) -> float:
        p = self.p if self.sees_source else self.p * self.penalty
        return p * 0.6 if subtle else p        # subtle/plausible errors are harder to catch
    def intercepts(self, wrong: bool, rng, subtle: bool = False, u: float | None = None) -> bool:
        """True if the reviewer stops a wrong suggestion (never 'intercepts' a correct one).
        ``u`` is a pre-drawn uniform for this demand (common random numbers across conditions, so
        adding a layer can never raise harm by chance alone); falls back to ``rng`` if omitted."""
        draw = u if u is not None else rng.random()
        return bool(wrong and draw < self.effective_rate(subtle))
