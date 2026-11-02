"""PL5 Audit mechanisms: tamper-evident, hash-chained log with periodic checkpoints.

No credit by default (a log documents a wrong referral but does not prevent it).
Detection of alteration/removal is measured in scenario S8. A checkpoint stores
(sequence, head hash) every ``checkpoint_every`` entries, standing in for an
independent anchor; entries removed after the last checkpoint are therefore
detectable only by sequence-gap checks against the consumer-side counter.
"""
from __future__ import annotations
import hashlib, json, time

class AuditLog:
    def __init__(self, checkpoint_every: int = 25):
        self.entries: list[dict] = []
        self.checkpoints: dict[int, str] = {}
        self.every = checkpoint_every
        self.expected_count = 0          # writers' own counter (completeness evidence)
    @staticmethod
    def _h(prev, body):
        return hashlib.sha256((prev + json.dumps(body, sort_keys=True, default=str)).encode()).hexdigest()
    def append(self, actor: str, action: str, ref_id, detail: dict | None = None, ts: float | None = None):
        prev = self.entries[-1]["hash"] if self.entries else "GENESIS"
        body = {"seq": len(self.entries), "ts": ts if ts is not None else time.time(),
                "actor": actor, "action": action, "ref": ref_id, "detail": detail or {}}
        e = {**body, "prev": prev, "hash": self._h(prev, body)}
        self.entries.append(e)
        self.expected_count += 1
        if (e["seq"] + 1) % self.every == 0:
            self.checkpoints[e["seq"]] = e["hash"]
    def verify(self):
        """Return (ok, problems). Checks hash chain, sequence gaps, checkpoints and completeness."""
        prev, problems = "GENESIS", []
        for i, e in enumerate(self.entries):
            body = {k: e[k] for k in ("seq", "ts", "actor", "action", "ref", "detail")}
            if e["prev"] != prev or e["hash"] != self._h(prev, body):
                problems.append(f"chain_break@{i}")
            if e["seq"] != i:
                problems.append(f"seq_gap@{i}")
            prev = e["hash"]
        for seq, h in self.checkpoints.items():
            if seq >= len(self.entries) or self.entries[seq]["hash"] != h:
                problems.append(f"checkpoint_mismatch@{seq}")
        if len(self.entries) != self.expected_count:
            problems.append("incomplete")
        return (not problems), problems
    def trail(self, ref_id):
        """Reconstruct the recorded history of one referral."""
        return [e for e in self.entries if e["ref"] == ref_id]
