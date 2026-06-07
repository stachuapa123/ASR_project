"""Calibrated phone-noise augmentation: the P2G train/serve bridge.

P2G is served noisy CTC phones but trained on clean oracle phones; the mismatch
makes it collapse. We measure the CTC error profile on the VAL set and replay it
onto clean phones at train time so train phones match serve phones.

Profile = per-oracle-phone outcome distribution (correct/sub/del) + insertion
distribution, from a Levenshtein alignment of CTC predictions vs oracle phones.
``build_phone_noise.py`` writes ``data/phone_noise.json``.
"""

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


def align_ref_hyp(
    ref: list[str], hyp: list[str]
) -> list[tuple[str, str | None, str | None]]:
    """
    Levenshtein alignment of ``ref`` (oracle) vs ``hyp`` (predicted).

    Returns a list of ``(op, ref_sym, hyp_sym)`` where ``op`` is one of
    ``match`` / ``sub`` / ``del`` / ``ins`` and the irrelevant symbol is ``None``.
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    ops: list[tuple[str, str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1)
        ):
            op = "match" if ref[i - 1] == hyp[j - 1] else "sub"
            ops.append((op, ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


@dataclass
class PhoneNoiseProfile:
    """Empirical CTC error distribution; see module docstring."""

    sub: dict[str, dict[str, int]]  # oracle phone -> {predicted phone: count}
    correct: dict[str, int]  # oracle phone -> times left unchanged
    deletion: dict[str, int]  # oracle phone -> times dropped
    insertion: dict[str, int]  # spuriously inserted phone -> count
    n_ref: int  # total oracle phones (insertion-rate denominator)
    _compiled: dict | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def estimate(cls, pairs: list[tuple[list[str], list[str]]]) -> "PhoneNoiseProfile":
        """Build a profile from ``(ref_labels, hyp_labels)`` pairs."""
        sub: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        correct: dict[str, int] = defaultdict(int)
        deletion: dict[str, int] = defaultdict(int)
        insertion: dict[str, int] = defaultdict(int)
        n_ref = 0
        for ref, hyp in pairs:
            n_ref += len(ref)
            for op, r, h in align_ref_hyp(ref, hyp):
                if op == "match":
                    correct[r] += 1
                elif op == "sub":
                    sub[r][h] += 1
                elif op == "del":
                    deletion[r] += 1
                else:  # ins
                    insertion[h] += 1
        return cls(
            sub={k: dict(v) for k, v in sub.items()},
            correct=dict(correct),
            deletion=dict(deletion),
            insertion=dict(insertion),
            n_ref=n_ref,
        )

    # serialization
    def to_dict(self) -> dict:
        return {
            "sub": self.sub,
            "correct": self.correct,
            "deletion": self.deletion,
            "insertion": self.insertion,
            "n_ref": self.n_ref,
        }

    # deserialization
    @classmethod
    def from_dict(cls, d: dict) -> "PhoneNoiseProfile":
        return cls(
            sub=d["sub"],
            correct=d["correct"],
            deletion=d["deletion"],
            insertion=d["insertion"],
            n_ref=int(d["n_ref"]),
        )

    def expected_per(self) -> float:
        """PER this profile reproduces when replayed on clean phones (S+D+I)/N."""
        if self.n_ref == 0:
            return 0.0
        subs = sum(sum(v.values()) for v in self.sub.values())
        dels = sum(self.deletion.values())
        ins = sum(self.insertion.values())
        return (subs + dels + ins) / self.n_ref

    def _compile(self) -> dict:
        """Precompute per-phone outcome categoricals + a global fallback + ins choices."""
        per_phone: dict[str, tuple[list, list]] = {}
        g_choices: list = []
        g_weights: list = []
        phones = set(self.correct) | set(self.sub) | set(self.deletion)
        for p in phones:
            choices: list = [("keep", p)]
            weights: list = [self.correct.get(p, 0)]
            for h, c in self.sub.get(p, {}).items():
                choices.append(("sub", h))
                weights.append(c)
            choices.append(("del", None))
            weights.append(self.deletion.get(p, 0))
            if sum(weights) > 0:
                per_phone[p] = (choices, weights)
                for ch, w in zip(choices, weights):
                    g_choices.append(ch)
                    g_weights.append(w)
        ins_choices = list(self.insertion.keys())
        ins_weights = list(self.insertion.values())
        ins_rate = (sum(ins_weights) / self.n_ref) if self.n_ref else 0.0
        return {
            "per_phone": per_phone,
            "global": (g_choices, g_weights) if g_choices else None,
            "ins_choices": ins_choices,
            "ins_weights": ins_weights,
            "ins_rate": ins_rate,
        }

    def corrupt(self, labels: list[str], rng: random.Random) -> list[str]:
        """Replay the error profile onto a clean phone-label list."""
        if self._compiled is None:
            self._compiled = self._compile()
        c = self._compiled
        out: list[str] = []
        for lbl in labels:
            if c["ins_choices"] and rng.random() < c["ins_rate"]:
                out.append(rng.choices(c["ins_choices"], c["ins_weights"])[0])
            choices_weights = c["per_phone"].get(lbl) or c["global"]
            if choices_weights is None:
                out.append(lbl)
                continue
            choices, weights = choices_weights
            op, sym = rng.choices(choices, weights)[0]
            if op == "del":
                continue
            out.append(sym if op == "sub" else lbl)
        return out


def corrupt_phones(
    labels: list[str], profile: PhoneNoiseProfile, rng: random.Random
) -> list[str]:
    """Functional wrapper around :meth:`PhoneNoiseProfile.corrupt`."""
    return profile.corrupt(labels, rng)


def save_profile(path: str | Path, profile: PhoneNoiseProfile) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_profile(path: str | Path) -> PhoneNoiseProfile:
    return PhoneNoiseProfile.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
