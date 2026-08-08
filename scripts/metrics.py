#!/usr/bin/env python3
"""
scripts/metrics.py

Evaluation metrics, defined once and shared by training, inference and
reporting so the three can never drift apart.

Convention (Sec. V-A / Tables II-V): WER and CER are *corpus level* --
total edit distance over the corpus divided by the total number of reference
words (resp. characters). Sub/Del/Ins are likewise normalised by the total
reference word count, so they sum exactly to the total WER.

A per-utterance (macro) average is a different quantity and gives noticeably
different numbers on a 50-utterance set; use `macro_wer` only if you
explicitly want it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


# -----------------------------------------------------------------------------
# Edit distance with error-type backtrace
# -----------------------------------------------------------------------------
def edit_counts(ref: Sequence[str], hyp: Sequence[str]) -> Tuple[int, int, int]:
    """Return (substitutions, deletions, insertions) aligning hyp to ref."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    sub = dele = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] != hyp[j - 1]:
                sub += 1
            i -= 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return sub, dele, ins


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, len(b) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if a[i - 1] == b[j - 1] else 1))
            prev = cur
    return dp[len(b)]


# -----------------------------------------------------------------------------
# Corpus-level metrics
# -----------------------------------------------------------------------------
@dataclass
class Scores:
    wer: float
    cer: float
    exact_match: float  # percent
    sub: float
    dele: float
    ins: float
    # Character-level split, normalised by the reference character count so
    # csub + cdel + cins == cer exactly. Unlike the word-level split this is
    # not degenerate on a fixed-arity corpus: reference lengths run from 6 to
    # 19 characters because month names differ in length, so a wrong month is
    # a genuine insertion or deletion rather than a substitution.
    csub: float
    cdel: float
    cins: float
    n_utt: int
    n_words: int
    n_chars: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "WER": self.wer,
            "CER": self.cer,
            "ExactMatch": self.exact_match,
            "Sub": self.sub,
            "Del": self.dele,
            "Ins": self.ins,
            "cSub": self.csub,
            "cDel": self.cdel,
            "cIns": self.cins,
            "n_utt": self.n_utt,
        }


def score_corpus(refs: Sequence[str], hyps: Sequence[str], tt=None) -> Scores:
    """Corpus WER/CER/exact-match plus the WER decomposition (Table V)."""
    if len(refs) != len(hyps):
        raise ValueError(f"refs/hyps length mismatch: {len(refs)} vs {len(hyps)}")

    clean = (lambda s: tt.clean(s)) if tt is not None else (lambda s: (s or "").strip())

    S = D = I = 0
    cS = cD = cI = 0
    cerr = 0
    nw = nc = 0
    exact = 0

    for r, h in zip(refs, hyps):
        r = clean(r)
        h = clean(h)
        rw, hw = r.split(), h.split()
        s, d, i = edit_counts(rw, hw)
        S += s
        D += d
        I += i
        nw += len(rw)
        # CER ignores word boundaries: spaces are stripped from both sides, so
        # a purely segmentation-level error costs WER but not CER. Verified to
        # reproduce all seven CER values of Table II exactly.
        rc, hc = r.replace(" ", ""), h.replace(" ", "")
        cerr += edit_distance(list(rc), list(hc))
        cs, cd, ci = edit_counts(list(rc), list(hc))
        cS += cs
        cD += cd
        cI += ci
        nc += len(rc)
        exact += int(r == h)

    nw = max(nw, 1)
    nc = max(nc, 1)
    n = max(len(refs), 1)
    return Scores(
        wer=(S + D + I) / nw,
        cer=cerr / nc,
        exact_match=100.0 * exact / n,
        sub=S / nw,
        dele=D / nw,
        ins=I / nw,
        csub=cS / nc,
        cdel=cD / nc,
        cins=cI / nc,
        n_utt=len(refs),
        n_words=nw,
        n_chars=nc,
    )


# -----------------------------------------------------------------------------
# Field-level decomposition (Table V)
# -----------------------------------------------------------------------------
# The word-level Sub/Del/Ins split is uninformative on this corpus. Every
# reference is one of exactly two fixed-arity shapes -- a 3-word date
# ("august 19 1947") or a 2-word time ("1045 am") -- and both the AR beam and
# the utterance grammar emit the correct word count on essentially every
# utterance. Deletions and insertions are therefore identically zero and
# Sub == WER for every system, which measured out exactly that way:
#
#     ar_beam            Sub 0.462  Del 0.000  Ins 0.000   (WER 0.4621)
#     ns_joint_adaptive  Sub 0.402  Del 0.000  Ins 0.000   (WER 0.4015)
#
# That is a real property of a closed fixed-arity vocabulary, not a bug, and it
# is worth stating once. But it leaves Table V with no discriminating content,
# so the table body reports per-field accuracy instead: which semantic slot the
# system actually gets wrong, and whether the error is numeric or alphabetic.
# The corpus has THREE templates, not two, and the two 3-word ones are not
# interchangeable -- measured over all 500 references:
#
#     173  MONTH DD YYYY      ("august 19 1947")
#     167  WEEKDAY MONTH DD   ("sunday may 21")
#     160  HHMM {am,pm}       ("1045 am")
#
# Keying only on "3 words and the first is not a digit" silently labels
# "sunday may 21" as month=sunday / day=may / year=21, which is a third of the
# corpus scored against the wrong slots. The first word is therefore matched
# against the actual weekday and month vocabularies.
try:
    from scripts.grammar import MONTHS as _MONTHS, WEEKDAYS as _WEEKDAYS
except ImportError:  # running from inside scripts/
    from grammar import MONTHS as _MONTHS, WEEKDAYS as _WEEKDAYS

_MONTH_SET = set(_MONTHS)
_WEEKDAY_SET = set(_WEEKDAYS)

FIELD_ORDER = ("weekday", "month", "day", "year", "clock", "meridiem")


def _split_fields(words: Sequence[str]) -> List[Tuple[str, str]] | None:
    """Label a reference's words with their semantic slot, or None if off-template."""
    if len(words) == 3 and words[0] in _WEEKDAY_SET:
        return [("weekday", words[0]), ("month", words[1]), ("day", words[2])]
    if len(words) == 3 and words[0] in _MONTH_SET:
        return [("month", words[0]), ("day", words[1]), ("year", words[2])]
    if len(words) == 2 and words[0].isdigit():
        return [("clock", words[0]), ("meridiem", words[1])]
    return None


@dataclass
class FieldScores:
    """Per-slot accuracy, plus the numeric/alphabetic split that drives it."""

    per_field: Dict[str, Tuple[int, int]]  # slot -> (n_correct, n_total)
    numeric: Tuple[int, int]               # digit-valued slots
    alpha: Tuple[int, int]                 # word-valued slots
    shape_ok: Tuple[int, int]              # hyp word count matches the reference
    n_off_shape_refs: int                  # references matching neither template

    @staticmethod
    def _pct(pair: Tuple[int, int]) -> float:
        c, t = pair
        return 100.0 * c / t if t else float("nan")

    def as_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name in FIELD_ORDER:
            if self.per_field.get(name, (0, 0))[1]:
                out[name] = self._pct(self.per_field[name])
        out["numeric"] = self._pct(self.numeric)
        out["alpha"] = self._pct(self.alpha)
        out["shape"] = self._pct(self.shape_ok)
        return out


def field_breakdown(refs: Sequence[str], hyps: Sequence[str], tt=None) -> FieldScores:
    """Per-slot exact accuracy for Table V.

    Slots are compared positionally. A hypothesis whose word count differs from
    the reference cannot be aligned slot-to-slot, so every one of its fields is
    counted wrong -- charging the whole utterance rather than guessing at an
    alignment, which keeps the numbers a lower bound rather than an optimistic one.
    """
    clean = (lambda s: tt.clean(s)) if tt is not None else (lambda s: (s or "").strip())

    per: Dict[str, List[int]] = {f: [0, 0] for f in FIELD_ORDER}
    num_c = num_t = alp_c = alp_t = 0
    shape_c = shape_t = 0
    off_shape = 0

    for r, h in zip(refs, hyps):
        rw, hw = clean(r).split(), clean(h).split()
        fields = _split_fields(rw)
        if fields is None:
            off_shape += 1
            continue

        aligned = len(hw) == len(rw)
        shape_t += 1
        shape_c += int(aligned)

        for idx, (name, rtok) in enumerate(fields):
            ok = aligned and hw[idx] == rtok
            per[name][1] += 1
            per[name][0] += int(ok)
            if rtok.isdigit():
                num_t += 1
                num_c += int(ok)
            else:
                alp_t += 1
                alp_c += int(ok)

    return FieldScores(
        per_field={k: (v[0], v[1]) for k, v in per.items()},
        numeric=(num_c, num_t),
        alpha=(alp_c, alp_t),
        shape_ok=(shape_c, shape_t),
        n_off_shape_refs=off_shape,
    )


def macro_wer(refs: Sequence[str], hyps: Sequence[str], tt=None) -> float:
    """Mean per-utterance WER (reported separately from the corpus WER)."""
    clean = (lambda s: tt.clean(s)) if tt is not None else (lambda s: (s or "").strip())
    vals = []
    for r, h in zip(refs, hyps):
        rw, hw = clean(r).split(), clean(h).split()
        vals.append(edit_distance(rw, hw) / max(1, len(rw)))
    return sum(vals) / max(1, len(vals))


# -----------------------------------------------------------------------------
# Paired bootstrap over utterances (Fig. 3)
# -----------------------------------------------------------------------------
def bootstrap_ci(
    refs: Sequence[str],
    hyps: Sequence[str],
    tt=None,
    n_resamples: int = 10000,
    ci: float = 95.0,
    seed: int = 1234,
) -> Tuple[float, float, float]:
    """
    Percentile confidence interval for corpus WER via bootstrap resampling
    over utterances. Returns (point_estimate, lo, hi).
    """
    clean = (lambda s: tt.clean(s)) if tt is not None else (lambda s: (s or "").strip())
    per: List[Tuple[int, int]] = []  # (errors, ref_words) per utterance
    for r, h in zip(refs, hyps):
        rw, hw = clean(r).split(), clean(h).split()
        per.append((edit_distance(rw, hw), len(rw)))

    tot_e = sum(e for e, _ in per)
    tot_n = max(1, sum(n for _, n in per))
    point = tot_e / tot_n

    rng = random.Random(seed)
    N = len(per)
    if N == 0:
        return point, point, point

    draws: List[float] = []
    for _ in range(int(n_resamples)):
        e = n = 0
        for _ in range(N):
            de, dn = per[rng.randrange(N)]
            e += de
            n += dn
        draws.append(e / max(1, n))

    draws.sort()
    lo_q = (100.0 - ci) / 2.0
    hi_q = 100.0 - lo_q
    lo = draws[max(0, min(len(draws) - 1, int(round(lo_q / 100.0 * (len(draws) - 1)))))]
    hi = draws[max(0, min(len(draws) - 1, int(round(hi_q / 100.0 * (len(draws) - 1)))))]
    return point, lo, hi


def paired_bootstrap_pvalue(
    refs: Sequence[str],
    hyps_a: Sequence[str],
    hyps_b: Sequence[str],
    tt=None,
    n_resamples: int = 10000,
    seed: int = 1234,
) -> float:
    """
    Two-sided paired bootstrap p-value for "system A and B have equal WER",
    resampling utterances jointly so the comparison stays paired.
    """
    clean = (lambda s: tt.clean(s)) if tt is not None else (lambda s: (s or "").strip())
    per = []
    for r, ha, hb in zip(refs, hyps_a, hyps_b):
        rw = clean(r).split()
        per.append(
            (
                edit_distance(rw, clean(ha).split()),
                edit_distance(rw, clean(hb).split()),
                len(rw),
            )
        )

    def delta(sample) -> float:
        ea = sum(x[0] for x in sample)
        eb = sum(x[1] for x in sample)
        n = max(1, sum(x[2] for x in sample))
        return (ea - eb) / n

    obs = delta(per)
    rng = random.Random(seed)
    N = len(per)
    if N == 0:
        return 1.0

    count = 0
    for _ in range(int(n_resamples)):
        sample = [per[rng.randrange(N)] for _ in range(N)]
        if abs(delta(sample) - obs) >= abs(obs):
            count += 1
    return count / float(n_resamples)
