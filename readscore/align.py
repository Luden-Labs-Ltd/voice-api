"""
Word-level alignment using dynamic programming (Levenshtein distance).
Hebrew is the primary language; it receives aggressive soft-matching
post-processing to reduce false errors from ASR noise and uncertainty.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class AlignTag(str, Enum):
    CORRECT = "correct"              # Word read correctly
    NEAR_MATCH = "near_match"        # Close enough (Hebrew clitics, morphology)
    UNCERTAIN_ASR = "uncertain_asr"  # ASR confidence too low to count as error
    WRONG_WORD = "wrong_word"        # Clearly wrong — high confidence mismatch
    OMITTED = "omitted"              # Word was skipped by reader
    EXTRA = "extra"                  # Word spoken not in reference
    ASR_NOISE = "asr_noise"          # Bogus ASR output — not a reader error


# ── Hebrew thresholds ─────────────────────────────────────────────────────────

# Similarity >= this → NEAR_MATCH (clitics, suffix endings, final-form remnants)
HEBREW_NEAR_MATCH_THRESHOLD = 0.65

# Similarity >= this → UNCERTAIN_ASR (morphological variation)
HEBREW_UNCERTAIN_SIM_THRESHOLD = 0.35

# ASR word confidence below this → treat SUB as UNCERTAIN_ASR, not WRONG_WORD
HEBREW_ASR_CONF_THRESHOLD = 0.65

# INS confidence below this → ASR_NOISE (bogus token, not real extra word)
HEBREW_NOISE_CONF_THRESHOLD = 0.45


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class AlignedWord:
    """Represents an aligned word pair with timing and confidence."""
    ref: Optional[str]
    hyp: Optional[str]
    tag: AlignTag
    t0: Optional[float] = None
    t1: Optional[float] = None
    conf: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "ref": self.ref,
            "hyp": self.hyp,
            "tag": self.tag.value,
            "t0": self.t0,
            "t1": self.t1,
            "conf": self.conf,
        }


@dataclass
class AlignmentResult:
    """Result of word alignment with per-category counts."""
    alignment: List[AlignedWord]
    wer: float
    correct: int
    near_match: int
    uncertain_asr: int
    wrong_word: int
    omitted: int
    extra: int
    asr_noise: int
    lang: str = "en"

    # Backward-compat aliases used by pronunciation.py
    @property
    def insertions(self) -> int:
        return self.extra

    @property
    def deletions(self) -> int:
        return self.omitted

    @property
    def substitutions(self) -> int:
        return self.wrong_word

    def to_dict(self) -> dict:
        return {
            "wer": round(self.wer, 4),
            "counts": {
                "correct": self.correct,
                "near_match": self.near_match,
                "uncertain_asr": self.uncertain_asr,
                "wrong_word": self.wrong_word,
                "omitted": self.omitted,
                "extra": self.extra,
                "asr_noise": self.asr_noise,
            },
            "alignment": [w.to_dict() for w in self.alignment],
        }


# ── Character-level similarity helpers ───────────────────────────────────────

def _levenshtein_char_distance(s1: str, s2: str) -> int:
    """Compute character-level Levenshtein edit distance."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[-1] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _char_similarity(s1: str, s2: str) -> float:
    """
    Normalized character similarity between two strings.
    Returns 1.0 for identical, 0.0 for completely different.
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    dist = _levenshtein_char_distance(s1, s2)
    return 1.0 - dist / max(len(s1), len(s2))


# ── Main alignment function ───────────────────────────────────────────────────

def levenshtein_align(
    ref_words: List[str],
    hyp_words: List[str],
    hyp_timestamps: Optional[List[Tuple[float, float]]] = None,
    hyp_confidences: Optional[List[float]] = None,
    lang: str = "en",
) -> "AlignmentResult":
    """
    Align reference and hypothesis word sequences using Levenshtein distance.

    For Hebrew (lang='he'), substitutions and insertions are post-processed into
    finer categories so that poor ASR performance does not masquerade as reading errors:

      WRONG_WORD (sub):
        sim >= 0.65              → NEAR_MATCH   (morphological variant, no penalty)
        conf < 0.65              → UNCERTAIN_ASR (ASR not confident, no penalty)
        sim >= 0.35              → UNCERTAIN_ASR (somewhat similar, no penalty)
        else                     → WRONG_WORD    (high-confidence clear mismatch)

      EXTRA (ins):
        conf < 0.45 or len == 1 → ASR_NOISE     (bogus token, no penalty)
        else                     → EXTRA          (reader added a word)

    WER = (wrong_word + omitted + extra) / len(ref_words)
    near_match, uncertain_asr, and asr_noise are not counted as errors.
    """
    n = len(ref_words)
    m = len(hyp_words)
    INF = float('inf')

    # ── DP table initialisation ──────────────────────────────────────────────
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    # ── Fill DP table ────────────────────────────────────────────────────────
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1].lower() == hyp_words[j - 1].lower():
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j - 1] + 1,  # substitution
                    dp[i][j - 1] + 1,       # insertion (extra hyp word)
                    dp[i - 1][j] + 1,       # deletion  (missing ref word)
                )

    # ── Backtracking — assign initial tags ───────────────────────────────────
    alignment: List[AlignedWord] = []
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            ref_w = ref_words[i - 1].lower()
            hyp_w = hyp_words[j - 1].lower()
            ts = hyp_timestamps[j - 1] if hyp_timestamps else (None, None)
            conf = hyp_confidences[j - 1] if hyp_confidences else None

            if ref_w == hyp_w and dp[i][j] == dp[i - 1][j - 1]:
                alignment.append(AlignedWord(
                    ref=ref_words[i - 1], hyp=hyp_words[j - 1],
                    tag=AlignTag.CORRECT, t0=ts[0], t1=ts[1], conf=conf,
                ))
                i -= 1; j -= 1
            elif dp[i][j] == dp[i - 1][j - 1] + 1:
                alignment.append(AlignedWord(
                    ref=ref_words[i - 1], hyp=hyp_words[j - 1],
                    tag=AlignTag.WRONG_WORD, t0=ts[0], t1=ts[1], conf=conf,
                ))
                i -= 1; j -= 1
            elif dp[i][j] == dp[i][j - 1] + 1:
                alignment.append(AlignedWord(
                    ref=None, hyp=hyp_words[j - 1],
                    tag=AlignTag.EXTRA, t0=ts[0], t1=ts[1], conf=conf,
                ))
                j -= 1
            else:
                alignment.append(AlignedWord(
                    ref=ref_words[i - 1], hyp=None,
                    tag=AlignTag.OMITTED,
                ))
                i -= 1
        elif j > 0:
            ts = hyp_timestamps[j - 1] if hyp_timestamps else (None, None)
            conf = hyp_confidences[j - 1] if hyp_confidences else None
            alignment.append(AlignedWord(
                ref=None, hyp=hyp_words[j - 1],
                tag=AlignTag.EXTRA, t0=ts[0], t1=ts[1], conf=conf,
            ))
            j -= 1
        else:
            alignment.append(AlignedWord(
                ref=ref_words[i - 1], hyp=None,
                tag=AlignTag.OMITTED,
            ))
            i -= 1

    alignment.reverse()

    # ── Hebrew soft-matching post-processing ─────────────────────────────────
    # Reclassify mismatches and insertions so that ASR noise / uncertainty
    # does not masquerade as reading errors.
    if lang == "he":
        for item in alignment:
            conf = item.conf if item.conf is not None else 1.0

            if item.tag == AlignTag.WRONG_WORD:
                sim = _char_similarity(item.ref or "", item.hyp or "")
                if sim >= HEBREW_NEAR_MATCH_THRESHOLD:
                    # Orthographically close: prefix clitics, suffix endings, final forms
                    item.tag = AlignTag.NEAR_MATCH
                elif conf < HEBREW_ASR_CONF_THRESHOLD:
                    # ASR wasn't confident — may not be the reader's fault
                    item.tag = AlignTag.UNCERTAIN_ASR
                elif sim >= HEBREW_UNCERTAIN_SIM_THRESHOLD:
                    # Somewhat similar — could be morphological variation
                    item.tag = AlignTag.UNCERTAIN_ASR
                # else: high-confidence, clearly different → stays WRONG_WORD

            elif item.tag == AlignTag.EXTRA:
                if (conf < HEBREW_NOISE_CONF_THRESHOLD or
                        (item.hyp and len(item.hyp) <= 1)):
                    # Very low confidence or single-char token — likely ASR hallucination
                    item.tag = AlignTag.ASR_NOISE

    # ── Tally counts ─────────────────────────────────────────────────────────
    counts = {t: 0 for t in AlignTag}
    for item in alignment:
        counts[item.tag] += 1

    # WER: only hard errors count (wrong_word + omitted + extra)
    # near_match, uncertain_asr, and asr_noise are excluded
    total_ref = len(ref_words)
    hard_errors = (
        counts[AlignTag.WRONG_WORD] +
        counts[AlignTag.OMITTED] +
        counts[AlignTag.EXTRA]
    )
    if total_ref == 0:
        wer = 0.0 if len(hyp_words) == 0 else 1.0
    else:
        wer = hard_errors / total_ref

    return AlignmentResult(
        alignment=alignment,
        wer=wer,
        correct=counts[AlignTag.CORRECT],
        near_match=counts[AlignTag.NEAR_MATCH],
        uncertain_asr=counts[AlignTag.UNCERTAIN_ASR],
        wrong_word=counts[AlignTag.WRONG_WORD],
        omitted=counts[AlignTag.OMITTED],
        extra=counts[AlignTag.EXTRA],
        asr_noise=counts[AlignTag.ASR_NOISE],
        lang=lang,
    )
