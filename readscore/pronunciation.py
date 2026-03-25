"""
Pronunciation quality analysis module.

Uses ASR confidence scores and alignment data to estimate pronunciation quality.
For Hebrew, scoring is more lenient when ASR reliability is low, because Whisper
base/small models are less calibrated for Hebrew than for English.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from .align import AlignmentResult, AlignTag


# ── ASR reliability thresholds ────────────────────────────────────────────────
# Hebrew uses lower thresholds: Whisper base is less calibrated for Hebrew,
# so lower average confidence is still acceptable.

_RELIABILITY = {
    "he": {
        "low_conf_word_threshold": 0.65,   # per-word threshold
        "stable_avg": 0.75,
        "stable_low_ratio": 0.15,
        "mixed_avg": 0.58,
        "mixed_low_ratio": 0.45,
    },
    "default": {
        "low_conf_word_threshold": 0.70,
        "stable_avg": 0.85,
        "stable_low_ratio": 0.10,
        "mixed_avg": 0.70,
        "mixed_low_ratio": 0.25,
    },
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class PronunciationResult:
    """Result of pronunciation quality analysis."""
    score_0_100: float
    asr_avg_conf: float
    substitution_severity: float
    low_confidence_words: int
    total_words: int
    notes: List[str]
    asr_reliability: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score_0_100": round(self.score_0_100, 1),
            "signals": {
                "asr_avg_conf": round(self.asr_avg_conf, 4),
                "substitution_severity": round(self.substitution_severity, 4),
                "low_confidence_word_ratio": round(
                    self.low_confidence_words / max(1, self.total_words), 4
                ),
            },
            "asr_reliability": self.asr_reliability,
            "notes": self.notes,
        }


# ── ASR reliability computation ───────────────────────────────────────────────

def compute_asr_reliability(confidences: List[float], lang: str = "en") -> Dict[str, Any]:
    """
    Classify ASR reliability as 'stable', 'mixed', or 'unstable'.

    Hebrew uses more lenient thresholds because Whisper is less calibrated
    for Hebrew than for English or Russian.
    """
    if not confidences:
        return {"avg_confidence": None, "low_confidence_ratio": None, "status": "unknown"}

    cfg = _RELIABILITY.get(lang, _RELIABILITY["default"])
    low_thresh = cfg["low_conf_word_threshold"]

    avg = sum(confidences) / len(confidences)
    low_ratio = sum(1 for c in confidences if c < low_thresh) / len(confidences)

    if avg >= cfg["stable_avg"] and low_ratio <= cfg["stable_low_ratio"]:
        status = "stable"
    elif avg >= cfg["mixed_avg"] and low_ratio <= cfg["mixed_low_ratio"]:
        status = "mixed"
    else:
        status = "unstable"

    return {
        "avg_confidence": round(avg, 4),
        "low_confidence_ratio": round(low_ratio, 4),
        "status": status,   # "stable" | "mixed" | "unstable" | "unknown"
    }


# ── Main pronunciation analysis ───────────────────────────────────────────────

def analyze_pronunciation(
    alignment: AlignmentResult,
    confidence_threshold: float = 0.7,
    lang: str = "en",
) -> PronunciationResult:
    """
    Analyze pronunciation quality based on ASR output and alignment.

    For Hebrew (lang='he'):
    - ASR reliability is assessed with Hebrew-specific thresholds.
    - If ASR is 'unstable', the score is boosted to avoid punishing the reader
      for recognition errors rather than actual mispronunciations.
    - near_match and low_confidence_match words (from align.py) are already
      excluded from substitutions, so they don't double-penalise here.

    Args:
        alignment: Word alignment result
        confidence_threshold: Per-word confidence threshold (fallback; reliability
                              assessment uses language-specific thresholds internally)
        lang: Language code ('en', 'ru', 'he')

    Returns:
        PronunciationResult with quality assessment and ASR reliability block
    """
    notes: List[str] = []

    # Collect confidence scores from aligned/substituted words
    confidences: List[float] = []
    _SPOKEN_TAGS = {AlignTag.CORRECT, AlignTag.WRONG_WORD, AlignTag.NEAR_MATCH, AlignTag.UNCERTAIN_ASR}
    for word in alignment.alignment:
        if word.conf is not None and word.tag in _SPOKEN_TAGS:
            confidences.append(word.conf)

    total_words = len([w for w in alignment.alignment
                       if w.tag not in (AlignTag.EXTRA, AlignTag.ASR_NOISE)])

    # Compute ASR reliability with language-specific thresholds
    reliability = compute_asr_reliability(confidences, lang=lang)

    if not confidences:
        return PronunciationResult(
            score_0_100=50.0,
            asr_avg_conf=0.0,
            substitution_severity=0.0,
            low_confidence_words=0,
            total_words=total_words,
            notes=["No confidence data available from ASR"],
            asr_reliability=reliability,
        )

    avg_conf = sum(confidences) / len(confidences)
    low_conf_words = sum(1 for c in confidences if c < confidence_threshold)
    sub_severity = _calculate_substitution_severity(alignment)

    score = _calculate_pronunciation_score(
        avg_conf, low_conf_words, len(confidences), sub_severity, alignment, notes
    )

    # ── Hebrew leniency adjustment ────────────────────────────────────────────
    # When Hebrew ASR is unreliable, reduce the penalty weight so that
    # recognition instability does not masquerade as reading mistakes.
    if lang == "he":
        status = reliability.get("status", "stable")
        if status == "unstable":
            boost = 15
            score = min(100, score + boost)
            notes.append(
                "ASR reliability: unstable — score adjusted upward. "
                "Some apparent errors likely reflect recognition uncertainty, not mispronunciation."
            )
        elif status == "mixed":
            boost = 7
            score = min(100, score + boost)
            notes.append(
                "ASR reliability: mixed — minor score adjustment applied. "
                "Uncertain words are treated more leniently."
            )

    return PronunciationResult(
        score_0_100=score,
        asr_avg_conf=avg_conf,
        substitution_severity=sub_severity,
        low_confidence_words=low_conf_words,
        total_words=total_words,
        notes=notes,
        asr_reliability=reliability,
    )


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _calculate_substitution_severity(alignment: AlignmentResult) -> float:
    """
    Calculate severity of substitutions as a pronunciation proxy.
    Uses character-level edit distance normalised by word length.
    near_match and low_confidence_match words are excluded — they were
    already determined to be acceptable by align.py.
    """
    if alignment.substitutions == 0:
        return 0.0

    total_distance = 0.0
    sub_count = 0

    for word in alignment.alignment:
        if word.tag == AlignTag.WRONG_WORD and word.ref and word.hyp:
            ref = word.ref.lower()
            hyp = word.hyp.lower()
            dist = _levenshtein_distance(ref, hyp)
            max_len = max(len(ref), len(hyp))
            if max_len > 0:
                total_distance += dist / max_len
                sub_count += 1

    return total_distance / sub_count if sub_count > 0 else 0.0


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Character-level Levenshtein distance."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _calculate_pronunciation_score(
    avg_conf: float,
    low_conf_words: int,
    total_words: int,
    sub_severity: float,
    alignment: AlignmentResult,
    notes: List[str],
) -> float:
    """Calculate overall pronunciation score 0–100 (before language adjustments)."""

    # Confidence component (40% weight)
    if avg_conf >= 0.9:
        conf_score = 40
    elif avg_conf >= 0.8:
        conf_score = 35
    elif avg_conf >= 0.7:
        conf_score = 30
    elif avg_conf >= 0.6:
        conf_score = 20
        notes.append("Below average ASR confidence — possible pronunciation issues")
    else:
        conf_score = 10
        notes.append("Low ASR confidence — likely pronunciation issues")

    # Low-confidence word ratio component (20% weight)
    low_conf_ratio = low_conf_words / max(1, total_words)
    if low_conf_ratio <= 0.05:
        low_conf_score = 20
    elif low_conf_ratio <= 0.10:
        low_conf_score = 15
    elif low_conf_ratio <= 0.20:
        low_conf_score = 10
        notes.append("Multiple words with low recognition confidence")
    else:
        low_conf_score = 5
        notes.append("Many words with low recognition confidence")

    # Substitution severity component (25% weight)
    if sub_severity == 0:
        sub_score = 25
    elif sub_severity < 0.3:
        sub_score = 20
        notes.append("Minor word substitutions detected")
    elif sub_severity < 0.5:
        sub_score = 15
        notes.append("Moderate word substitutions — possible mispronunciations")
    else:
        sub_score = 5
        notes.append("Significant substitutions — likely mispronunciations")

    # Deletion component (15% weight)
    total_ref = (alignment.correct + alignment.omitted + alignment.wrong_word +
                 alignment.near_match + alignment.uncertain_asr)
    if total_ref > 0:
        del_ratio = alignment.omitted / total_ref
        if del_ratio <= 0.02:
            del_score = 15
        elif del_ratio <= 0.05:
            del_score = 12
        elif del_ratio <= 0.10:
            del_score = 8
            notes.append("Some words not recognised — may be unclear")
        else:
            del_score = 3
            notes.append("Many words not recognised — speech may be unclear")
    else:
        del_score = 15

    score = conf_score + low_conf_score + sub_score + del_score

    if not notes:
        notes.append("Pronunciation appears clear and accurate")

    return min(100, max(0, float(score)))
