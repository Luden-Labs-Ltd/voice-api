"""
Pronunciation quality analysis module.

Uses ASR confidence scores and alignment data to estimate pronunciation quality.
"""

from dataclasses import dataclass
from typing import List, Optional
from .align import AlignmentResult, AlignTag


@dataclass
class PronunciationResult:
    """Result of pronunciation quality analysis."""
    score_0_100: float
    asr_avg_conf: float
    substitution_severity: float
    low_confidence_words: int
    total_words: int
    notes: List[str]

    def to_dict(self) -> dict:
        return {
            "score_0_100": round(self.score_0_100, 1),
            "signals": {
                "asr_avg_conf": round(self.asr_avg_conf, 4),
                "substitution_severity": round(self.substitution_severity, 4),
                "low_confidence_word_ratio": round(
                    self.low_confidence_words / max(1, self.total_words), 4
                )
            },
            "notes": self.notes
        }


def analyze_pronunciation(
    alignment: AlignmentResult,
    confidence_threshold: float = 0.7
) -> PronunciationResult:
    """
    Analyze pronunciation quality based on ASR output and alignment.

    This is a pragmatic baseline using:
    - ASR word confidence scores
    - Substitution patterns as proxy for mispronunciation
    - Deletion patterns as proxy for unclear speech

    Args:
        alignment: Word alignment result
        confidence_threshold: Threshold below which a word is considered low-confidence

    Returns:
        PronunciationResult with quality assessment
    """
    notes = []

    # Extract confidence scores from alignment
    confidences = []
    for word in alignment.alignment:
        if word.conf is not None and word.tag in [AlignTag.OK, AlignTag.SUB]:
            confidences.append(word.conf)

    total_words = len([w for w in alignment.alignment if w.tag != AlignTag.INS])

    if not confidences:
        return PronunciationResult(
            score_0_100=50.0,  # Neutral score when no data
            asr_avg_conf=0.0,
            substitution_severity=0.0,
            low_confidence_words=0,
            total_words=total_words,
            notes=["No confidence data available from ASR"]
        )

    # Calculate average confidence
    avg_conf = sum(confidences) / len(confidences)

    # Count low-confidence words
    low_conf_words = sum(1 for c in confidences if c < confidence_threshold)

    # Calculate substitution severity
    # Weight substitutions by how different the words might be
    sub_severity = _calculate_substitution_severity(alignment)

    # Calculate pronunciation score
    score = _calculate_pronunciation_score(
        avg_conf,
        low_conf_words,
        len(confidences),
        sub_severity,
        alignment,
        notes
    )

    return PronunciationResult(
        score_0_100=score,
        asr_avg_conf=avg_conf,
        substitution_severity=sub_severity,
        low_confidence_words=low_conf_words,
        total_words=total_words,
        notes=notes
    )


def _calculate_substitution_severity(alignment: AlignmentResult) -> float:
    """
    Calculate severity of substitutions as pronunciation proxy.

    Uses simple character-level distance as proxy for phonetic distance.
    """
    if alignment.substitutions == 0:
        return 0.0

    total_distance = 0
    sub_count = 0

    for word in alignment.alignment:
        if word.tag == AlignTag.SUB and word.ref and word.hyp:
            # Simple edit distance ratio
            ref = word.ref.lower()
            hyp = word.hyp.lower()
            distance = _levenshtein_distance(ref, hyp)
            max_len = max(len(ref), len(hyp))
            if max_len > 0:
                normalized_distance = distance / max_len
                total_distance += normalized_distance
                sub_count += 1

    return total_distance / sub_count if sub_count > 0 else 0.0


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _calculate_pronunciation_score(
    avg_conf: float,
    low_conf_words: int,
    total_words: int,
    sub_severity: float,
    alignment: AlignmentResult,
    notes: List[str]
) -> float:
    """Calculate overall pronunciation score 0-100."""
    score = 100.0

    # Confidence component (40% weight)
    # Average confidence: 0.9+ is excellent, 0.7-0.9 is good, below 0.7 is concerning
    if avg_conf >= 0.9:
        conf_score = 40
    elif avg_conf >= 0.8:
        conf_score = 35
    elif avg_conf >= 0.7:
        conf_score = 30
    elif avg_conf >= 0.6:
        conf_score = 20
        notes.append("Below average ASR confidence - possible pronunciation issues")
    else:
        conf_score = 10
        notes.append("Low ASR confidence - likely pronunciation issues")

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
        notes.append("Moderate word substitutions - possible mispronunciations")
    else:
        sub_score = 5
        notes.append("Significant substitutions - likely mispronunciations")

    # Deletion component (15% weight)
    # Deletions might indicate unclear/mumbled speech
    total_ref = alignment.correct + alignment.deletions + alignment.substitutions
    if total_ref > 0:
        del_ratio = alignment.deletions / total_ref
        if del_ratio <= 0.02:
            del_score = 15
        elif del_ratio <= 0.05:
            del_score = 12
        elif del_ratio <= 0.10:
            del_score = 8
            notes.append("Some words not recognized - may be unclear")
        else:
            del_score = 3
            notes.append("Many words not recognized - speech may be unclear")
    else:
        del_score = 15

    score = conf_score + low_conf_score + sub_score + del_score

    if not notes:
        notes.append("Pronunciation appears clear and accurate")

    return min(100, max(0, score))
