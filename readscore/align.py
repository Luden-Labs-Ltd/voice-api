"""
Word-level alignment using dynamic programming (Levenshtein distance).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class AlignTag(str, Enum):
    OK = "ok"
    SUB = "sub"
    INS = "ins"
    DEL = "del"


@dataclass
class AlignedWord:
    """Represents an aligned word pair."""
    ref: Optional[str]  # Reference word (None for insertions)
    hyp: Optional[str]  # Hypothesis word (None for deletions)
    tag: AlignTag
    t0: Optional[float] = None  # Start time
    t1: Optional[float] = None  # End time
    conf: Optional[float] = None  # Confidence score

    def to_dict(self) -> dict:
        return {
            "ref": self.ref,
            "hyp": self.hyp,
            "tag": self.tag.value,
            "t0": self.t0,
            "t1": self.t1,
            "conf": self.conf
        }


@dataclass
class AlignmentResult:
    """Result of word alignment."""
    alignment: List[AlignedWord]
    wer: float
    insertions: int
    deletions: int
    substitutions: int
    correct: int

    def to_dict(self) -> dict:
        return {
            "wer": round(self.wer, 4),
            "counts": {
                "ins": self.insertions,
                "del": self.deletions,
                "sub": self.substitutions,
                "correct": self.correct
            },
            "alignment": [w.to_dict() for w in self.alignment]
        }


def levenshtein_align(
    ref_words: List[str],
    hyp_words: List[str],
    hyp_timestamps: Optional[List[Tuple[float, float]]] = None,
    hyp_confidences: Optional[List[float]] = None
) -> AlignmentResult:
    """
    Align reference and hypothesis word sequences using Levenshtein distance.

    Args:
        ref_words: Reference word sequence
        hyp_words: Hypothesis (ASR output) word sequence
        hyp_timestamps: Optional list of (start, end) times for each hyp word
        hyp_confidences: Optional confidence scores for each hyp word

    Returns:
        AlignmentResult with alignment details and WER
    """
    n = len(ref_words)
    m = len(hyp_words)

    # DP table: dp[i][j] = (cost, operation)
    # Operations: 'ok', 'sub', 'ins', 'del'
    INF = float('inf')

    # Initialize DP table
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    # First column: deletions
    for i in range(1, n + 1):
        dp[i][0] = i

    # First row: insertions
    for j in range(1, m + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            ref_word = ref_words[i - 1].lower()
            hyp_word = hyp_words[j - 1].lower()

            if ref_word == hyp_word:
                # Match
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Minimum of substitution, insertion, deletion
                dp[i][j] = min(
                    dp[i - 1][j - 1] + 1,  # Substitution
                    dp[i][j - 1] + 1,       # Insertion
                    dp[i - 1][j] + 1        # Deletion
                )

    # Backtrack to find alignment
    alignment = []
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            ref_word = ref_words[i - 1].lower()
            hyp_word = hyp_words[j - 1].lower()

            if ref_word == hyp_word and dp[i][j] == dp[i - 1][j - 1]:
                # Match
                ts = hyp_timestamps[j - 1] if hyp_timestamps else (None, None)
                conf = hyp_confidences[j - 1] if hyp_confidences else None
                alignment.append(AlignedWord(
                    ref=ref_words[i - 1],
                    hyp=hyp_words[j - 1],
                    tag=AlignTag.OK,
                    t0=ts[0], t1=ts[1], conf=conf
                ))
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j - 1] + 1:
                # Substitution
                ts = hyp_timestamps[j - 1] if hyp_timestamps else (None, None)
                conf = hyp_confidences[j - 1] if hyp_confidences else None
                alignment.append(AlignedWord(
                    ref=ref_words[i - 1],
                    hyp=hyp_words[j - 1],
                    tag=AlignTag.SUB,
                    t0=ts[0], t1=ts[1], conf=conf
                ))
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i][j - 1] + 1:
                # Insertion (extra word in hypothesis)
                ts = hyp_timestamps[j - 1] if hyp_timestamps else (None, None)
                conf = hyp_confidences[j - 1] if hyp_confidences else None
                alignment.append(AlignedWord(
                    ref=None,
                    hyp=hyp_words[j - 1],
                    tag=AlignTag.INS,
                    t0=ts[0], t1=ts[1], conf=conf
                ))
                j -= 1
            else:
                # Deletion (word missing from hypothesis)
                alignment.append(AlignedWord(
                    ref=ref_words[i - 1],
                    hyp=None,
                    tag=AlignTag.DEL,
                    t0=None, t1=None, conf=None
                ))
                i -= 1
        elif j > 0:
            # Remaining insertions
            ts = hyp_timestamps[j - 1] if hyp_timestamps else (None, None)
            conf = hyp_confidences[j - 1] if hyp_confidences else None
            alignment.append(AlignedWord(
                ref=None,
                hyp=hyp_words[j - 1],
                tag=AlignTag.INS,
                t0=ts[0], t1=ts[1], conf=conf
            ))
            j -= 1
        else:
            # Remaining deletions
            alignment.append(AlignedWord(
                ref=ref_words[i - 1],
                hyp=None,
                tag=AlignTag.DEL,
                t0=None, t1=None, conf=None
            ))
            i -= 1

    # Reverse to get correct order
    alignment.reverse()

    # Count errors
    insertions = sum(1 for a in alignment if a.tag == AlignTag.INS)
    deletions = sum(1 for a in alignment if a.tag == AlignTag.DEL)
    substitutions = sum(1 for a in alignment if a.tag == AlignTag.SUB)
    correct = sum(1 for a in alignment if a.tag == AlignTag.OK)

    # Calculate WER
    total_ref = len(ref_words)
    if total_ref == 0:
        wer = 0.0 if len(hyp_words) == 0 else 1.0
    else:
        wer = (insertions + deletions + substitutions) / total_ref

    return AlignmentResult(
        alignment=alignment,
        wer=wer,
        insertions=insertions,
        deletions=deletions,
        substitutions=substitutions,
        correct=correct
    )
