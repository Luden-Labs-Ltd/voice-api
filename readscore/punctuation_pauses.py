"""
Punctuation-aware pause analysis module.

Evaluates whether pauses are appropriate at punctuation boundaries:
- comma, period, question, exclamation, colon, semicolon, ellipsis
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


# Default expected pause ranges (seconds) per punctuation type
DEFAULT_PAUSE_RANGES = {
    ",": {"min": 0.10, "max": 0.60},
    ".": {"min": 0.25, "max": 1.20},
    ";": {"min": 0.20, "max": 1.00},
    ":": {"min": 0.20, "max": 1.00},
    "?": {"min": 0.25, "max": 1.30},
    "!": {"min": 0.25, "max": 1.30},
    "...": {"min": 0.40, "max": 2.00},
    "…": {"min": 0.40, "max": 2.00},
}


@dataclass
class PauseConfig:
    """Configuration for punctuation pause analysis."""
    pause_ranges: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: DEFAULT_PAUSE_RANGES.copy()
    )

    @classmethod
    def from_dict(cls, d: dict) -> "PauseConfig":
        ranges = d.get("punctuation_pause_ranges", DEFAULT_PAUSE_RANGES.copy())
        # Merge with defaults for any missing punctuation types
        merged = DEFAULT_PAUSE_RANGES.copy()
        merged.update(ranges)
        return cls(pause_ranges=merged)


@dataclass
class PauseEvent:
    """A punctuation pause event with its analysis."""
    punct: str
    ref_word: str
    ref_word_index: int
    t_prev_end: Optional[float] = None
    t_next_start: Optional[float] = None
    pause_sec: Optional[float] = None
    expected_min: Optional[float] = None
    expected_max: Optional[float] = None
    classification: str = "unknown"
    score_0_100: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "punct": self.punct,
            "ref_word": self.ref_word,
            "t_prev_end": self.t_prev_end,
            "t_next_start": self.t_next_start,
            "pause_sec": self.pause_sec,
            "expected": {
                "min": self.expected_min,
                "max": self.expected_max
            } if self.expected_min is not None else None,
            "classification": self.classification,
            "score_0_100": self.score_0_100,
            "notes": self.notes
        }


@dataclass
class PauseAnalysisResult:
    """Result of punctuation pause analysis."""
    events: List[PauseEvent]
    avg_score: Optional[float]
    count: int
    ok_count: int
    too_short_count: int
    too_long_count: int
    missing_count: int

    def to_dict(self) -> dict:
        return {
            "events": [e.to_dict() for e in self.events],
            "summary": {
                "avg_score": self.avg_score,
                "count": self.count,
                "ok": self.ok_count,
                "too_short": self.too_short_count,
                "too_long": self.too_long_count,
                "missing": self.missing_count
            }
        }


def parse_punctuation_events(reference_text: str) -> List[Tuple[str, str, int]]:
    """
    Parse reference text and find all punctuation events.

    Returns:
        List of (punctuation, word_before, word_index) tuples.
        word_index is the 0-based index of the word in the normalized word list.
    """
    events = []

    # Pattern to match words and punctuation separately
    # Handles ellipsis as single token, other punctuation individually
    token_pattern = r"[\w']+|\.{3}|…|[.,;:?!]"
    tokens = re.findall(token_pattern, reference_text)

    word_index = 0
    last_word = None
    last_word_index = -1

    for token in tokens:
        # Check if token is punctuation
        if token in [",", ".", ";", ":", "?", "!", "...", "…"]:
            if last_word is not None:
                # Normalize ellipsis
                punct = "…" if token == "..." else token
                events.append((punct, last_word.lower(), last_word_index))
        else:
            # It's a word
            last_word = token
            last_word_index = word_index
            word_index += 1

    return events


def find_alignment_item_by_ref_index(
    alignment: List[dict],
    target_ref_index: int
) -> Optional[dict]:
    """
    Find the alignment item corresponding to a reference word index.

    Args:
        alignment: List of alignment items with ref, hyp, tag, t0, t1
        target_ref_index: 0-based index of reference word

    Returns:
        The alignment item or None if not found
    """
    ref_count = 0
    for item in alignment:
        # Count reference words (ok, sub, del have ref words)
        if item.get("tag") in ["correct", "wrong_word", "omitted", "near_match", "uncertain_asr"]:
            if ref_count == target_ref_index:
                return item
            ref_count += 1
    return None


def find_next_spoken_word(
    alignment: List[dict],
    after_ref_index: int
) -> Optional[dict]:
    """
    Find the next spoken word after a reference word index.

    Args:
        alignment: List of alignment items
        after_ref_index: The reference word index to search after

    Returns:
        The next alignment item that has timestamps (ok or sub), or None
    """
    ref_count = 0
    found_target = False

    for item in alignment:
        tag = item.get("tag")

        if tag in ["correct", "wrong_word", "omitted", "near_match", "uncertain_asr"]:
            if ref_count == after_ref_index:
                found_target = True
            ref_count += 1

            # After finding target, look for next spoken word
            if found_target and ref_count > after_ref_index + 1:
                if tag in ["correct", "wrong_word", "near_match", "uncertain_asr"] and item.get("t0") is not None:
                    return item

        elif tag in ("extra", "asr_noise"):
            # Insertions don't have ref index but might be the next spoken
            if found_target and item.get("t0") is not None:
                return item

    return None


def classify_pause(
    pause_sec: Optional[float],
    expected_min: float,
    expected_max: float
) -> Tuple[str, Optional[float], List[str]]:
    """
    Classify a pause duration and compute score.

    Returns:
        (classification, score_0_100, notes)
    """
    notes = []

    if pause_sec is None:
        return "missing", None, ["No pause data available"]

    if pause_sec < 0:
        # Overlapping speech or timing error
        return "missing", 0, ["Negative pause (overlapping speech or timing error)"]

    # Classification
    if pause_sec < expected_min:
        classification = "too_short"
        # Score decreases as pause gets shorter
        if expected_min > 0:
            ratio = pause_sec / expected_min
            score = max(0, ratio * 70)  # 0-70 range for too short
        else:
            score = 70
        notes.append(f"Pause {pause_sec:.2f}s is below minimum {expected_min:.2f}s")

    elif pause_sec > expected_max:
        classification = "too_long"
        # Score decreases as pause gets longer
        excess = pause_sec - expected_max
        # Every 0.5s over max loses 15 points, starting from 70
        score = max(0, 70 - (excess / 0.5) * 15)
        notes.append(f"Pause {pause_sec:.2f}s exceeds maximum {expected_max:.2f}s")

    else:
        classification = "ok"
        # Score 85-100 based on how centered in the range
        mid = (expected_min + expected_max) / 2
        range_half = (expected_max - expected_min) / 2
        if range_half > 0:
            distance_from_mid = abs(pause_sec - mid) / range_half
            score = 100 - (distance_from_mid * 15)  # 85-100 range
        else:
            score = 100
        notes.append(f"Pause {pause_sec:.2f}s is within expected range")

    return classification, round(score, 1), notes


def analyze_punctuation_pauses(
    reference_text: str,
    alignment: List[dict],
    config: Optional[PauseConfig] = None
) -> PauseAnalysisResult:
    """
    Analyze pauses at punctuation boundaries.

    Args:
        reference_text: Original reference text with punctuation
        alignment: Word alignment from accuracy analysis
        config: Optional configuration for pause ranges

    Returns:
        PauseAnalysisResult with analysis for each punctuation event
    """
    if config is None:
        config = PauseConfig()

    # Parse punctuation events from reference text
    punct_events = parse_punctuation_events(reference_text)

    events = []
    scores = []
    classifications = {"ok": 0, "too_short": 0, "too_long": 0, "missing": 0}

    for punct, ref_word, ref_word_index in punct_events:
        event = PauseEvent(
            punct=punct,
            ref_word=ref_word,
            ref_word_index=ref_word_index
        )

        # Get expected range for this punctuation type
        range_config = config.pause_ranges.get(punct, {"min": 0.2, "max": 1.0})
        event.expected_min = range_config["min"]
        event.expected_max = range_config["max"]

        # Find the alignment item for the word before punctuation
        prev_item = find_alignment_item_by_ref_index(alignment, ref_word_index)

        if prev_item is None:
            event.notes.append("Could not find word in alignment")
            event.classification = "missing"
            classifications["missing"] += 1
            events.append(event)
            continue

        # Get end time of previous word
        if prev_item.get("tag") == "omitted":
            event.notes.append("Word was deleted (not spoken)")
            event.classification = "missing"
            classifications["missing"] += 1
            events.append(event)
            continue

        event.t_prev_end = prev_item.get("t1")

        if event.t_prev_end is None:
            event.notes.append("No timestamp for previous word")
            event.classification = "missing"
            classifications["missing"] += 1
            events.append(event)
            continue

        # Find the next spoken word
        next_item = find_next_spoken_word(alignment, ref_word_index)

        if next_item is None:
            event.notes.append("No next word found (end of utterance)")
            event.classification = "missing"
            classifications["missing"] += 1
            events.append(event)
            continue

        event.t_next_start = next_item.get("t0")

        if event.t_next_start is None:
            event.notes.append("No timestamp for next word")
            event.classification = "missing"
            classifications["missing"] += 1
            events.append(event)
            continue

        # Compute pause duration
        event.pause_sec = round(event.t_next_start - event.t_prev_end, 3)

        # Classify the pause
        classification, score, notes = classify_pause(
            event.pause_sec,
            event.expected_min,
            event.expected_max
        )

        event.classification = classification
        event.score_0_100 = score
        event.notes.extend(notes)

        classifications[classification] += 1
        if score is not None:
            scores.append(score)

        events.append(event)

    # Calculate summary
    avg_score = round(sum(scores) / len(scores), 1) if scores else None

    return PauseAnalysisResult(
        events=events,
        avg_score=avg_score,
        count=len(events),
        ok_count=classifications["ok"],
        too_short_count=classifications["too_short"],
        too_long_count=classifications["too_long"],
        missing_count=classifications["missing"]
    )


def get_pause_for_prosody_event(
    pause_result: PauseAnalysisResult,
    punct: str,
    ref_word: str
) -> Optional[dict]:
    """
    Get pause data for a prosody punctuation event.

    Used to enrich prosody_punctuation events with pause information.

    Args:
        pause_result: Result from analyze_punctuation_pauses
        punct: Punctuation symbol ('?' or '!')
        ref_word: Reference word before punctuation

    Returns:
        Dict with pause_sec and pause_classification, or None if not found
    """
    for event in pause_result.events:
        if event.punct == punct and event.ref_word == ref_word:
            return {
                "pause_sec": event.pause_sec,
                "pause_classification": event.classification,
                "pause_score_0_100": event.score_0_100
            }
    return None
