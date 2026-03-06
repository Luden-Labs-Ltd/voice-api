"""
Punctuation-aware prosody analysis module.

Analyzes prosody at punctuation boundaries:
- Rising intonation before '?' (questions)
- Emphatic delivery before '!' (exclamations)
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

# Default window expansion sequence for finding voiced frames
DEFAULT_WINDOW_EXPANSION = [0.8, 1.2, 1.6]


def hz_to_semitones(f1: float, f2: float) -> float:
    """
    Convert pitch change from Hz to semitones.
    Semitones = 12 * log2(f2/f1)

    This normalizes pitch changes across different speaker ranges.
    E.g., 10Hz rise at 100Hz (male) = 1.7 semitones
          10Hz rise at 200Hz (female) = 0.85 semitones
    """
    import math
    if f1 <= 0 or f2 <= 0:
        return 0.0
    return 12.0 * math.log2(f2 / f1)


@dataclass
class PunctuationConfig:
    """Configuration for punctuation prosody analysis."""
    window_sec: float = 0.8  # Initial analysis window before punctuation (increased from 0.45)
    window_expansion: List[float] = field(default_factory=lambda: DEFAULT_WINDOW_EXPANSION.copy())

    # Question thresholds - now using semitones for normalization
    question_pitch_delta_hz: float = 4.0  # Min pitch rise in Hz (lowered for sensitivity)
    question_pitch_slope_hz_per_s: float = 8.0  # Alternative slope threshold (lowered)
    question_pitch_delta_semitones: float = 0.8  # Min pitch rise in semitones (primary)
    question_pitch_slope_semitones_per_s: float = 1.5  # Alternative semitone slope

    # Exclamation thresholds
    exclaim_energy_ratio: float = 1.25  # Energy boost ratio threshold
    exclaim_pitch_std_hz: float = 15.0  # Pitch variability threshold
    exclaim_pitch_range_hz: float = 25.0  # Pitch range threshold
    exclaim_pitch_range_semitones: float = 4.0  # Pitch range in semitones

    # Minimum voiced frames required for analysis
    min_voiced_frames: int = 3

    # Use semitone-based classification (more robust across speakers)
    use_semitones: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "PunctuationConfig":
        return cls(
            window_sec=d.get("window_sec", 0.8),
            window_expansion=d.get("window_expansion", DEFAULT_WINDOW_EXPANSION.copy()),
            question_pitch_delta_hz=d.get("question_pitch_delta_hz", 4.0),
            question_pitch_slope_hz_per_s=d.get("question_pitch_slope_hz_per_s", 8.0),
            question_pitch_delta_semitones=d.get("question_pitch_delta_semitones", 0.8),
            question_pitch_slope_semitones_per_s=d.get("question_pitch_slope_semitones_per_s", 1.5),
            exclaim_energy_ratio=d.get("exclaim_energy_ratio", 1.25),
            exclaim_pitch_std_hz=d.get("exclaim_pitch_std_hz", 15.0),
            exclaim_pitch_range_hz=d.get("exclaim_pitch_range_hz", 25.0),
            exclaim_pitch_range_semitones=d.get("exclaim_pitch_range_semitones", 4.0),
            min_voiced_frames=d.get("min_voiced_frames", 3),
            use_semitones=d.get("use_semitones", True),
        )


@dataclass
class PunctuationEvent:
    """A punctuation event with its analysis."""
    punct: str  # '?' or '!'
    ref_word: str  # Reference word before punctuation
    ref_word_index: int  # Index in reference text
    aligned_word: Optional[str] = None
    t_anchor: Optional[float] = None
    features: Dict[str, Optional[float]] = field(default_factory=dict)
    classification: str = "unknown"
    score_0_100: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    window_used_sec: Optional[float] = None  # Actual window used after expansion

    def to_dict(self) -> dict:
        return {
            "punct": self.punct,
            "ref_word": self.ref_word,
            "aligned_word": self.aligned_word,
            "t_anchor": self.t_anchor,
            "features": self.features,
            "classification": self.classification,
            "score_0_100": self.score_0_100,
            "notes": self.notes
        }


@dataclass
class PunctuationProsodyResult:
    """Result of punctuation prosody analysis."""
    window_sec: float
    events: List[PunctuationEvent]
    question_avg_score: Optional[float]
    exclaim_avg_score: Optional[float]
    question_count: int
    exclaim_count: int

    def to_dict(self) -> dict:
        return {
            "window_sec": self.window_sec,
            "events": [e.to_dict() for e in self.events],
            "summary": {
                "question_avg_score": self.question_avg_score,
                "exclaim_avg_score": self.exclaim_avg_score,
                "question_count": self.question_count,
                "exclaim_count": self.exclaim_count
            }
        }


def find_punctuation_events(reference_text: str) -> List[Tuple[str, str, int]]:
    """
    Find all '?' and '!' punctuation events in reference text.
    Unicode-safe: works with Latin, Cyrillic, Hebrew, and other scripts.

    Returns:
        List of (punctuation, word_before, word_index) tuples
    """
    events = []

    # Unicode-safe tokenization: matches any letter (including Cyrillic, Hebrew)
    # \p{L} = any Unicode letter, \p{M} = combining marks (for Hebrew niqqud etc.)
    # Using [\u0080-\uFFFF\w'] as Python re doesn't support \p{} directly
    # This pattern matches: ASCII word chars + apostrophe + any non-ASCII letters
    tokens = re.findall(r"[a-zA-Z0-9_'\u0080-\uFFFF]+|[?!]", reference_text)

    word_index = 0
    last_word = None
    last_word_index = -1

    for token in tokens:
        if token in ['?', '!']:
            if last_word is not None:
                events.append((token, last_word.lower(), last_word_index))
        else:
            last_word = token
            last_word_index = word_index
            word_index += 1

    return events


def map_event_to_alignment(
    ref_word: str,
    ref_word_index: int,
    alignment: List[dict]
) -> Tuple[Optional[str], Optional[float]]:
    """
    Map a punctuation event to aligned word timestamps.

    Args:
        ref_word: The reference word before punctuation
        ref_word_index: Index of the word in reference
        alignment: List of alignment items with ref, hyp, tag, t0, t1

    Returns:
        Tuple of (aligned_word, t_anchor) or (None, None) if not found
    """
    # Count reference words in alignment to find the right one
    ref_count = 0
    target_item = None

    for item in alignment:
        if item.get("tag") in ["ok", "sub", "del"]:
            if ref_count == ref_word_index:
                target_item = item
                break
            ref_count += 1

    if target_item is None:
        return None, None

    # Check if we have valid timestamps
    if target_item.get("tag") == "del":
        # Deletion - no timestamp, try to find nearest previous with timestamps
        return _find_nearest_timestamp(alignment, ref_word_index)

    aligned_word = target_item.get("hyp")
    t_anchor = target_item.get("t1")

    if t_anchor is None:
        return _find_nearest_timestamp(alignment, ref_word_index)

    return aligned_word, t_anchor


def _find_nearest_timestamp(
    alignment: List[dict],
    target_index: int
) -> Tuple[Optional[str], Optional[float]]:
    """Find nearest previous aligned word with valid timestamp."""
    ref_count = 0
    best_item = None

    for item in alignment:
        if item.get("tag") in ["ok", "sub", "del"]:
            if ref_count >= target_index:
                break
            if item.get("t1") is not None and item.get("tag") != "del":
                best_item = item
            ref_count += 1

    if best_item:
        return best_item.get("hyp"), best_item.get("t1")

    return None, None


def extract_window_features(
    t_anchor: float,
    window_sec: float,
    pitch_times: List[float],
    pitch_values: List[Optional[float]],
    energy_times: List[float],
    energy_values: List[float],
    config: PunctuationConfig
) -> Tuple[Dict[str, Optional[float]], float, List[str]]:
    """
    Extract prosody features from a time window ending at t_anchor.
    Implements progressive window expansion if insufficient voiced frames.

    Args:
        t_anchor: End time of analysis window
        window_sec: Initial window duration in seconds
        pitch_times: Time points for pitch values
        pitch_values: F0 values in Hz (None for unvoiced)
        energy_times: Time points for energy values
        energy_values: RMS energy values
        config: Configuration with expansion settings

    Returns:
        Tuple of (features_dict, actual_window_used, notes)
    """
    import numpy as np

    features = {
        "pitch_start_hz": None,
        "pitch_end_hz": None,
        "pitch_delta_hz": None,
        "pitch_slope_hz_per_s": None,
        "pitch_delta_semitones": None,
        "pitch_slope_semitones_per_s": None,
        "pitch_std_hz": None,
        "pitch_range_semitones": None,
        "energy_start": None,
        "energy_end": None,
        "energy_delta_ratio": None,
        "energy_std": None,
    }

    notes = []
    actual_window = window_sec
    edge_duration = 0.08  # 80ms for start/end median calculation

    # Convert to numpy arrays for easier manipulation
    if pitch_times and pitch_values:
        pitch_times_arr = np.array(pitch_times)
        # Convert None to NaN for numpy operations
        pitch_values_arr = np.array([v if v is not None else np.nan for v in pitch_values])
    else:
        pitch_times_arr = np.array([])
        pitch_values_arr = np.array([])

    if energy_times and energy_values:
        energy_times_arr = np.array(energy_times)
        energy_values_arr = np.array(energy_values)
    else:
        energy_times_arr = np.array([])
        energy_values_arr = np.array([])

    # Try progressive window expansion for pitch
    voiced_pitch = np.array([])
    voiced_times = np.array([])

    for try_window in config.window_expansion:
        if try_window < window_sec:
            continue

        t_start = t_anchor - try_window

        if len(pitch_times_arr) > 0:
            # Get window indices
            window_mask = (pitch_times_arr >= t_start) & (pitch_times_arr <= t_anchor)
            window_pitch = pitch_values_arr[window_mask]
            window_times = pitch_times_arr[window_mask]

            # Filter to voiced frames (non-NaN and positive)
            voiced_mask = np.isfinite(window_pitch) & (window_pitch > 0)
            voiced_pitch = window_pitch[voiced_mask]
            voiced_times = window_times[voiced_mask]

            if len(voiced_pitch) >= config.min_voiced_frames:
                actual_window = try_window
                if try_window > window_sec:
                    notes.append(f"Window expanded to {try_window}s to find voiced frames")
                break

    # Extract pitch features if we have enough voiced frames
    if len(voiced_pitch) >= config.min_voiced_frames:
        t_start = t_anchor - actual_window

        # Start region: first 80ms of voiced frames in window
        start_mask = voiced_times < (t_start + edge_duration)
        if not start_mask.any():
            # Fall back to first few frames
            start_pitch = voiced_pitch[:max(1, len(voiced_pitch) // 4)]
        else:
            start_pitch = voiced_pitch[start_mask]

        # End region: last 80ms of voiced frames in window
        end_mask = voiced_times > (t_anchor - edge_duration)
        if not end_mask.any():
            # Fall back to last few frames
            end_pitch = voiced_pitch[-max(1, len(voiced_pitch) // 4):]
        else:
            end_pitch = voiced_pitch[end_mask]

        if len(start_pitch) > 0 and len(end_pitch) > 0:
            features["pitch_start_hz"] = float(np.median(start_pitch))
            features["pitch_end_hz"] = float(np.median(end_pitch))
            features["pitch_delta_hz"] = features["pitch_end_hz"] - features["pitch_start_hz"]
            features["pitch_slope_hz_per_s"] = features["pitch_delta_hz"] / actual_window

            # Compute semitone-based delta (normalized across speaker ranges)
            features["pitch_delta_semitones"] = hz_to_semitones(
                features["pitch_start_hz"], features["pitch_end_hz"]
            )
            features["pitch_slope_semitones_per_s"] = features["pitch_delta_semitones"] / actual_window

        features["pitch_std_hz"] = float(np.std(voiced_pitch))

        # Also compute pitch range for exclamation analysis
        features["pitch_range_hz"] = float(np.max(voiced_pitch) - np.min(voiced_pitch))
        features["pitch_range_semitones"] = hz_to_semitones(
            float(np.min(voiced_pitch)), float(np.max(voiced_pitch))
        )
    else:
        if len(pitch_times_arr) > 0:
            notes.append(f"Insufficient voiced frames ({len(voiced_pitch)}) even after window expansion")
        else:
            notes.append("No pitch data available")

    # Extract energy features (no expansion needed, energy is always available)
    t_start = t_anchor - actual_window

    if len(energy_times_arr) > 0:
        window_mask = (energy_times_arr >= t_start) & (energy_times_arr <= t_anchor)
        window_energy = energy_values_arr[window_mask]
        window_etimes = energy_times_arr[window_mask]

        if len(window_energy) >= 2:
            # Start region
            start_mask = window_etimes < (t_start + edge_duration)
            start_energy = window_energy[start_mask] if start_mask.any() else window_energy[:2]

            # End region
            end_mask = window_etimes > (t_anchor - edge_duration)
            end_energy = window_energy[end_mask] if end_mask.any() else window_energy[-2:]

            if len(start_energy) > 0 and len(end_energy) > 0:
                features["energy_start"] = float(np.median(start_energy))
                features["energy_end"] = float(np.median(end_energy))

                eps = 1e-10
                features["energy_delta_ratio"] = features["energy_end"] / max(features["energy_start"], eps)

            features["energy_std"] = float(np.std(window_energy))

    return features, actual_window, notes


def classify_question(
    features: Dict[str, Optional[float]],
    config: PunctuationConfig
) -> Tuple[str, Optional[float], List[str]]:
    """
    Classify question intonation and compute score.

    Uses semitone-based thresholds by default for better normalization
    across different speaker pitch ranges (male vs female voices).

    Falls back to energy-based estimation if pitch data is unavailable.

    Returns:
        (classification, score, notes)
    """
    notes = []

    pitch_delta_hz = features.get("pitch_delta_hz")
    pitch_slope_hz = features.get("pitch_slope_hz_per_s")
    pitch_delta_st = features.get("pitch_delta_semitones")
    pitch_slope_st = features.get("pitch_slope_semitones_per_s")

    # Check if we have enough pitch data
    if pitch_delta_hz is None or pitch_slope_hz is None:
        # Try energy-based fallback
        energy_ratio = features.get("energy_delta_ratio")
        energy_std = features.get("energy_std")

        if energy_ratio is not None:
            # Use energy as a weak proxy for intonation
            # Rising energy at end often correlates with questions
            notes.append("Pitch data insufficient; using energy-based estimation")

            if energy_ratio > 1.1:
                # Energy rises toward end - weak signal for question
                classification = "rising"
                score = 55.0  # Slightly above neutral
                notes.append(f"Energy rises at end (ratio={energy_ratio:.2f})")
            elif energy_ratio < 0.85:
                # Energy drops - less typical for questions
                classification = "falling"
                score = 40.0
                notes.append(f"Energy drops at end (ratio={energy_ratio:.2f})")
            else:
                classification = "flat"
                score = 50.0  # Neutral
                notes.append(f"Energy stable (ratio={energy_ratio:.2f})")

            return classification, round(score, 1), notes

        # No usable data at all
        return "unknown", 50.0, ["Insufficient pitch and energy data for classification"]

    # Use semitone-based classification if enabled and available
    if config.use_semitones and pitch_delta_st is not None and pitch_slope_st is not None:
        # Semitone-based thresholds (more sensitive and normalized)
        delta_thresh = config.question_pitch_delta_semitones
        slope_thresh = config.question_pitch_slope_semitones_per_s

        is_rising = (pitch_delta_st >= delta_thresh or pitch_slope_st >= slope_thresh)
        is_falling = (pitch_delta_st <= -delta_thresh)

        if is_rising:
            classification = "rising"
            rise_factor = max(
                pitch_delta_st / delta_thresh if delta_thresh > 0 else 0,
                pitch_slope_st / slope_thresh if slope_thresh > 0 else 0
            )
            # More gradual scoring: 60 at threshold, 100 at 2.5x threshold
            score = min(100, 60 + (rise_factor - 1) * 26.67)
            notes.append(
                f"Rising intonation detected (delta={pitch_delta_st:.2f}st, "
                f"slope={pitch_slope_st:.2f}st/s, {pitch_delta_hz:.1f}Hz)"
            )
        elif is_falling:
            classification = "falling"
            fall_factor = abs(pitch_delta_st) / delta_thresh
            # Falling intonation: score drops from 40 at threshold to 15 at 2x
            score = max(15, 40 - (fall_factor - 1) * 25)
            notes.append(
                f"Falling intonation detected (delta={pitch_delta_st:.2f}st, {pitch_delta_hz:.1f}Hz)"
            )
        else:
            # Near-flat: use continuous scoring based on direction
            classification = "flat"
            # Give partial credit for slight rises, penalize slight falls
            if pitch_delta_st > 0:
                # Slight rise: score 50-60 based on how close to threshold
                partial_factor = pitch_delta_st / delta_thresh
                score = 50 + partial_factor * 10
            else:
                # Slight fall: score 40-50 based on how close to threshold
                partial_factor = abs(pitch_delta_st) / delta_thresh
                score = 50 - partial_factor * 10
            notes.append(
                f"Flat intonation (delta={pitch_delta_st:.2f}st, {pitch_delta_hz:.1f}Hz)"
            )
    else:
        # Fallback to Hz-based classification with lowered thresholds
        delta_thresh = config.question_pitch_delta_hz
        slope_thresh = config.question_pitch_slope_hz_per_s

        if pitch_delta_hz >= delta_thresh or pitch_slope_hz >= slope_thresh:
            classification = "rising"
            rise_factor = max(
                pitch_delta_hz / delta_thresh if delta_thresh > 0 else 0,
                pitch_slope_hz / slope_thresh if slope_thresh > 0 else 0
            )
            score = min(100, 60 + (rise_factor - 1) * 26.67)
            notes.append(f"Rising intonation detected (delta={pitch_delta_hz:.1f}Hz, slope={pitch_slope_hz:.1f}Hz/s)")
        elif pitch_delta_hz <= -delta_thresh:
            classification = "falling"
            fall_factor = abs(pitch_delta_hz) / delta_thresh
            score = max(15, 40 - (fall_factor - 1) * 25)
            notes.append(f"Falling intonation detected (delta={pitch_delta_hz:.1f}Hz)")
        else:
            classification = "flat"
            # Continuous scoring for near-flat
            if pitch_delta_hz > 0:
                partial_factor = pitch_delta_hz / delta_thresh
                score = 50 + partial_factor * 10
            else:
                partial_factor = abs(pitch_delta_hz) / delta_thresh
                score = 50 - partial_factor * 10
            notes.append(f"Flat intonation (delta={pitch_delta_hz:.1f}Hz)")

    return classification, round(score, 1), notes


def classify_exclamation(
    features: Dict[str, Optional[float]],
    config: PunctuationConfig
) -> Tuple[str, Optional[float], List[str]]:
    """
    Classify exclamation emphasis and compute score.

    Returns:
        (classification, score, notes)
    """
    notes = []
    score = 50.0  # Neutral baseline

    energy_ratio = features.get("energy_delta_ratio")
    pitch_std = features.get("pitch_std_hz")
    pitch_range_hz = features.get("pitch_range_hz")
    pitch_range_st = features.get("pitch_range_semitones")

    emphasis_signals = 0
    has_any_data = False

    # Check energy boost
    if energy_ratio is not None:
        has_any_data = True
        if energy_ratio >= config.exclaim_energy_ratio:
            emphasis_signals += 1
            score += (energy_ratio - 1) * 30  # Boost score for energy
            notes.append(f"Energy boost detected (ratio={energy_ratio:.2f})")
        elif energy_ratio < 0.8:
            score -= 15
            notes.append(f"Energy drop detected (ratio={energy_ratio:.2f})")

    # Check pitch variability (std)
    if pitch_std is not None:
        has_any_data = True
        if pitch_std >= config.exclaim_pitch_std_hz:
            emphasis_signals += 1
            score += (pitch_std / config.exclaim_pitch_std_hz - 1) * 20
            notes.append(f"High pitch variability (std={pitch_std:.1f}Hz)")

    # Check pitch range (prefer semitone-based if available)
    if config.use_semitones and pitch_range_st is not None:
        has_any_data = True
        if pitch_range_st >= config.exclaim_pitch_range_semitones:
            emphasis_signals += 1
            score += (pitch_range_st / config.exclaim_pitch_range_semitones - 1) * 15
            notes.append(f"Wide pitch range ({pitch_range_st:.1f}st, {pitch_range_hz:.1f}Hz)")
    elif pitch_range_hz is not None:
        has_any_data = True
        if pitch_range_hz >= config.exclaim_pitch_range_hz:
            emphasis_signals += 1
            score += (pitch_range_hz / config.exclaim_pitch_range_hz - 1) * 15
            notes.append(f"Wide pitch range ({pitch_range_hz:.1f}Hz)")

    # Classify
    if not has_any_data:
        classification = "unknown"
        score = 50.0  # Neutral fallback instead of None
        notes.append("Insufficient data for classification; using neutral score")
    elif emphasis_signals == 0:
        classification = "neutral"
        notes.append("No strong emphasis detected")
    elif emphasis_signals >= 2:
        classification = "emphatic"
    else:
        classification = "mild_emphasis"

    score = round(min(100, max(0, score)), 1)

    return classification, score, notes


def analyze_punctuation_prosody(
    reference_text: str,
    alignment: List[dict],
    pitch_times: List[float],
    pitch_values: List[Optional[float]],
    energy_times: List[float],
    energy_values: List[float],
    config: Optional[PunctuationConfig] = None
) -> PunctuationProsodyResult:
    """
    Analyze prosody at punctuation boundaries.

    Args:
        reference_text: Original reference text
        alignment: Word alignment from accuracy analysis
        pitch_times: Time points for pitch contour (seconds)
        pitch_values: F0 values in Hz (None for unvoiced)
        energy_times: Time points for energy contour (seconds)
        energy_values: RMS energy values
        config: Optional configuration

    Returns:
        PunctuationProsodyResult with analysis for each punctuation event
    """
    if config is None:
        config = PunctuationConfig()

    # Find punctuation events
    punct_events = find_punctuation_events(reference_text)

    events = []
    question_scores = []
    exclaim_scores = []

    for punct, ref_word, ref_word_index in punct_events:
        event = PunctuationEvent(
            punct=punct,
            ref_word=ref_word,
            ref_word_index=ref_word_index
        )

        # Map to alignment
        aligned_word, t_anchor = map_event_to_alignment(
            ref_word, ref_word_index, alignment
        )

        event.aligned_word = aligned_word
        event.t_anchor = t_anchor

        if t_anchor is None:
            event.notes.append("Could not map punctuation to aligned word with timestamp")
            events.append(event)
            continue

        # Extract features with window expansion
        features, window_used, feature_notes = extract_window_features(
            t_anchor,
            config.window_sec,
            pitch_times,
            pitch_values,
            energy_times,
            energy_values,
            config
        )

        event.features = features
        event.window_used_sec = window_used
        event.notes.extend(feature_notes)

        # Classify based on punctuation type
        if punct == "?":
            classification, score, class_notes = classify_question(features, config)
            event.classification = classification
            event.score_0_100 = score
            event.notes.extend(class_notes)
            if score is not None:
                question_scores.append(score)
        else:  # '!'
            classification, score, class_notes = classify_exclamation(features, config)
            event.classification = classification
            event.score_0_100 = score
            event.notes.extend(class_notes)
            if score is not None:
                exclaim_scores.append(score)

        events.append(event)

    # Calculate summary
    question_avg = sum(question_scores) / len(question_scores) if question_scores else None
    exclaim_avg = sum(exclaim_scores) / len(exclaim_scores) if exclaim_scores else None

    if question_avg is not None:
        question_avg = round(question_avg, 1)
    if exclaim_avg is not None:
        exclaim_avg = round(exclaim_avg, 1)

    return PunctuationProsodyResult(
        window_sec=config.window_sec,
        events=events,
        question_avg_score=question_avg,
        exclaim_avg_score=exclaim_avg,
        question_count=len([e for e in events if e.punct == "?"]),
        exclaim_count=len([e for e in events if e.punct == "!"])
    )
