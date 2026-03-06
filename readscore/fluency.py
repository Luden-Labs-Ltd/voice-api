"""
Fluency (Speed) analysis module.

Measures:
- Words per minute (WPM)
- Pause distribution
- Speaking rate metrics
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import statistics


@dataclass
class FluencyConfig:
    """Configuration for fluency analysis."""
    wpm_min: float = 110.0  # Minimum expected WPM
    wpm_max: float = 170.0  # Maximum expected WPM
    pause_thresholds_ms: List[int] = field(default_factory=lambda: [250, 500, 1000, 2000])

    @classmethod
    def from_dict(cls, d: dict) -> "FluencyConfig":
        return cls(
            wpm_min=d.get("wpm_min", 110.0),
            wpm_max=d.get("wpm_max", 170.0),
            pause_thresholds_ms=d.get("pause_thresholds_ms", [250, 500, 1000, 2000])
        )


@dataclass
class PauseStats:
    """Statistics about pauses in speech."""
    count: int
    total_duration: float
    mean: float
    p50: float
    p90: float
    buckets_ms: Dict[str, int]  # Counts per threshold bucket


@dataclass
class FluencyResult:
    """Result of fluency analysis."""
    wpm: float
    avg_word_dur_sec: float
    total_dur_sec: float
    pauses: PauseStats
    range_wpm_min: float
    range_wpm_max: float
    score_0_100: float
    within_range: bool
    notes: List[str]

    def to_dict(self) -> dict:
        return {
            "wpm": round(self.wpm, 2),
            "avg_word_dur_sec": round(self.avg_word_dur_sec, 4),
            "total_dur_sec": round(self.total_dur_sec, 2),
            "pauses": {
                "count": self.pauses.count,
                "total_duration_sec": round(self.pauses.total_duration, 3),
                "mean": round(self.pauses.mean, 4) if self.pauses.mean else 0,
                "p50": round(self.pauses.p50, 4) if self.pauses.p50 else 0,
                "p90": round(self.pauses.p90, 4) if self.pauses.p90 else 0,
                "buckets_ms": self.pauses.buckets_ms
            },
            "range": {
                "wpm_min": self.range_wpm_min,
                "wpm_max": self.range_wpm_max
            },
            "score_0_100": round(self.score_0_100, 1),
            "within_range": self.within_range,
            "notes": self.notes
        }


def analyze_fluency(
    word_timestamps: List[tuple],
    total_duration: float,
    config: Optional[FluencyConfig] = None
) -> FluencyResult:
    """
    Analyze fluency/speed metrics from word timestamps.

    Args:
        word_timestamps: List of (start, end) times for each word
        total_duration: Total audio duration in seconds
        config: Optional configuration for reference ranges

    Returns:
        FluencyResult with speed and pause metrics
    """
    if config is None:
        config = FluencyConfig()

    notes = []

    # Handle edge cases
    if not word_timestamps or total_duration <= 0:
        return FluencyResult(
            wpm=0,
            avg_word_dur_sec=0,
            total_dur_sec=total_duration,
            pauses=PauseStats(0, 0, 0, 0, 0, {}),
            range_wpm_min=config.wpm_min,
            range_wpm_max=config.wpm_max,
            score_0_100=0,
            within_range=False,
            notes=["No words detected"]
        )

    num_words = len(word_timestamps)

    # Calculate speaking duration (from first word start to last word end)
    first_start = word_timestamps[0][0]
    last_end = word_timestamps[-1][1]
    speaking_duration = last_end - first_start

    if speaking_duration <= 0:
        speaking_duration = total_duration

    # Words per minute
    wpm = (num_words / speaking_duration) * 60 if speaking_duration > 0 else 0

    # Average word duration
    word_durations = [end - start for start, end in word_timestamps]
    avg_word_dur = statistics.mean(word_durations) if word_durations else 0

    # Pause analysis
    pauses = []
    for i in range(1, len(word_timestamps)):
        prev_end = word_timestamps[i - 1][1]
        curr_start = word_timestamps[i][0]
        gap = curr_start - prev_end
        if gap > 0.05:  # Only count gaps > 50ms as pauses
            pauses.append(gap)

    # Pause statistics
    pause_buckets = {str(t): 0 for t in config.pause_thresholds_ms}

    for pause in pauses:
        pause_ms = pause * 1000
        for threshold in config.pause_thresholds_ms:
            if pause_ms >= threshold:
                pause_buckets[str(threshold)] += 1

    if pauses:
        pause_mean = statistics.mean(pauses)
        sorted_pauses = sorted(pauses)
        pause_p50 = sorted_pauses[len(sorted_pauses) // 2]
        pause_p90_idx = int(len(sorted_pauses) * 0.9)
        pause_p90 = sorted_pauses[min(pause_p90_idx, len(sorted_pauses) - 1)]
        pause_total = sum(pauses)
    else:
        pause_mean = 0
        pause_p50 = 0
        pause_p90 = 0
        pause_total = 0

    pause_stats = PauseStats(
        count=len(pauses),
        total_duration=pause_total,
        mean=pause_mean,
        p50=pause_p50,
        p90=pause_p90,
        buckets_ms=pause_buckets
    )

    # Check if within range
    within_range = config.wpm_min <= wpm <= config.wpm_max

    # Calculate fluency score
    score = _calculate_fluency_score(wpm, pause_stats, config, notes)

    # Add notes based on analysis
    if wpm < config.wpm_min:
        notes.append(f"Speaking rate below expected range ({wpm:.0f} < {config.wpm_min:.0f} WPM)")
    elif wpm > config.wpm_max:
        notes.append(f"Speaking rate above expected range ({wpm:.0f} > {config.wpm_max:.0f} WPM)")

    if pause_stats.count > num_words * 0.3:
        notes.append("High frequency of pauses detected")

    if pause_p90 > 1.5:
        notes.append("Some very long pauses detected (>1.5s)")

    return FluencyResult(
        wpm=wpm,
        avg_word_dur_sec=avg_word_dur,
        total_dur_sec=total_duration,
        pauses=pause_stats,
        range_wpm_min=config.wpm_min,
        range_wpm_max=config.wpm_max,
        score_0_100=score,
        within_range=within_range,
        notes=notes
    )


def _calculate_fluency_score(
    wpm: float,
    pause_stats: PauseStats,
    config: FluencyConfig,
    notes: List[str]
) -> float:
    """Calculate overall fluency score 0-100."""
    score = 100.0

    # WPM component (50% of score)
    wpm_mid = (config.wpm_min + config.wpm_max) / 2
    wpm_range = (config.wpm_max - config.wpm_min) / 2

    if config.wpm_min <= wpm <= config.wpm_max:
        # Within range: small penalty for being far from middle
        wpm_deviation = abs(wpm - wpm_mid) / wpm_range
        wpm_score = 50 - (wpm_deviation * 5)  # Max 5 point penalty
    else:
        # Outside range: larger penalty
        if wpm < config.wpm_min:
            distance = (config.wpm_min - wpm) / config.wpm_min
        else:
            distance = (wpm - config.wpm_max) / config.wpm_max
        wpm_score = max(0, 50 - (distance * 50))

    score = wpm_score

    # Pause component (50% of score)
    pause_score = 50.0

    # Penalize for excessive pauses
    if pause_stats.count > 0:
        # Penalize long pauses
        if pause_stats.p90 > 2.0:
            pause_score -= 20
        elif pause_stats.p90 > 1.0:
            pause_score -= 10
        elif pause_stats.p90 > 0.5:
            pause_score -= 5

        # Penalize excessive total pause time
        # Assume reasonable pause ratio is ~20% of speech
        # We'd need total speech duration to calculate properly
        if pause_stats.mean > 0.8:
            pause_score -= 15
        elif pause_stats.mean > 0.5:
            pause_score -= 10

    pause_score = max(0, pause_score)
    score += pause_score

    return min(100, max(0, score))
