"""
Tests for fluency analysis module.
"""

import pytest
from readscore.fluency import (
    analyze_fluency,
    FluencyConfig,
    FluencyResult,
    PauseStats
)


class TestFluencyConfig:
    def test_defaults(self):
        config = FluencyConfig()
        assert config.wpm_min == 110.0
        assert config.wpm_max == 170.0
        assert config.pause_thresholds_ms == [250, 500, 1000, 2000]

    def test_from_dict(self):
        config = FluencyConfig.from_dict({
            "wpm_min": 100,
            "wpm_max": 180,
            "pause_thresholds_ms": [200, 400, 800]
        })
        assert config.wpm_min == 100
        assert config.wpm_max == 180
        assert config.pause_thresholds_ms == [200, 400, 800]


class TestAnalyzeFluency:
    def test_normal_speech(self):
        # Simulate 10 words over 5 seconds = 120 WPM
        timestamps = [(i * 0.4, i * 0.4 + 0.3) for i in range(10)]
        total_duration = 5.0

        result = analyze_fluency(timestamps, total_duration)

        assert result.wpm > 100
        assert result.wpm < 200
        assert result.within_range
        assert result.score_0_100 > 50

    def test_slow_speech(self):
        # Simulate 10 words over 10 seconds = 60 WPM
        timestamps = [(i * 0.9, i * 0.9 + 0.5) for i in range(10)]
        total_duration = 10.0

        result = analyze_fluency(timestamps, total_duration)

        assert result.wpm < 110
        assert not result.within_range
        assert any("below" in note.lower() for note in result.notes)

    def test_fast_speech(self):
        # Simulate 30 words over 8 seconds = 225 WPM
        timestamps = [(i * 0.25, i * 0.25 + 0.2) for i in range(30)]
        total_duration = 8.0

        result = analyze_fluency(timestamps, total_duration)

        assert result.wpm > 170
        assert not result.within_range
        assert any("above" in note.lower() for note in result.notes)

    def test_with_pauses(self):
        # Words with pauses between them
        timestamps = [
            (0.0, 0.3),
            (0.5, 0.8),   # 0.2s pause
            (1.5, 1.8),   # 0.7s pause
            (3.0, 3.3),   # 1.2s pause
        ]
        total_duration = 4.0

        result = analyze_fluency(timestamps, total_duration)

        assert result.pauses.count == 3
        assert result.pauses.buckets_ms["250"] >= 2  # 2 pauses >= 250ms
        assert result.pauses.buckets_ms["500"] >= 2  # 2 pauses >= 500ms
        assert result.pauses.buckets_ms["1000"] >= 1  # 1 pause >= 1000ms

    def test_empty_timestamps(self):
        result = analyze_fluency([], 5.0)

        assert result.wpm == 0
        assert result.score_0_100 == 0
        assert not result.within_range

    def test_zero_duration(self):
        timestamps = [(0.0, 0.5)]
        result = analyze_fluency(timestamps, 0.0)

        assert result.total_dur_sec == 0.0

    def test_average_word_duration(self):
        timestamps = [
            (0.0, 0.3),   # 0.3s
            (0.5, 0.9),   # 0.4s
            (1.0, 1.5),   # 0.5s
        ]
        total_duration = 2.0

        result = analyze_fluency(timestamps, total_duration)

        assert result.avg_word_dur_sec == pytest.approx(0.4, rel=0.01)


class TestFluencyResult:
    def test_to_dict(self):
        result = FluencyResult(
            wpm=140.0,
            avg_word_dur_sec=0.35,
            total_dur_sec=10.0,
            pauses=PauseStats(
                count=3,
                total_duration=1.5,
                mean=0.5,
                p50=0.4,
                p90=0.8,
                buckets_ms={"250": 3, "500": 2, "1000": 0, "2000": 0}
            ),
            range_wpm_min=110.0,
            range_wpm_max=170.0,
            score_0_100=85.0,
            within_range=True,
            notes=["Good pace"]
        )

        d = result.to_dict()

        assert d["wpm"] == 140.0
        assert d["range"]["wpm_min"] == 110.0
        assert d["range"]["wpm_max"] == 170.0
        assert d["within_range"] is True
        assert d["pauses"]["count"] == 3
        assert "Good pace" in d["notes"]
