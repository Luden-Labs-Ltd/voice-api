"""
Tests for punctuation-aware prosody analysis module.
"""

import json
import pytest
from readscore.prosody_punct import (
    find_punctuation_events,
    map_event_to_alignment,
    extract_window_features,
    classify_question,
    classify_exclamation,
    analyze_punctuation_prosody,
    hz_to_semitones,
    PunctuationConfig,
    PunctuationEvent,
)


class TestFindPunctuationEvents:
    """Tests for punctuation event detection."""

    def test_single_question(self):
        text = "How are you?"
        events = find_punctuation_events(text)
        assert len(events) == 1
        assert events[0] == ("?", "you", 2)

    def test_single_exclamation(self):
        text = "Hello world!"
        events = find_punctuation_events(text)
        assert len(events) == 1
        assert events[0] == ("!", "world", 1)

    def test_multiple_punctuation(self):
        text = "How are you? I am fine! What about you?"
        events = find_punctuation_events(text)
        assert len(events) == 3
        assert events[0] == ("?", "you", 2)
        assert events[1] == ("!", "fine", 5)
        assert events[2] == ("?", "you", 8)

    def test_no_punctuation(self):
        text = "Hello world"
        events = find_punctuation_events(text)
        assert len(events) == 0

    def test_mixed_punctuation(self):
        text = "Wow! Is that true? Amazing!"
        events = find_punctuation_events(text)
        assert len(events) == 3
        punct_types = [e[0] for e in events]
        assert punct_types == ["!", "?", "!"]

    def test_punctuation_preserves_word_index(self):
        text = "First second third? Fourth fifth!"
        events = find_punctuation_events(text)
        assert events[0] == ("?", "third", 2)
        assert events[1] == ("!", "fifth", 4)


class TestMapEventToAlignment:
    """Tests for mapping punctuation events to alignment."""

    def test_exact_match(self):
        alignment = [
            {"ref": "hello", "hyp": "hello", "tag": "ok", "t0": 0.0, "t1": 0.5},
            {"ref": "world", "hyp": "world", "tag": "ok", "t0": 0.6, "t1": 1.0},
        ]
        word, t_anchor = map_event_to_alignment("world", 1, alignment)
        assert word == "world"
        assert t_anchor == 1.0

    def test_substitution_match(self):
        alignment = [
            {"ref": "hello", "hyp": "hello", "tag": "ok", "t0": 0.0, "t1": 0.5},
            {"ref": "world", "hyp": "word", "tag": "sub", "t0": 0.6, "t1": 1.0},
        ]
        word, t_anchor = map_event_to_alignment("world", 1, alignment)
        assert word == "word"
        assert t_anchor == 1.0

    def test_deletion_fallback(self):
        alignment = [
            {"ref": "hello", "hyp": "hello", "tag": "ok", "t0": 0.0, "t1": 0.5},
            {"ref": "world", "hyp": None, "tag": "del", "t0": None, "t1": None},
        ]
        word, t_anchor = map_event_to_alignment("world", 1, alignment)
        assert word == "hello"
        assert t_anchor == 0.5

    def test_no_match(self):
        alignment = [
            {"ref": "hello", "hyp": "hello", "tag": "ok", "t0": 0.0, "t1": 0.5},
        ]
        word, t_anchor = map_event_to_alignment("world", 5, alignment)
        assert word is None
        assert t_anchor is None


class TestExtractWindowFeatures:
    """Tests for feature extraction from prosody contours."""

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy not installed"
    )
    def test_rising_pitch_extraction(self):
        """Test that rising pitch is correctly extracted."""
        import numpy as np

        config = PunctuationConfig()

        # Create mock pitch data with clear rising pattern
        # 100 frames at 10ms = 1 second, pitch rises from 150 to 200 Hz
        pitch_times = [i * 0.01 for i in range(100)]
        pitch_values = [150.0 + i * 0.5 for i in range(100)]  # Rising pitch

        energy_times = [i * 0.032 for i in range(32)]
        energy_values = [0.05 for _ in range(32)]

        features, window_used, notes = extract_window_features(
            t_anchor=0.9,
            window_sec=0.45,
            pitch_times=pitch_times,
            pitch_values=pitch_values,
            energy_times=energy_times,
            energy_values=energy_values,
            config=config
        )

        assert features["pitch_start_hz"] is not None
        assert features["pitch_end_hz"] is not None
        assert features["pitch_delta_hz"] is not None
        assert features["pitch_delta_hz"] > 0  # Rising pitch
        assert features["pitch_end_hz"] > features["pitch_start_hz"]

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy not installed"
    )
    def test_falling_pitch_extraction(self):
        """Test that falling pitch is correctly extracted."""
        import numpy as np

        config = PunctuationConfig()

        # Create mock pitch data with falling pattern
        pitch_times = [i * 0.01 for i in range(100)]
        pitch_values = [200.0 - i * 0.5 for i in range(100)]  # Falling pitch

        energy_times = [i * 0.032 for i in range(32)]
        energy_values = [0.05 for _ in range(32)]

        features, window_used, notes = extract_window_features(
            t_anchor=0.9,
            window_sec=0.45,
            pitch_times=pitch_times,
            pitch_values=pitch_values,
            energy_times=energy_times,
            energy_values=energy_values,
            config=config
        )

        assert features["pitch_delta_hz"] is not None
        assert features["pitch_delta_hz"] < 0  # Falling pitch

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy not installed"
    )
    def test_window_expansion(self):
        """Test that window expands when insufficient voiced frames."""
        import numpy as np

        config = PunctuationConfig(min_voiced_frames=3)

        # Sparse pitch data - only a few voiced frames
        pitch_times = [i * 0.01 for i in range(150)]
        # Most frames unvoiced (None), only frames 100-120 have pitch
        pitch_values = [None if i < 100 or i > 120 else 150.0 + i for i in range(150)]

        energy_times = [i * 0.032 for i in range(50)]
        energy_values = [0.05 for _ in range(50)]

        features, window_used, notes = extract_window_features(
            t_anchor=1.2,
            window_sec=0.45,
            pitch_times=pitch_times,
            pitch_values=pitch_values,
            energy_times=energy_times,
            energy_values=energy_values,
            config=config
        )

        # Window should have expanded to find voiced frames
        assert window_used >= 0.45
        assert features["pitch_start_hz"] is not None or any("expanded" in n for n in notes)

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy not installed"
    )
    def test_empty_data(self):
        """Test handling of empty pitch/energy data."""
        config = PunctuationConfig()

        features, window_used, notes = extract_window_features(
            t_anchor=1.0,
            window_sec=0.45,
            pitch_times=[],
            pitch_values=[],
            energy_times=[],
            energy_values=[],
            config=config
        )

        assert features["pitch_start_hz"] is None
        assert features["pitch_end_hz"] is None
        assert features["energy_start"] is None
        assert len(notes) > 0  # Should have a note about missing data

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy not installed"
    )
    def test_all_unvoiced(self):
        """Test handling when all frames are unvoiced."""
        config = PunctuationConfig()

        pitch_times = [i * 0.01 for i in range(100)]
        pitch_values = [None for _ in range(100)]  # All unvoiced

        energy_times = [i * 0.032 for i in range(32)]
        energy_values = [0.05 for _ in range(32)]

        features, window_used, notes = extract_window_features(
            t_anchor=0.9,
            window_sec=0.45,
            pitch_times=pitch_times,
            pitch_values=pitch_values,
            energy_times=energy_times,
            energy_values=energy_values,
            config=config
        )

        # Pitch features should be None
        assert features["pitch_start_hz"] is None
        assert features["pitch_delta_hz"] is None
        # Energy should still be extracted
        assert features["energy_start"] is not None
        # Should have note about insufficient voiced frames
        assert any("voiced" in n.lower() or "insufficient" in n.lower() for n in notes)


class TestClassifyQuestion:
    """Tests for question intonation classification."""

    def test_rising_intonation(self):
        config = PunctuationConfig()
        # Clear rising pitch: 150 -> 170 Hz = ~2.04 semitones
        features = {
            "pitch_start_hz": 150.0,
            "pitch_end_hz": 170.0,
            "pitch_delta_hz": 20.0,
            "pitch_slope_hz_per_s": 25.0,  # At 0.8s window
            "pitch_delta_semitones": hz_to_semitones(150.0, 170.0),  # ~2.04st
            "pitch_slope_semitones_per_s": hz_to_semitones(150.0, 170.0) / 0.8,
            "pitch_std_hz": 10.0,
        }
        classification, score, notes = classify_question(features, config)
        assert classification == "rising"
        assert score is not None
        assert score > 60  # Starts at 60 for rising
        assert any("Rising" in note for note in notes)

    def test_falling_intonation(self):
        config = PunctuationConfig()
        # Clear falling pitch: 170 -> 150 Hz = ~-2.04 semitones
        features = {
            "pitch_start_hz": 170.0,
            "pitch_end_hz": 150.0,
            "pitch_delta_hz": -20.0,
            "pitch_slope_hz_per_s": -25.0,  # At 0.8s window
            "pitch_delta_semitones": hz_to_semitones(170.0, 150.0),  # ~-2.04st
            "pitch_slope_semitones_per_s": hz_to_semitones(170.0, 150.0) / 0.8,
            "pitch_std_hz": 10.0,
        }
        classification, score, notes = classify_question(features, config)
        assert classification == "falling"
        assert score is not None
        assert score < 45  # Falling scores cap at 40, drops from there
        assert any("Falling" in note for note in notes)

    def test_flat_intonation(self):
        config = PunctuationConfig()
        # Small pitch change (0.23 semitones, below 0.8st threshold)
        features = {
            "pitch_start_hz": 150.0,
            "pitch_end_hz": 152.0,
            "pitch_delta_hz": 2.0,
            "pitch_slope_hz_per_s": 2.5,  # At 0.8s window
            "pitch_delta_semitones": hz_to_semitones(150.0, 152.0),  # ~0.23st
            "pitch_slope_semitones_per_s": hz_to_semitones(150.0, 152.0) / 0.8,
            "pitch_std_hz": 5.0,
        }
        classification, score, notes = classify_question(features, config)
        assert classification == "flat"
        # With continuous scoring, slight rise gets partial credit (50-60 range)
        assert 48 <= score <= 60

    def test_unknown_insufficient_data(self):
        config = PunctuationConfig()
        features = {
            "pitch_start_hz": None,
            "pitch_end_hz": None,
            "pitch_delta_hz": None,
            "pitch_slope_hz_per_s": None,
        }
        classification, score, notes = classify_question(features, config)
        assert classification == "unknown"
        assert score is None
        assert any("Insufficient" in note for note in notes)


class TestClassifyExclamation:
    """Tests for exclamation emphasis classification."""

    def test_emphatic_energy_boost(self):
        config = PunctuationConfig()
        features = {
            "energy_start": 0.05,
            "energy_end": 0.08,
            "energy_delta_ratio": 1.6,
            "energy_std": 0.02,
            "pitch_std_hz": 20.0,
            "pitch_range_hz": 40.0,
            "pitch_range_semitones": 5.0,  # Above 4.0st threshold
        }
        classification, score, notes = classify_exclamation(features, config)
        assert classification == "emphatic"
        assert score is not None
        assert score > 50

    def test_neutral_no_emphasis(self):
        config = PunctuationConfig()
        features = {
            "energy_start": 0.05,
            "energy_end": 0.05,
            "energy_delta_ratio": 1.0,
            "energy_std": 0.01,
            "pitch_std_hz": 5.0,
            "pitch_range_hz": 10.0,
            "pitch_range_semitones": 1.5,  # Below 4.0st threshold
        }
        classification, score, notes = classify_exclamation(features, config)
        assert classification == "neutral"
        assert score is not None
        assert score <= 50

    def test_unknown_no_data(self):
        config = PunctuationConfig()
        features = {
            "energy_start": None,
            "energy_end": None,
            "energy_delta_ratio": None,
            "pitch_std_hz": None,
            "pitch_range_hz": None,
            "pitch_range_semitones": None,
        }
        classification, score, notes = classify_exclamation(features, config)
        assert classification == "unknown"
        assert score is None


class TestAnalyzePunctuationProsody:
    """Integration tests for full punctuation prosody analysis."""

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy not installed"
    )
    def test_full_analysis_with_rising_pitch(self):
        """Test that a question with rising pitch gets classified as 'rising'."""
        import numpy as np

        reference_text = "How are you?"
        alignment = [
            {"ref": "how", "hyp": "how", "tag": "ok", "t0": 0.0, "t1": 0.3},
            {"ref": "are", "hyp": "are", "tag": "ok", "t0": 0.4, "t1": 0.6},
            {"ref": "you", "hyp": "you", "tag": "ok", "t0": 0.7, "t1": 1.0},
        ]

        # Create rising pitch contour ending at t=1.0
        pitch_times = [i * 0.01 for i in range(120)]
        # Rising pitch: starts at 150Hz, rises to 200Hz near the end
        pitch_values = [150.0 + (i * 0.5) for i in range(120)]

        energy_times = [i * 0.032 for i in range(35)]
        energy_values = [0.05 for _ in range(35)]

        result = analyze_punctuation_prosody(
            reference_text=reference_text,
            alignment=alignment,
            pitch_times=pitch_times,
            pitch_values=pitch_values,
            energy_times=energy_times,
            energy_values=energy_values
        )

        assert result.question_count == 1
        assert len(result.events) == 1
        event = result.events[0]
        assert event.punct == "?"
        assert event.features["pitch_delta_hz"] is not None
        assert event.features["pitch_delta_hz"] > 0  # Rising
        assert event.classification == "rising"
        assert event.score_0_100 is not None
        assert event.score_0_100 > 50

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy not installed"
    )
    def test_full_analysis_multiple_punctuation(self):
        """Test analysis with multiple punctuation marks."""
        import numpy as np

        reference_text = "How are you? I am great!"
        alignment = [
            {"ref": "how", "hyp": "how", "tag": "ok", "t0": 0.0, "t1": 0.3},
            {"ref": "are", "hyp": "are", "tag": "ok", "t0": 0.4, "t1": 0.6},
            {"ref": "you", "hyp": "you", "tag": "ok", "t0": 0.7, "t1": 1.0},
            {"ref": "i", "hyp": "i", "tag": "ok", "t0": 1.2, "t1": 1.3},
            {"ref": "am", "hyp": "am", "tag": "ok", "t0": 1.4, "t1": 1.6},
            {"ref": "great", "hyp": "great", "tag": "ok", "t0": 1.7, "t1": 2.0},
        ]

        pitch_times = [i * 0.01 for i in range(250)]
        pitch_values = [150.0 + 10 * np.sin(i * 0.1) for i in range(250)]
        energy_times = [i * 0.032 for i in range(70)]
        energy_values = [0.05 + 0.01 * np.sin(i * 0.2) for i in range(70)]

        result = analyze_punctuation_prosody(
            reference_text=reference_text,
            alignment=alignment,
            pitch_times=pitch_times,
            pitch_values=pitch_values,
            energy_times=energy_times,
            energy_values=energy_values
        )

        assert result.question_count == 1
        assert result.exclaim_count == 1
        assert len(result.events) == 2
        assert result.events[0].punct == "?"
        assert result.events[1].punct == "!"

    def test_no_punctuation(self):
        reference_text = "Hello world"
        alignment = [
            {"ref": "hello", "hyp": "hello", "tag": "ok", "t0": 0.0, "t1": 0.5},
            {"ref": "world", "hyp": "world", "tag": "ok", "t0": 0.6, "t1": 1.0},
        ]

        result = analyze_punctuation_prosody(
            reference_text=reference_text,
            alignment=alignment,
            pitch_times=[],
            pitch_values=[],
            energy_times=[],
            energy_values=[]
        )

        assert result.question_count == 0
        assert result.exclaim_count == 0
        assert len(result.events) == 0

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy not installed"
    )
    def test_no_voiced_frames_does_not_crash(self):
        """Ensure analysis completes even with no voiced frames."""
        reference_text = "Is this working?"
        alignment = [
            {"ref": "is", "hyp": "is", "tag": "ok", "t0": 0.0, "t1": 0.2},
            {"ref": "this", "hyp": "this", "tag": "ok", "t0": 0.3, "t1": 0.5},
            {"ref": "working", "hyp": "working", "tag": "ok", "t0": 0.6, "t1": 1.0},
        ]

        # All unvoiced
        pitch_times = [i * 0.01 for i in range(120)]
        pitch_values = [None for _ in range(120)]
        energy_times = [i * 0.032 for i in range(35)]
        energy_values = [0.05 for _ in range(35)]

        # Should not raise
        result = analyze_punctuation_prosody(
            reference_text=reference_text,
            alignment=alignment,
            pitch_times=pitch_times,
            pitch_values=pitch_values,
            energy_times=energy_times,
            energy_values=energy_values
        )

        assert result.question_count == 1
        event = result.events[0]
        assert event.classification == "unknown"
        assert event.score_0_100 is None


class TestJSONSerializable:
    """Test that results are JSON serializable."""

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy not installed"
    )
    def test_result_serializable(self):
        import numpy as np
        from readscore.report import convert_to_serializable

        reference_text = "Is this working?"
        alignment = [
            {"ref": "is", "hyp": "is", "tag": "ok", "t0": 0.0, "t1": 0.2},
            {"ref": "this", "hyp": "this", "tag": "ok", "t0": 0.3, "t1": 0.5},
            {"ref": "working", "hyp": "working", "tag": "ok", "t0": 0.6, "t1": 1.0},
        ]

        pitch_times = [i * 0.01 for i in range(120)]
        pitch_values = [150.0 + i * 0.3 for i in range(120)]
        energy_times = [i * 0.032 for i in range(35)]
        energy_values = [0.05 for _ in range(35)]

        result = analyze_punctuation_prosody(
            reference_text=reference_text,
            alignment=alignment,
            pitch_times=pitch_times,
            pitch_values=pitch_values,
            energy_times=energy_times,
            energy_values=energy_values
        )

        result_dict = result.to_dict()
        result_dict = convert_to_serializable(result_dict)

        # Should not raise
        json_str = json.dumps(result_dict, indent=2)
        assert json_str

        # Verify structure
        parsed = json.loads(json_str)
        assert "events" in parsed
        assert "summary" in parsed
        assert parsed["summary"]["question_count"] == 1


class TestPunctuationConfig:
    """Tests for configuration."""

    def test_default_config(self):
        config = PunctuationConfig()
        assert config.window_sec == 0.8  # Increased from 0.45
        assert config.question_pitch_delta_hz == 4.0  # Lowered from 8.0
        assert config.question_pitch_delta_semitones == 0.8  # New semitone threshold
        assert config.exclaim_energy_ratio == 1.25
        assert config.min_voiced_frames == 3
        assert config.use_semitones is True  # New default
        assert 0.8 in config.window_expansion

    def test_from_dict(self):
        d = {
            "window_sec": 0.5,
            "question_pitch_delta_hz": 10.0,
            "question_pitch_delta_semitones": 1.0,
            "exclaim_energy_ratio": 1.5,
            "min_voiced_frames": 5,
            "use_semitones": False,
        }
        config = PunctuationConfig.from_dict(d)
        assert config.window_sec == 0.5
        assert config.question_pitch_delta_hz == 10.0
        assert config.question_pitch_delta_semitones == 1.0
        assert config.exclaim_energy_ratio == 1.5
        assert config.min_voiced_frames == 5
        assert config.use_semitones is False


class TestHzToSemitones:
    """Tests for Hz to semitone conversion."""

    def test_octave_is_12_semitones(self):
        # 200Hz is one octave above 100Hz = 12 semitones
        st = hz_to_semitones(100.0, 200.0)
        assert abs(st - 12.0) < 0.001

    def test_same_frequency_is_zero(self):
        st = hz_to_semitones(150.0, 150.0)
        assert st == 0.0

    def test_falling_pitch_is_negative(self):
        st = hz_to_semitones(200.0, 100.0)
        assert st < 0
        assert abs(st - (-12.0)) < 0.001

    def test_small_rise_male_voice(self):
        # 10Hz rise at 100Hz (typical male) = about 1.66 semitones
        st = hz_to_semitones(100.0, 110.0)
        assert 1.6 < st < 1.7

    def test_small_rise_female_voice(self):
        # 10Hz rise at 200Hz (typical female) = about 0.85 semitones
        st = hz_to_semitones(200.0, 210.0)
        assert 0.8 < st < 0.9

    def test_zero_frequency_returns_zero(self):
        assert hz_to_semitones(0.0, 100.0) == 0.0
        assert hz_to_semitones(100.0, 0.0) == 0.0
