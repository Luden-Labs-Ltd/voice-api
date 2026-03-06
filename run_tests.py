#!/usr/bin/env python3
"""
Standalone test script - run without pytest.
Usage: python run_tests.py
"""

import json
import sys

# Add project to path
sys.path.insert(0, '.')

def test_convert_to_serializable():
    """Test the JSON serialization fix."""
    from readscore.report import convert_to_serializable

    print("Testing convert_to_serializable...")

    # Test 1: Standard types
    assert convert_to_serializable(None) is None
    assert convert_to_serializable(True) is True
    assert convert_to_serializable(42) == 42
    assert convert_to_serializable("hello") == "hello"
    print("  [PASS] Standard types")

    # Test 2: Dict/List
    data = {"a": 1, "b": [2, 3]}
    result = convert_to_serializable(data)
    assert json.dumps(result)
    print("  [PASS] Dict/List conversion")

    # Test 3: Numpy types (if available)
    try:
        import numpy as np

        # Test numpy.bool_
        data = {"flag": np.bool_(True)}
        result = convert_to_serializable(data)
        assert result["flag"] is True
        assert isinstance(result["flag"], bool)
        json_str = json.dumps(result)
        print("  [PASS] numpy.bool_ -> bool")

        # Test numpy integers
        data = {"count": np.int64(42)}
        result = convert_to_serializable(data)
        assert result["count"] == 42
        assert isinstance(result["count"], int)
        json.dumps(result)
        print("  [PASS] numpy.int64 -> int")

        # Test numpy floats
        data = {"score": np.float32(3.14)}
        result = convert_to_serializable(data)
        assert isinstance(result["score"], float)
        json.dumps(result)
        print("  [PASS] numpy.float32 -> float")

        # Test numpy array
        data = {"arr": np.array([1, 2, 3])}
        result = convert_to_serializable(data)
        assert result["arr"] == [1, 2, 3]
        json.dumps(result)
        print("  [PASS] numpy.ndarray -> list")

        # Test nested report structure
        report = {
            "accuracy": {
                "wer": np.float64(0.1),
                "counts": {"ins": np.int32(0), "del": np.int32(1)},
            },
            "fluency_speed": {
                "wpm": np.float64(150.0),
                "within_range": np.bool_(True),
            },
        }
        result = convert_to_serializable(report)
        json_str = json.dumps(result, indent=2)
        parsed = json.loads(json_str)
        assert parsed["fluency_speed"]["within_range"] is True
        print("  [PASS] Nested report with numpy types")

    except ImportError:
        print("  [SKIP] numpy not installed")

    print("\nAll serialization tests passed!")


def test_normalize():
    """Test text normalization."""
    from readscore.normalize import normalize_text, tokenize

    print("\nTesting normalize...")

    # Basic normalization
    assert normalize_text("Hello World", convert_numbers=False) == "hello world"
    print("  [PASS] Lowercase")

    # Punctuation removal
    result = normalize_text("Hello, world!", convert_numbers=False)
    assert "," not in result and "!" not in result
    print("  [PASS] Punctuation removal")

    # Tokenization
    words = tokenize("The quick brown fox")
    assert words == ["the", "quick", "brown", "fox"]
    print("  [PASS] Tokenization")

    print("\nAll normalize tests passed!")


def test_multilingual():
    """Test multi-language normalization (en, ru, he)."""
    from readscore.normalize import (
        detect_language, detect_script, tokenize, normalize_text,
        strip_hebrew_niqqud, normalize_russian
    )

    print("\nTesting multilingual normalization...")

    # Test script detection
    assert detect_script("Hello world") == "latin"
    assert detect_script("Привет мир") == "cyrillic"
    assert detect_script("שלום עולם") == "hebrew"
    print("  [PASS] Script detection")

    # Test language detection
    assert detect_language("The quick brown fox") == "en"
    assert detect_language("Привет, как дела?") == "ru"
    assert detect_language("שלום, מה שלומך?") == "he"
    print("  [PASS] Language detection")

    # Test Russian normalization (ё -> е)
    assert normalize_russian("ёлка") == "елка"
    assert normalize_russian("Ёжик") == "Ежик"
    print("  [PASS] Russian ё normalization")

    # Test Russian tokenization
    words = tokenize("Привет, мир!", lang="ru")
    assert words == ["привет", "мир"]
    print("  [PASS] Russian tokenization")

    # Test Hebrew niqqud stripping
    # Hebrew with vowel points (niqqud)
    hebrew_with_niqqud = "שָׁלוֹם"  # shalom with niqqud
    hebrew_stripped = strip_hebrew_niqqud(hebrew_with_niqqud)
    assert hebrew_stripped == "שלום"
    print("  [PASS] Hebrew niqqud stripping")

    # Test Hebrew tokenization
    words = tokenize("שלום, מה שלומך?", lang="he")
    assert len(words) == 3
    assert words[0] == "שלום"
    print("  [PASS] Hebrew tokenization")

    # Test auto-detection in tokenize
    words = tokenize("Как дела?", lang="auto")
    assert words == ["как", "дела"]
    print("  [PASS] Auto-detect Russian in tokenize")

    # Test mixed script handling
    mixed_text = "Hello привет שלום"
    script = detect_script(mixed_text)
    assert script in ["latin", "cyrillic", "hebrew"]  # Any is valid
    print("  [PASS] Mixed script handling")

    # Test number conversion in different languages
    assert "one" in normalize_text("I have 1 apple", lang="en")
    assert "один" in normalize_text("У меня 1 яблоко", lang="ru")
    print("  [PASS] Number conversion per language")

    print("\nAll multilingual tests passed!")


def test_align():
    """Test word alignment."""
    from readscore.align import levenshtein_align, AlignTag

    print("\nTesting align...")

    # Perfect match
    result = levenshtein_align(["hello", "world"], ["hello", "world"])
    assert result.wer == 0.0
    assert result.correct == 2
    print("  [PASS] Perfect match WER=0")

    # Single substitution
    result = levenshtein_align(["hello", "world"], ["hello", "word"])
    assert result.substitutions == 1
    assert result.wer == 0.5
    print("  [PASS] Single substitution")

    # Insertion
    result = levenshtein_align(["hello", "world"], ["hello", "big", "world"])
    assert result.insertions == 1
    print("  [PASS] Insertion detection")

    # Deletion
    result = levenshtein_align(["hello", "big", "world"], ["hello", "world"])
    assert result.deletions == 1
    print("  [PASS] Deletion detection")

    print("\nAll align tests passed!")


def test_prosody_punct():
    """Test punctuation prosody analysis."""
    from readscore.prosody_punct import (
        find_punctuation_events,
        map_event_to_alignment,
        classify_question,
        classify_exclamation,
        analyze_punctuation_prosody,
        PunctuationConfig,
    )

    print("\nTesting prosody_punct...")

    # Test punctuation detection
    events = find_punctuation_events("How are you?")
    assert len(events) == 1
    assert events[0] == ("?", "you", 2)
    print("  [PASS] Single question detection")

    events = find_punctuation_events("Wow! Is that true? Amazing!")
    assert len(events) == 3
    print("  [PASS] Multiple punctuation detection")

    # Test Unicode-safe punctuation detection (Russian)
    events = find_punctuation_events("Как дела?")
    assert len(events) == 1
    assert events[0] == ("?", "дела", 1)
    print("  [PASS] Russian punctuation detection")

    # Test Unicode-safe punctuation detection (Hebrew)
    events = find_punctuation_events("מה שלומך?")
    assert len(events) == 1
    assert events[0] == ("?", "שלומך", 1)
    print("  [PASS] Hebrew punctuation detection")

    # Test alignment mapping
    alignment = [
        {"ref": "hello", "hyp": "hello", "tag": "ok", "t0": 0.0, "t1": 0.5},
        {"ref": "world", "hyp": "world", "tag": "ok", "t0": 0.6, "t1": 1.0},
    ]
    word, t_anchor = map_event_to_alignment("world", 1, alignment)
    assert word == "world"
    assert t_anchor == 1.0
    print("  [PASS] Alignment mapping")

    # Test question classification
    config = PunctuationConfig()
    features = {
        "pitch_start_hz": 150.0,
        "pitch_end_hz": 170.0,
        "pitch_delta_hz": 20.0,
        "pitch_slope_hz_per_s": 44.4,
        "pitch_std_hz": 10.0,
    }
    classification, score, notes = classify_question(features, config)
    assert classification == "rising"
    assert score > 50
    print("  [PASS] Rising intonation classification")

    # Test exclamation classification
    features = {
        "energy_start": 0.05,
        "energy_end": 0.08,
        "energy_delta_ratio": 1.6,
        "energy_std": 0.02,
        "pitch_std_hz": 20.0,
        "pitch_range_hz": 40.0,
    }
    classification, score, notes = classify_exclamation(features, config)
    assert classification == "emphatic"
    assert score > 50
    print("  [PASS] Emphatic exclamation classification")

    # Test full analysis with rising pitch contour
    try:
        import numpy as np
        from readscore.report import convert_to_serializable

        reference_text = "How are you?"
        alignment = [
            {"ref": "how", "hyp": "how", "tag": "ok", "t0": 0.0, "t1": 0.3},
            {"ref": "are", "hyp": "are", "tag": "ok", "t0": 0.4, "t1": 0.6},
            {"ref": "you", "hyp": "you", "tag": "ok", "t0": 0.7, "t1": 1.0},
        ]

        # Rising pitch contour
        pitch_times = [i * 0.01 for i in range(120)]
        pitch_values = [150.0 + i * 0.5 for i in range(120)]  # Rising from 150 to 210 Hz
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
        event = result.events[0]
        assert event.features["pitch_delta_hz"] is not None, "pitch_delta_hz should not be None"
        assert event.features["pitch_delta_hz"] > 0, f"Expected rising pitch, got delta={event.features['pitch_delta_hz']}"
        assert event.classification == "rising", f"Expected 'rising', got '{event.classification}'"
        assert event.score_0_100 is not None and event.score_0_100 > 50
        print("  [PASS] Full analysis: rising pitch correctly classified")

        # Test JSON serialization
        result_dict = convert_to_serializable(result.to_dict())
        json_str = json.dumps(result_dict, indent=2)
        assert json_str
        print("  [PASS] JSON serialization works")

        # Test with no voiced frames - now uses energy-based fallback
        pitch_values_unvoiced = [None for _ in range(120)]
        result_unvoiced = analyze_punctuation_prosody(
            reference_text=reference_text,
            alignment=alignment,
            pitch_times=pitch_times,
            pitch_values=pitch_values_unvoiced,
            energy_times=energy_times,
            energy_values=energy_values
        )
        # With energy fallback, we get a classification based on energy trend
        # instead of "unknown"
        assert result_unvoiced.events[0].classification in ["rising", "falling", "flat"]
        assert result_unvoiced.events[0].score_0_100 is not None  # Always provides a score now
        assert any("energy" in note.lower() for note in result_unvoiced.events[0].notes)
        print("  [PASS] Unvoiced frames: energy-based fallback works")

    except ImportError:
        print("  [SKIP] Full analysis (numpy not installed)")

    print("\nAll prosody_punct tests passed!")


def test_punctuation_pauses():
    """Test punctuation pause analysis."""
    from readscore.punctuation_pauses import (
        parse_punctuation_events,
        classify_pause,
        analyze_punctuation_pauses,
        PauseConfig,
    )

    print("\nTesting punctuation_pauses...")

    # Test punctuation parsing
    events = parse_punctuation_events("Hello. How are you?")
    assert len(events) == 2
    assert events[0] == (".", "hello", 0)
    assert events[1] == ("?", "you", 3)
    print("  [PASS] Punctuation parsing")

    # Test with multiple punctuation types
    events = parse_punctuation_events("Wait, what? Yes! Really...")
    assert len(events) == 4
    punct_types = [e[0] for e in events]
    assert punct_types == [",", "?", "!", "…"]
    print("  [PASS] Multiple punctuation types")

    # Test pause classification
    classification, score, notes = classify_pause(0.35, 0.25, 1.20)  # Period
    assert classification == "ok"
    assert score > 80
    print("  [PASS] OK pause classification")

    classification, score, notes = classify_pause(0.05, 0.25, 1.20)  # Too short
    assert classification == "too_short"
    assert score < 70
    print("  [PASS] Too short pause classification")

    classification, score, notes = classify_pause(2.0, 0.25, 1.20)  # Too long
    assert classification == "too_long"
    assert score < 70
    print("  [PASS] Too long pause classification")

    # Test full analysis
    alignment = [
        {"ref": "hello", "hyp": "hello", "tag": "ok", "t0": 0.0, "t1": 0.5},
        {"ref": "how", "hyp": "how", "tag": "ok", "t0": 1.0, "t1": 1.3},
        {"ref": "are", "hyp": "are", "tag": "ok", "t0": 1.4, "t1": 1.6},
        {"ref": "you", "hyp": "you", "tag": "ok", "t0": 1.7, "t1": 2.0},
    ]

    result = analyze_punctuation_pauses("Hello. How are you?", alignment)
    assert result.count == 2
    assert result.events[0].punct == "."
    assert result.events[0].pause_sec == 0.5  # 1.0 - 0.5
    assert result.events[0].classification == "ok"
    print("  [PASS] Full pause analysis")

    # Test with deletion (missing word)
    alignment_with_del = [
        {"ref": "hello", "hyp": None, "tag": "del", "t0": None, "t1": None},
        {"ref": "world", "hyp": "world", "tag": "ok", "t0": 0.5, "t1": 1.0},
    ]
    result = analyze_punctuation_pauses("Hello. World.", alignment_with_del)
    assert result.events[0].classification == "missing"
    print("  [PASS] Deleted word handling")

    print("\nAll punctuation_pauses tests passed!")


def main():
    print("=" * 50)
    print("ReadScore Test Suite")
    print("=" * 50)

    try:
        test_convert_to_serializable()
        test_normalize()
        test_multilingual()
        test_align()
        test_prosody_punct()
        test_punctuation_pauses()

        print("\n" + "=" * 50)
        print("ALL TESTS PASSED")
        print("=" * 50)
        return 0
    except AssertionError as e:
        print(f"\n[FAIL] Assertion error: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
