"""
Tests for word alignment module.
"""

import pytest
from readscore.align import (
    levenshtein_align,
    AlignTag,
    AlignedWord,
    AlignmentResult
)


class TestLevenshteinAlign:
    def test_perfect_match(self):
        ref = ["hello", "world"]
        hyp = ["hello", "world"]
        result = levenshtein_align(ref, hyp)

        assert result.wer == 0.0
        assert result.correct == 2
        assert result.insertions == 0
        assert result.deletions == 0
        assert result.substitutions == 0
        assert len(result.alignment) == 2
        assert all(a.tag == AlignTag.OK for a in result.alignment)

    def test_single_substitution(self):
        ref = ["hello", "world"]
        hyp = ["hello", "word"]
        result = levenshtein_align(ref, hyp)

        assert result.wer == 0.5
        assert result.correct == 1
        assert result.substitutions == 1
        assert result.insertions == 0
        assert result.deletions == 0

    def test_single_insertion(self):
        ref = ["hello", "world"]
        hyp = ["hello", "big", "world"]
        result = levenshtein_align(ref, hyp)

        assert result.insertions == 1
        assert result.correct == 2
        assert result.wer == 0.5  # 1 error / 2 ref words

    def test_single_deletion(self):
        ref = ["hello", "big", "world"]
        hyp = ["hello", "world"]
        result = levenshtein_align(ref, hyp)

        assert result.deletions == 1
        assert result.correct == 2
        assert result.wer == pytest.approx(1/3, rel=0.01)

    def test_empty_hypothesis(self):
        ref = ["hello", "world"]
        hyp = []
        result = levenshtein_align(ref, hyp)

        assert result.deletions == 2
        assert result.correct == 0
        assert result.wer == 1.0

    def test_empty_reference(self):
        ref = []
        hyp = ["hello", "world"]
        result = levenshtein_align(ref, hyp)

        assert result.insertions == 2
        assert result.correct == 0
        # WER with empty reference is 1.0 if hyp is non-empty
        assert result.wer == 1.0

    def test_both_empty(self):
        result = levenshtein_align([], [])
        assert result.wer == 0.0
        assert len(result.alignment) == 0

    def test_case_insensitive(self):
        ref = ["Hello", "World"]
        hyp = ["hello", "world"]
        result = levenshtein_align(ref, hyp)

        assert result.wer == 0.0
        assert result.correct == 2

    def test_with_timestamps(self):
        ref = ["hello", "world"]
        hyp = ["hello", "world"]
        timestamps = [(0.0, 0.5), (0.6, 1.0)]
        result = levenshtein_align(ref, hyp, timestamps)

        assert result.alignment[0].t0 == 0.0
        assert result.alignment[0].t1 == 0.5
        assert result.alignment[1].t0 == 0.6
        assert result.alignment[1].t1 == 1.0

    def test_with_confidences(self):
        ref = ["hello", "world"]
        hyp = ["hello", "world"]
        timestamps = [(0.0, 0.5), (0.6, 1.0)]
        confidences = [0.95, 0.88]
        result = levenshtein_align(ref, hyp, timestamps, confidences)

        assert result.alignment[0].conf == 0.95
        assert result.alignment[1].conf == 0.88

    def test_complex_alignment(self):
        # Reference: "the quick brown fox"
        # Hypothesis: "the slow brown dog jumped"
        # Expected: the(ok), quick->slow(sub), brown(ok), fox->dog(sub), jumped(ins)
        ref = ["the", "quick", "brown", "fox"]
        hyp = ["the", "slow", "brown", "dog", "jumped"]
        result = levenshtein_align(ref, hyp)

        assert result.correct == 2  # "the" and "brown"
        assert result.substitutions == 2  # quick->slow, fox->dog
        assert result.insertions == 1  # "jumped"
        assert result.deletions == 0
        assert result.wer == pytest.approx(3/4, rel=0.01)  # 3 errors / 4 ref words

    def test_alignment_order(self):
        ref = ["a", "b", "c"]
        hyp = ["a", "b", "c"]
        result = levenshtein_align(ref, hyp)

        assert result.alignment[0].ref == "a"
        assert result.alignment[1].ref == "b"
        assert result.alignment[2].ref == "c"


class TestAlignmentResult:
    def test_to_dict(self):
        ref = ["hello", "world"]
        hyp = ["hello", "word"]
        timestamps = [(0.0, 0.5), (0.6, 1.0)]
        confidences = [0.9, 0.8]
        result = levenshtein_align(ref, hyp, timestamps, confidences)

        d = result.to_dict()
        assert "wer" in d
        assert "counts" in d
        assert "alignment" in d
        assert d["counts"]["sub"] == 1
        assert len(d["alignment"]) == 2


class TestAlignedWord:
    def test_to_dict(self):
        word = AlignedWord(
            ref="hello",
            hyp="hello",
            tag=AlignTag.OK,
            t0=0.0,
            t1=0.5,
            conf=0.95
        )
        d = word.to_dict()

        assert d["ref"] == "hello"
        assert d["hyp"] == "hello"
        assert d["tag"] == "ok"
        assert d["t0"] == 0.0
        assert d["t1"] == 0.5
        assert d["conf"] == 0.95

    def test_to_dict_with_none(self):
        word = AlignedWord(
            ref="hello",
            hyp=None,
            tag=AlignTag.DEL,
            t0=None,
            t1=None,
            conf=None
        )
        d = word.to_dict()

        assert d["ref"] == "hello"
        assert d["hyp"] is None
        assert d["tag"] == "del"
