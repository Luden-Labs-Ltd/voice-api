"""
Tests for text normalization module.
"""

import pytest
from readscore.normalize import (
    normalize_text,
    tokenize,
    normalize_word,
    number_to_words
)


class TestNumberToWords:
    def test_single_digits(self):
        assert number_to_words(0) == "zero"
        assert number_to_words(5) == "five"
        assert number_to_words(9) == "nine"

    def test_teens(self):
        assert number_to_words(11) == "eleven"
        assert number_to_words(15) == "fifteen"
        assert number_to_words(19) == "nineteen"

    def test_tens(self):
        assert number_to_words(20) == "twenty"
        assert number_to_words(50) == "fifty"
        assert number_to_words(90) == "ninety"

    def test_two_digits(self):
        assert number_to_words(21) == "twenty one"
        assert number_to_words(42) == "forty two"
        assert number_to_words(99) == "ninety nine"

    def test_hundreds(self):
        assert number_to_words(100) == "one hundred"
        assert number_to_words(123) == "one hundred twenty three"
        assert number_to_words(500) == "five hundred"
        assert number_to_words(999) == "nine hundred ninety nine"

    def test_negative(self):
        assert number_to_words(-5) == "negative five"
        assert number_to_words(-42) == "negative forty two"


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello World", convert_numbers=False) == "hello world"

    def test_punctuation_removal(self):
        result = normalize_text("Hello, world! How are you?", convert_numbers=False)
        assert result == "hello world how are you"

    def test_number_conversion(self):
        result = normalize_text("I have 3 apples", convert_numbers=True)
        assert result == "i have three apples"

    def test_number_conversion_complex(self):
        result = normalize_text("The year is 2024", convert_numbers=True)
        assert "two thousand" in result or "twenty" in result

    def test_whitespace_normalization(self):
        result = normalize_text("hello   world\n\tfoo", convert_numbers=False)
        assert result == "hello world foo"

    def test_apostrophes_preserved(self):
        result = normalize_text("don't won't can't", convert_numbers=False)
        assert "don't" in result or "dont" in result

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_only_punctuation(self):
        assert normalize_text("...!!!???") == ""


class TestTokenize:
    def test_basic_tokenization(self):
        words = tokenize("Hello world")
        assert words == ["hello", "world"]

    def test_with_punctuation(self):
        words = tokenize("Hello, world! How are you?")
        assert words == ["hello", "world", "how", "are", "you"]

    def test_with_numbers(self):
        words = tokenize("I have 5 apples")
        assert "five" in words
        assert len(words) == 4

    def test_empty_string(self):
        assert tokenize("") == []

    def test_whitespace_only(self):
        assert tokenize("   \n\t  ") == []

    def test_multiple_spaces(self):
        words = tokenize("hello    world")
        assert words == ["hello", "world"]


class TestNormalizeWord:
    def test_lowercase(self):
        assert normalize_word("HELLO") == "hello"
        assert normalize_word("HeLLo") == "hello"

    def test_strip_whitespace(self):
        assert normalize_word("  hello  ") == "hello"

    def test_strip_punctuation(self):
        assert normalize_word("hello,") == "hello"
        assert normalize_word("'hello'") == "hello"
        assert normalize_word("...world...") == "world"

    def test_empty_string(self):
        assert normalize_word("") == ""

    def test_only_punctuation(self):
        assert normalize_word("...") == ""
