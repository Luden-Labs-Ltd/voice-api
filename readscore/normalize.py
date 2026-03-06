"""
Text normalization utilities for alignment.
Supports English (en), Russian (ru), and Hebrew (he).
"""

import re
import unicodedata
from typing import List, Optional
from dataclasses import dataclass


# Common number words for basic number-to-word conversion (English)
NUMBER_WORDS_EN = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
    14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
    18: "eighteen", 19: "nineteen", 20: "twenty", 30: "thirty",
    40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
    80: "eighty", 90: "ninety"
}

# Russian number words (basic)
NUMBER_WORDS_RU = {
    0: "ноль", 1: "один", 2: "два", 3: "три", 4: "четыре",
    5: "пять", 6: "шесть", 7: "семь", 8: "восемь", 9: "девять",
    10: "десять"
}

# Hebrew number words (basic)
NUMBER_WORDS_HE = {
    0: "אפס", 1: "אחת", 2: "שתיים", 3: "שלוש", 4: "ארבע",
    5: "חמש", 6: "שש", 7: "שבע", 8: "שמונה", 9: "תשע",
    10: "עשר"
}


@dataclass
class NormalizationConfig:
    """Language-specific normalization configuration."""
    lang: str = "en"
    convert_numbers: bool = True
    ru_normalize_yo: bool = True  # Convert ё to е
    he_strip_niqqud: bool = True  # Strip Hebrew vowel points


def number_to_words(n: int, lang: str = "en") -> str:
    """Convert integer to words (basic support)."""
    number_words = {
        "en": NUMBER_WORDS_EN,
        "ru": NUMBER_WORDS_RU,
        "he": NUMBER_WORDS_HE,
    }.get(lang, NUMBER_WORDS_EN)

    if n < 0:
        prefix = {"en": "negative ", "ru": "минус ", "he": "מינוס "}.get(lang, "negative ")
        return prefix + number_to_words(-n, lang)

    if n in number_words:
        return number_words[n]

    # For English, handle tens
    if lang == "en" and n < 100:
        tens, ones = divmod(n, 10)
        if tens * 10 in number_words:
            return number_words[tens * 10] + (" " + number_words[ones] if ones else "")

    if lang == "en" and n < 1000:
        hundreds, remainder = divmod(n, 100)
        result = number_words[hundreds] + " hundred"
        if remainder:
            result += " " + number_to_words(remainder, lang)
        return result

    # For other languages or larger numbers, return digits as words
    return " ".join(number_words.get(int(d), str(d)) for d in str(n))


def strip_hebrew_niqqud(text: str) -> str:
    """
    Strip Hebrew niqqud (vowel points) from text.
    Niqqud Unicode range: U+0591 to U+05C7
    """
    # Remove Hebrew points and marks
    return re.sub(r'[\u0591-\u05C7]', '', text)


def normalize_russian(text: str, normalize_yo: bool = True) -> str:
    """Apply Russian-specific normalization."""
    if normalize_yo:
        # Replace ё with е (common normalization)
        text = text.replace('ё', 'е').replace('Ё', 'Е')
    return text


def detect_script(text: str) -> str:
    """
    Detect the primary script of text.
    Returns: 'latin', 'cyrillic', 'hebrew', or 'unknown'
    """
    # Count characters by script
    latin = 0
    cyrillic = 0
    hebrew = 0

    for char in text:
        if char.isalpha():
            # Check Unicode script
            name = unicodedata.name(char, '')
            if 'LATIN' in name:
                latin += 1
            elif 'CYRILLIC' in name:
                cyrillic += 1
            elif 'HEBREW' in name:
                hebrew += 1

    total = latin + cyrillic + hebrew
    if total == 0:
        return 'unknown'

    if hebrew > latin and hebrew > cyrillic:
        return 'hebrew'
    if cyrillic > latin and cyrillic > hebrew:
        return 'cyrillic'
    return 'latin'


def detect_language(text: str) -> str:
    """
    Detect language from text based on script.
    Returns: 'en', 'ru', 'he', or 'en' as default
    """
    script = detect_script(text)
    if script == 'hebrew':
        return 'he'
    if script == 'cyrillic':
        return 'ru'
    return 'en'


def normalize_text(
    text: str,
    lang: str = "en",
    convert_numbers: bool = True,
    config: Optional[NormalizationConfig] = None
) -> str:
    """
    Normalize text for alignment.
    Supports English, Russian, and Hebrew.

    Args:
        text: Input text
        lang: Language code ('en', 'ru', 'he', or 'auto')
        convert_numbers: Whether to convert numbers to words
        config: Optional normalization config

    Returns:
        Normalized text
    """
    if config is None:
        config = NormalizationConfig(lang=lang, convert_numbers=convert_numbers)

    # Auto-detect language if needed
    if lang == "auto":
        lang = detect_language(text)

    # Normalize unicode (NFKC for compatibility)
    text = unicodedata.normalize("NFKC", text)

    # Language-specific pre-processing
    if lang == "ru":
        text = normalize_russian(text, config.ru_normalize_yo)
    elif lang == "he" and config.he_strip_niqqud:
        text = strip_hebrew_niqqud(text)

    # Lowercase (works for all Unicode scripts)
    text = text.lower()

    # Convert numbers to words if requested
    if convert_numbers:
        def replace_number(match):
            num = int(match.group())
            if num < 10000:  # Only convert reasonable numbers
                return number_to_words(num, lang)
            return match.group()
        text = re.sub(r'\b\d+\b', replace_number, text)

    # Remove punctuation but keep letters (Unicode-aware) and spaces
    # This regex keeps: Unicode letters, numbers, spaces, apostrophes, hyphens
    # \w in Python 3 with re.UNICODE is Unicode-aware
    text = re.sub(r"[^\w\s'\-]", " ", text, flags=re.UNICODE)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize(text: str, lang: str = "en") -> List[str]:
    """
    Tokenize text into words.
    Unicode-safe, works with Latin, Cyrillic, and Hebrew scripts.

    Args:
        text: Input text
        lang: Language code

    Returns:
        List of normalized word tokens
    """
    # Auto-detect if needed
    if lang == "auto":
        lang = detect_language(text)

    normalized = normalize_text(text, lang=lang)

    # Split on whitespace
    words = normalized.split()

    # Filter out empty strings and standalone punctuation
    return [w for w in words if w and w not in ("'", "-", "'")]


def normalize_word(word: str, lang: str = "en") -> str:
    """
    Normalize a single word for comparison.
    Unicode-safe.

    Args:
        word: Input word
        lang: Language code

    Returns:
        Normalized word
    """
    if lang == "auto":
        lang = detect_language(word)

    word = word.lower().strip()

    # Language-specific normalization
    if lang == "ru":
        word = normalize_russian(word)
    elif lang == "he":
        word = strip_hebrew_niqqud(word)

    # Remove leading/trailing non-letter characters (Unicode-aware)
    # Match any non-letter, non-number at start or end
    word = re.sub(r'^[^\w]+|[^\w]+$', '', word, flags=re.UNICODE)

    return word


# Backward compatibility aliases
NUMBER_WORDS = NUMBER_WORDS_EN
