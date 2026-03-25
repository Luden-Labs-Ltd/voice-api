"""
Text normalization utilities for alignment.
Hebrew is the primary language for this system.
Supports English (en), Russian (ru), and Hebrew (he).
"""

import re
import unicodedata
from typing import List, Optional
from dataclasses import dataclass, field


# ── Number words ─────────────────────────────────────────────────────────────

NUMBER_WORDS_EN = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
    14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
    18: "eighteen", 19: "nineteen", 20: "twenty", 30: "thirty",
    40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
    80: "eighty", 90: "ninety"
}

NUMBER_WORDS_RU = {
    0: "ноль", 1: "один", 2: "два", 3: "три", 4: "четыре",
    5: "пять", 6: "шесть", 7: "семь", 8: "восемь", 9: "девять",
    10: "десять"
}

NUMBER_WORDS_HE = {
    0: "אפס", 1: "אחת", 2: "שתיים", 3: "שלוש", 4: "ארבע",
    5: "חמש", 6: "שש", 7: "שבע", 8: "שמונה", 9: "תשע",
    10: "עשר"
}

# ── Hebrew character tables ───────────────────────────────────────────────────

# Five Hebrew letters have distinct end-of-word (final) forms.
# Normalizing them prevents false mismatches when ASR outputs the wrong form,
# e.g. "ילדים" read as "ילדימ" due to final-mem confusion.
HEBREW_FINAL_FORMS = {
    '\u05DA': '\u05DB',  # ך (final kaf)    → כ (kaf)
    '\u05DD': '\u05DE',  # ם (final mem)    → מ (mem)
    '\u05DF': '\u05E0',  # ן (final nun)    → נ (nun)
    '\u05E3': '\u05E4',  # ף (final pe)     → פ (pe)
    '\u05E5': '\u05E6',  # ץ (final tsadi)  → צ (tsadi)
}


@dataclass
class NormalizationConfig:
    """Language-specific normalization configuration."""
    lang: str = "en"
    convert_numbers: bool = True
    ru_normalize_yo: bool = True       # Convert ё→е (Russian)
    he_strip_niqqud: bool = True       # Strip Hebrew vowel points
    he_normalize_finals: bool = True   # Normalize Hebrew final letter forms
    he_strip_special: bool = True      # Strip geresh, gershayim, maqaf


# ── Hebrew normalization helpers ──────────────────────────────────────────────

def strip_hebrew_niqqud(text: str) -> str:
    """
    Strip Hebrew niqqud (vowel diacritics) from text.
    Unicode range U+0591–U+05C7 covers all niqqud and cantillation marks.
    """
    return re.sub(r'[\u0591-\u05C7]', '', text)


def normalize_hebrew_finals(text: str) -> str:
    """
    Normalize Hebrew final letter forms to their base (medial) forms.
    Prevents false mismatches: ך↔כ, ם↔מ, ן↔נ, ף↔פ, ץ↔צ.
    Applied AFTER niqqud stripping so the mapping is on clean characters.
    """
    for final, base in HEBREW_FINAL_FORMS.items():
        text = text.replace(final, base)
    return text


def strip_hebrew_punctuation(text: str) -> str:
    """
    Strip/replace Hebrew-specific punctuation marks:
    - Maqaf ־ (U+05BE): Hebrew word-joining hyphen; replace with space.
    - Geresh ׳ (U+05F3): abbreviation mark; remove.
    - Gershayim ״ (U+05F4): acronym mark; remove.
    """
    text = text.replace('\u05BE', ' ')            # maqaf → space (word boundary)
    text = re.sub(r'[\u05F3\u05F4]', '', text)   # geresh / gershayim → remove
    return text


# ── Russian normalization ─────────────────────────────────────────────────────

def normalize_russian(text: str, normalize_yo: bool = True) -> str:
    """Apply Russian-specific normalization."""
    if normalize_yo:
        text = text.replace('ё', 'е').replace('Ё', 'Е')
    return text


# ── Script / language detection ───────────────────────────────────────────────

def detect_script(text: str) -> str:
    """
    Detect the primary script of text.
    Returns: 'hebrew', 'cyrillic', 'latin', or 'unknown'.
    Hebrew is checked first because it is the priority language.
    """
    latin = cyrillic = hebrew = 0
    for char in text:
        if char.isalpha():
            name = unicodedata.name(char, '')
            if 'HEBREW' in name:
                hebrew += 1
            elif 'CYRILLIC' in name:
                cyrillic += 1
            elif 'LATIN' in name:
                latin += 1
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
    Hebrew is detected first as the priority language.
    Returns: 'he', 'ru', or 'en'.
    """
    script = detect_script(text)
    if script == 'hebrew':
        return 'he'
    if script == 'cyrillic':
        return 'ru'
    return 'en'


# ── Number conversion ─────────────────────────────────────────────────────────

def number_to_words(n: int, lang: str = "en") -> str:
    """Convert integer to words (basic support for en, ru, he)."""
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

    # For other languages / larger numbers: spell out each digit
    return " ".join(number_words.get(int(d), str(d)) for d in str(n))


# ── Main normalization API ────────────────────────────────────────────────────

def normalize_text(
    text: str,
    lang: str = "en",
    convert_numbers: bool = True,
    config: Optional[NormalizationConfig] = None
) -> str:
    """
    Normalize text for alignment. Supports English, Russian, and Hebrew.

    Hebrew pipeline (in order):
      1. NFKC unicode normalization
      2. Strip niqqud (vowel diacritics)
      3. Normalize final letter forms (ך→כ etc.)
      4. Strip maqaf / geresh / gershayim
      5. Lowercase
      6. Number → words conversion
      7. Strip non-letter punctuation
      8. Collapse whitespace
    """
    if config is None:
        config = NormalizationConfig(lang=lang, convert_numbers=convert_numbers)

    if lang == "auto":
        lang = detect_language(text)

    # Step 1: Unicode compatibility normalization
    text = unicodedata.normalize("NFKC", text)

    # Step 2-4: Language-specific pre-processing
    if lang == "ru":
        text = normalize_russian(text, config.ru_normalize_yo)
    elif lang == "he":
        if config.he_strip_niqqud:
            text = strip_hebrew_niqqud(text)
        if config.he_normalize_finals:
            text = normalize_hebrew_finals(text)
        if config.he_strip_special:
            text = strip_hebrew_punctuation(text)

    # Step 5: Lowercase (Unicode-safe)
    text = text.lower()

    # Step 6: Convert numbers to words
    if convert_numbers:
        def replace_number(match):
            num = int(match.group())
            if num < 10000:
                return number_to_words(num, lang)
            return match.group()
        text = re.sub(r'\b\d+\b', replace_number, text)

    # Step 7: Remove punctuation; keep Unicode letters, digits, spaces, apostrophes, hyphens
    text = re.sub(r"[^\w\s'\-]", " ", text, flags=re.UNICODE)

    # Step 8: Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize(text: str, lang: str = "en") -> List[str]:
    """
    Tokenize text into normalized word tokens.
    Unicode-safe; works with Latin, Cyrillic, and Hebrew scripts.
    """
    if lang == "auto":
        lang = detect_language(text)

    normalized = normalize_text(text, lang=lang)
    words = normalized.split()
    return [w for w in words if w and w not in ("'", "-", "\u2019")]


def normalize_word(word: str, lang: str = "en") -> str:
    """
    Normalize a single word for comparison. Unicode-safe.
    Applies the full Hebrew normalization pipeline for Hebrew words.
    """
    if lang == "auto":
        lang = detect_language(word)

    word = word.lower().strip()

    if lang == "ru":
        word = normalize_russian(word)
    elif lang == "he":
        word = strip_hebrew_niqqud(word)
        word = normalize_hebrew_finals(word)
        word = strip_hebrew_punctuation(word).strip()

    # Strip leading/trailing non-letter characters (Unicode-aware)
    word = re.sub(r'^[^\w]+|[^\w]+$', '', word, flags=re.UNICODE)

    return word


# Backward compatibility alias
NUMBER_WORDS = NUMBER_WORDS_EN
