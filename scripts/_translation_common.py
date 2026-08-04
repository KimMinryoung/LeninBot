"""Helpers shared by the translation pipeline scripts."""


def hangul_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Hangul syllables."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    hangul = sum(1 for ch in letters if "가" <= ch <= "힣")
    return hangul / len(letters)
