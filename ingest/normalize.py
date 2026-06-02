"""Text normalization for Arabic + English book content.

Pure-python (stdlib only). Normalizes Unicode, tidies whitespace, and applies
Arabic-specific orthographic normalization so that variant spellings collapse
to a canonical form before chunking/embedding. Language detection is delegated
to ``langdetect`` (imported lazily and guarded), classifying text as
``"ar"`` | ``"en"`` | ``"mixed"`` | ``"unknown"``.
"""

from __future__ import annotations

import re
import unicodedata

from config import settings
from core.logging import get_logger

logger = get_logger(__name__)

# --- Arabic Unicode reference points -----------------------------------------

# Tatweel / kashida elongation character (purely cosmetic, carries no meaning).
_TATWEEL = "ـ"

# Harakat (short-vowel diacritics) and tanwin: fathatan..sukun (U+064B..U+0652),
# plus superscript alef (U+0670) which behaves like a diacritic in vowelled text.
_HARAKAT = (
    "ًٌٍ"          # tanwin: fathatan, dammatan, kasratan
    "َُِ"          # fatha, damma, kasra
    "ّْ"                # shadda, sukun
    "ٕٓٔ"          # maddah above, hamza above, hamza below
    "ٖٗ٘ٙ"    # subscript alef + extended Quranic marks
    "ٜٟٚٛٝٞ"
    "ٰ"                      # superscript (dagger) alef
)
_HARAKAT_RE = re.compile("[" + _HARAKAT + "]")

# Alef variants that normalize to a bare alef (U+0627):
#   أ U+0623 (hamza above), إ U+0625 (hamza below), آ U+0622 (maddah),
#   ٱ U+0671 (wasla), ٲ/ٳ extended forms.
_ALEF_VARIANTS = "أإآٱٲٳ"
_ALEF_TRANS = {ord(ch): "ا" for ch in _ALEF_VARIANTS}

# ya / alef-maqsura normalization: ى (U+0649) -> ي (U+064A). Configurable because
# Egyptian convention writes final ya as alef-maqsura; many indices prefer ي.
_ALEF_MAQSURA = "ى"
_YA = "ي"

# Detached hamza-on-the-line + hamza carriers we leave mostly intact, except the
# ya-with-hamza (ئ U+0626) -> ي and waw-with-hamza (ؤ U+0624) -> و are common
# "search friendly" reductions. teh marbuta (ة U+0629) is intentionally KEPT.
_YA_HAMZA = "ئ"
_WAW_HAMZA = "ؤ"
_WAW = "و"

# Arabic Presentation Forms / common substitutes that NFC alone won't fold:
#   ﻻ ligatures are handled by NFKC-style decomposition we do manually below.
_LAM_ALEF_LIGATURES = {
    "ﻻ": "لا",  # LAM WITH ALEF
    "ﻷ": "لأ",  # LAM WITH ALEF HAMZA ABOVE
    "ﻵ": "لآ",  # LAM WITH ALEF MADDA ABOVE
    "ﻹ": "لإ",  # LAM WITH ALEF HAMZA BELOW
}

# Arabic-Indic digits (U+0660..U+0669) and Extended (U+06F0..U+06F9). We keep
# them as-is by default (faithful to source); only whitespace/diacritics change.

# Range used for the Arabic-character ratio in language detection: Arabic block
# (U+0600..U+06FF), Supplement, Presentation Forms A/B.
_ARABIC_CHAR_RE = re.compile(
    "[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]"
)
_LATIN_CHAR_RE = re.compile("[A-Za-z]")

# Whitespace collapsing: any run of unicode whitespace -> single space, but keep
# paragraph breaks (blank lines) meaningful for downstream chunking.
_INLINE_WS_RE = re.compile(r"[^\S\n]+")          # spaces/tabs but not newlines
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")        # >2 blank lines -> 2
_TRAILING_WS_RE = re.compile(r"[^\S\n]+\n")      # trailing inline ws before \n

# Zero-width / bidi control marks that pollute extracted PDF text.
_ZERO_WIDTH_RE = re.compile(
    "[​‌‍‎‏‪-‮⁦-⁩﻿]"
)


def _strip_zero_width(text: str) -> str:
    """Remove zero-width and bidirectional control characters."""
    return _ZERO_WIDTH_RE.sub("", text)


def _collapse_whitespace(text: str) -> str:
    """Collapse runs of inline whitespace and excess blank lines.

    Inline spaces/tabs become a single space; runs of 3+ newlines become a
    paragraph break (two newlines). Leading/trailing whitespace is stripped.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INLINE_WS_RE.sub(" ", text)
    text = _TRAILING_WS_RE.sub("\n", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def normalize_arabic(
    text: str,
    *,
    strip_diacritics: bool = settings.strip_diacritics,
    strip_tatweel: bool = settings.strip_tatweel,
    normalize_alef_maqsura: bool = True,
    normalize_hamza_carriers: bool = True,
) -> str:
    """Apply Arabic orthographic normalization to ``text``.

    Steps (in order):
      1. Unicode NFC normalization.
      2. Expand lam-alef presentation ligatures to their two-character form.
      3. Map alef variants (أ إ آ ٱ ...) -> bare alef ا.
      4. Optionally map waw-hamza (ؤ) -> و and ya-hamza (ئ) -> ي
         (``normalize_hamza_carriers``).
      5. Optionally map alef-maqsura (ى) -> ya (ي) (``normalize_alef_maqsura``).
      6. teh marbuta (ة) is preserved (NOT folded to ه).
      7. Remove tatweel ـ when ``strip_tatweel``.
      8. Remove harakat / tanwin / superscript-alef when ``strip_diacritics``.

    Whitespace is intentionally left to :func:`normalize_text`; this function
    focuses on character-level Arabic folding so it can be reused standalone.

    Args:
        text: Input string (any script; non-Arabic chars pass through).
        strip_diacritics: Remove short-vowel marks (default from settings).
        strip_tatweel: Remove kashida elongation (default from settings).
        normalize_alef_maqsura: Fold ى -> ي.
        normalize_hamza_carriers: Fold ؤ -> و and ئ -> ي.

    Returns:
        The normalized string.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)

    # Lam-alef ligatures (common in OCR / presentation-form PDFs).
    for ligature, expansion in _LAM_ALEF_LIGATURES.items():
        if ligature in text:
            text = text.replace(ligature, expansion)

    # Alef family -> bare alef.
    text = text.translate(_ALEF_TRANS)

    if normalize_hamza_carriers:
        text = text.replace(_WAW_HAMZA, _WAW).replace(_YA_HAMZA, _YA)

    if normalize_alef_maqsura:
        text = text.replace(_ALEF_MAQSURA, _YA)

    if strip_tatweel:
        text = text.replace(_TATWEEL, "")

    if strip_diacritics:
        text = _HARAKAT_RE.sub("", text)

    return text


def normalize_text(text: str, lang: str | None = None) -> str:
    """Normalize arbitrary text for indexing/display.

    Always tidies whitespace and strips zero-width/bidi control marks. When the
    text contains Arabic characters (or ``lang`` indicates Arabic/mixed), it is
    additionally passed through :func:`normalize_arabic` using the configured
    ``settings.strip_diacritics`` / ``settings.strip_tatweel`` defaults.

    Args:
        text: Raw extracted or OCR'd text.
        lang: Optional language hint ("ar" | "en" | "mixed" | ...). When omitted
            the decision is made by scanning for Arabic characters.

    Returns:
        Cleaned, normalized text.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = _strip_zero_width(text)

    has_arabic = bool(_ARABIC_CHAR_RE.search(text))
    wants_arabic = lang in {"ar", "mixed"} if lang else False

    if has_arabic or wants_arabic:
        text = normalize_arabic(text)

    return _collapse_whitespace(text)


def _arabic_ratio(text: str) -> float:
    """Fraction of letter characters that are Arabic (ignores spaces/digits)."""
    arabic = len(_ARABIC_CHAR_RE.findall(text))
    latin = len(_LATIN_CHAR_RE.findall(text))
    letters = arabic + latin
    if letters == 0:
        return 0.0
    return arabic / letters


def detect_lang(text: str) -> str:
    """Classify ``text`` as ``"ar"``, ``"en"``, ``"mixed"``, or ``"unknown"``.

    Strategy:
      - Empty / letter-free text -> ``"unknown"``.
      - Compute the Arabic-letter ratio. A clearly dominant script
        (>= ~0.85 Arabic -> ``"ar"``; <= ~0.15 -> ``"en"``) is decided directly
        and cheaply, which is robust for the short page fragments we see.
      - For the ambiguous middle band, consult ``langdetect`` (lazy import,
        guarded). If both Arabic and Latin letters are present in meaningful
        proportion, return ``"mixed"``.

    ``langdetect`` failures (missing dependency, empty profile, etc.) never
    raise; they degrade to a ratio-based answer or ``"unknown"``.
    """
    if not text or not text.strip():
        return "unknown"

    cleaned = _strip_zero_width(text)
    arabic = len(_ARABIC_CHAR_RE.findall(cleaned))
    latin = len(_LATIN_CHAR_RE.findall(cleaned))
    letters = arabic + latin
    if letters == 0:
        return "unknown"

    ratio = arabic / letters

    # Confident single-script decisions without paying for langdetect.
    if ratio >= 0.85:
        return "ar"
    if ratio <= 0.15:
        return "en"

    # Ambiguous band: meaningful presence of both scripts -> mixed.
    if arabic >= 3 and latin >= 3:
        return "mixed"

    # Fall back to langdetect for the remaining borderline cases.
    detected = _detect_with_langdetect(cleaned)
    if detected == "ar":
        return "ar"
    if detected == "en":
        return "en"

    # Last resort: lean on the ratio.
    if ratio >= 0.5:
        return "ar"
    return "en"


def _detect_with_langdetect(text: str) -> str:
    """Best-effort language probe via ``langdetect`` (lazy, never raises).

    Returns ``"ar"``, ``"en"``, or ``"unknown"``. Any import or runtime error is
    swallowed and reported as ``"unknown"`` so callers can fall back gracefully.
    """
    try:
        from langdetect import LangDetectException, detect  # type: ignore
    except Exception as exc:  # pragma: no cover - missing dependency
        logger.debug("langdetect unavailable: %s", exc)
        return "unknown"

    try:
        code = detect(text)
    except LangDetectException:
        return "unknown"
    except Exception as exc:  # pragma: no cover - unexpected runtime failure
        logger.debug("langdetect failed: %s", exc)
        return "unknown"

    if code == "ar":
        return "ar"
    if code == "en":
        return "en"
    return "unknown"
