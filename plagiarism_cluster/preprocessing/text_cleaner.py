"""
preprocessing/text_cleaner.py
==============================
Text cleaning, sentence splitting, language detection — 100% local.
"""

import re
import unicodedata
import nltk
from typing import List, Dict

# Download required NLTK data silently
for pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg != "stopwords" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

try:
    from langdetect import detect as _langdetect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False


def normalize_unicode(text: str) -> str:
    """Normalize unicode — handles Arabic, accented chars, etc."""
    return unicodedata.normalize("NFC", text)


def clean_text(text: str, lowercase: bool = True) -> str:
    """Full text cleaning pipeline."""
    text = normalize_unicode(text)
    # Remove control characters (but keep newlines)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if lowercase:
        text = text.lower()
    return text.strip()


def split_sentences(text: str, min_len: int = 20) -> List[str]:
    """Split into sentences, filter very short ones."""
    try:
        sents = nltk.sent_tokenize(text)
    except Exception:
        # Fallback: split on punctuation
        sents = re.split(r"(?<=[.!?؟])\s+", text)
    return [s.strip() for s in sents if len(s.strip()) >= min_len]


def detect_language(text: str) -> str:
    """Detect document language (returns ISO code or 'unknown')."""
    if not HAS_LANGDETECT:
        return "unknown"
    try:
        return _langdetect(text[:2000])
    except Exception:
        return "unknown"


def preprocess_document(text: str) -> Dict:
    """
    Full preprocessing pipeline.
    Returns dict with clean_text, sentences, language, word_count.
    """
    cleaned   = clean_text(text)
    sentences = split_sentences(cleaned)
    language  = detect_language(cleaned)
    return {
        "clean_text":  cleaned,
        "sentences":   sentences,
        "language":    language,
        "word_count":  len(cleaned.split()),
        "char_count":  len(cleaned),
    }
