"""
preprocessing.py
----------------
Text preprocessing pipeline for Fake News Detection system.
Covers: cleaning, tokenization, stop-word removal, stemming/lemmatization.

SPPU IR Syllabus Topics Covered:
  - Tokenization
  - Stop-word removal
  - Stemming / Lemmatization
  - Document preprocessing
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Ensure required NLTK data packages are present
# ---------------------------------------------------------------------------
_NLTK_PACKAGES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]

def ensure_nltk_data():
    """Download NLTK resources if they are not already present."""
    for resource_path, package_id in _NLTK_PACKAGES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_id, quiet=True)

ensure_nltk_data()

# ---------------------------------------------------------------------------
# Initialise tools
# ---------------------------------------------------------------------------
_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Individual preprocessing steps
# ---------------------------------------------------------------------------

def remove_urls(text: str) -> str:
    """Remove HTTP/HTTPS URLs from text."""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_html(text: str) -> str:
    """Strip HTML tags."""
    return re.sub(r"<[^>]+>", " ", text)


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters."""
    return text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple spaces / newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def remove_numbers(text: str) -> str:
    """Remove standalone numeric tokens (keep alpha-numeric words)."""
    return re.sub(r"\b\d+\b", " ", text)


def tokenize(text: str) -> list:
    """Tokenize text into a list of word tokens (lowercased)."""
    return word_tokenize(text.lower())


def remove_stopwords(tokens: list) -> list:
    """Remove English stop words from a list of tokens."""
    return [t for t in tokens if t not in _stop_words and len(t) > 1]


def stem_tokens(tokens: list) -> list:
    """Apply Porter stemming to each token."""
    return [_stemmer.stem(t) for t in tokens]


def contextual_normalization(text: str) -> str:
    """
    Contextual semantic normalization for civic and governance entities.
    Maps collective object pronoun 'us' to 'citizens' in claims of control, surveillance, or harm.
    """
    text = re.sub(r'\b(control|controls|controlling|monitor|monitors|monitoring|track|tracks|tracking|watch|watches|watching|poison|poisons|poisoning|harm|harms|harming|kill|kills|killing|manipulate|manipulates|manipulating|spy|spies|spying)\s+on\s+us\b', r'\1 citizens', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(control|controls|controlling|monitor|monitors|monitoring|track|tracks|tracking|watch|watches|watching|poison|poisons|poisoning|harm|harms|harming|kill|kills|killing|manipulate|manipulates|manipulating|target|targets|targeting)\s+us\b', r'\1 citizens', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(against|over|ruling|enslaving|governing|attacking|subjugating)\s+us\b', r'\1 citizens', text, flags=re.IGNORECASE)
    text = re.sub(r'\bus\b', 'citizens', text, flags=re.IGNORECASE)
    return text


def lemmatize_tokens(tokens: list) -> list:
    """Apply WordNet lemmatization to each token, guarding against 'us' -> 'u'."""
    out = []
    for t in tokens:
        if t in ("us", "u"):
            out.append("citizen")
        else:
            lemma = _lemmatizer.lemmatize(t)
            if lemma == "u" and t == "us":
                out.append("citizen")
            else:
                out.append(lemma)
    return out


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_text(text: str, use_stemming: bool = False) -> str:
    """
    Full preprocessing pipeline.

    Steps:
        1. Contextual pronoun normalization ('us' -> 'citizens')
        2. Lowercase
        3. Remove URLs
        4. Remove HTML tags
        5. Remove punctuation
        6. Remove standalone numbers
        7. Tokenize
        8. Remove stop words
        9. Lemmatize (or stem if use_stemming=True)
        10. Rejoin into a single string
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # Step 1: contextual normalization
    text = contextual_normalization(text)

    # Step 2-6: cleaning
    text = text.lower()
    text = remove_urls(text)
    text = remove_html(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_extra_whitespace(text)

    # Step 7: tokenize
    tokens = tokenize(text)

    # Step 8: stop-word removal
    tokens = remove_stopwords(tokens)

    # Step 9: normalisation
    if use_stemming:
        tokens = stem_tokens(tokens)
    else:
        tokens = lemmatize_tokens(tokens)

    # Step 10: rejoin
    return " ".join(tokens)


def preprocess_pipeline_info() -> dict:
    """Return a human-readable description of the preprocessing pipeline."""
    return {
        "steps": [
            "Contextual semantic normalization ('us' -> 'citizens')",
            "Lowercase conversion",
            "URL removal",
            "HTML tag removal",
            "Punctuation removal",
            "Number removal",
            "Tokenization (NLTK word_tokenize)",
            "Stop-word removal (NLTK English stopwords)",
            "Lemmatization (WordNet) / Stemming (Porter)",
            "Rejoining tokens into clean text string",
        ],
        "stop_words_count": len(_stop_words),
        "stemmer": "PorterStemmer",
        "lemmatizer": "WordNetLemmatizer",
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "BREAKING: Politicians say the economy is booming!!! "
        "Click here http://fake.com/news to read more. "
        "This article was published in 2024."
    )
    print("Original :", sample)
    print("Processed:", preprocess_text(sample))
    print("\nPipeline info:", preprocess_pipeline_info())
