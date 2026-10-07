"""
retrieval.py
------------
Cosine-similarity based news retrieval using the Vector Space Model.

Workflow:
  1. Load pre-built TF-IDF vectorizer and corpus matrix from disk.
  2. Transform the user's query into the same vector space.
  3. Compute cosine similarities against every document.
  4. Return the top-k documents ranked by similarity.

SPPU IR Syllabus Topics Covered:
  - Vector Space Model (VSM)
  - Cosine Similarity
  - Document Ranking
  - Retrieval Evaluation
"""

import os
import sys
import logging

import numpy as np
import joblib
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS   = os.path.join(BASE_DIR, "models")

# Module-level cache so Flask only loads once
_vectorizer   = None
_corpus_tfidf = None
_corpus_meta  = None
_clf          = None


def _load_models():
    """Load all saved models/artefacts into module-level cache."""
    global _vectorizer, _corpus_tfidf, _corpus_meta, _clf

    vpath = os.path.join(MODELS, "tfidf_vectorizer.pkl")
    cpath = os.path.join(MODELS, "corpus_tfidf.npz")
    mpath = os.path.join(MODELS, "corpus_metadata.pkl")
    clf_path = os.path.join(MODELS, "fake_news_model.pkl")

    if not all(os.path.exists(p) for p in [vpath, cpath, mpath, clf_path]):
        raise FileNotFoundError(
            "Model files not found. Please run 'python src/train_model.py' first."
        )

    _vectorizer   = joblib.load(vpath)
    _corpus_tfidf = load_npz(cpath)
    _corpus_meta  = joblib.load(mpath)
    _clf          = joblib.load(clf_path)
    log.info("Models loaded. Corpus size: %d documents.", len(_corpus_meta))


def get_models():
    """Return loaded models, loading them if not yet cached."""
    if _vectorizer is None:
        _load_models()
    return _vectorizer, _corpus_tfidf, _corpus_meta, _clf


def models_ready() -> bool:
    """Check whether model files exist on disk."""
    required = [
        os.path.join(MODELS, "tfidf_vectorizer.pkl"),
        os.path.join(MODELS, "corpus_tfidf.npz"),
        os.path.join(MODELS, "corpus_metadata.pkl"),
        os.path.join(MODELS, "fake_news_model.pkl"),
    ]
    return all(os.path.exists(p) for p in required)


def predict_article(headline: str, body: str) -> dict:
    """
    Preprocess the user's input, classify it, and return:
      - label     : 'FAKE' or 'REAL'
      - confidence: float 0-1 (probability for the predicted class)
      - processed : the cleaned text string
    """
    vectorizer, corpus_tfidf, corpus_meta, clf = get_models()

    combined   = headline + " " + body
    processed  = preprocess_text(combined)

    if not processed.strip():
        return {"error": "Input text too short or contained no meaningful words after preprocessing."}

    vec = vectorizer.transform([processed])

    # predict
    label_id   = int(clf.predict(vec)[0])
    label_name = "FAKE" if label_id == 1 else "REAL"

    # confidence
    try:
        proba      = clf.predict_proba(vec)[0]
        confidence = float(max(proba))
        fake_prob  = float(proba[1]) if len(proba) > 1 else (confidence if label_name == "FAKE" else 1 - confidence)
        real_prob  = float(proba[0]) if len(proba) > 1 else (1 - fake_prob)
    except AttributeError:
        confidence = 1.0
        fake_prob  = 1.0 if label_name == "FAKE" else 0.0
        real_prob  = 1.0 - fake_prob

    return {
        "label":      label_name,
        "label_id":   label_id,
        "confidence": round(confidence * 100, 2),
        "fake_prob":  round(fake_prob * 100, 2),
        "real_prob":  round(real_prob * 100, 2),
        "processed":  processed,
        "vec":        vec,          # sparse matrix, used internally for retrieval
    }


def retrieve_similar(query_vec, top_k: int = 5, exclude_exact: bool = True) -> list:
    """
    Compute cosine similarity between query_vec and the full corpus.
    Returns a ranked list of the top-k most similar documents.

    Parameters
    ----------
    query_vec : sparse matrix (1, n_features)
    top_k     : number of results to return
    exclude_exact : whether to suppress perfect-score matches
                    (avoids returning the input itself if it's in the corpus)

    Returns
    -------
    list of dicts with keys: rank, title, text_snippet, label, similarity
    """
    vectorizer, corpus_tfidf, corpus_meta, clf = get_models()

    sims = cosine_similarity(query_vec, corpus_tfidf).flatten()

    # Sort descending
    ranked_indices = np.argsort(sims)[::-1]

    results = []
    for idx in ranked_indices:
        score = float(sims[idx])
        if exclude_exact and score >= 0.9999:
            continue                          # skip near-identical docs
        if score < 0.001:
            break                             # stop when similarity is negligible

        meta = corpus_meta[idx]
        text = meta.get("text", "")
        snippet = text[:280] + "…" if len(text) > 280 else text

        results.append({
            "rank":       len(results) + 1,
            "title":      meta.get("title", "Untitled"),
            "snippet":    snippet,
            "label":      meta.get("label_name", "UNKNOWN"),
            "similarity": round(score * 100, 2),
        })

        if len(results) >= top_k:
            break

    return results


def analyze(headline: str, body: str, top_k: int = 5) -> dict:
    """
    Combined entry point used by Flask routes.

    Returns a dict with:
      - prediction  : dict from predict_article()
      - similar     : list from retrieve_similar()
    """
    prediction = predict_article(headline, body)
    if "error" in prediction:
        return {"error": prediction["error"]}

    query_vec = prediction.pop("vec")           # remove sparse matrix from JSON-safe dict
    similar   = retrieve_similar(query_vec, top_k=top_k)

    return {
        "prediction": prediction,
        "similar":    similar,
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not models_ready():
        print("Models not found. Run 'python src/train_model.py' first.")
        raise SystemExit(1)

    headline = "Scientists Discover Breakthrough Treatment for Alzheimer's Disease"
    body     = (
        "Researchers at a leading university have published promising results for "
        "a new drug therapy that significantly reduces amyloid plaques in the brain, "
        "a key hallmark of Alzheimer's disease. The phase-2 clinical trial showed "
        "a 35 per cent reduction in cognitive decline over 18 months."
    )

    result = analyze(headline, body)
    print("\n=== Prediction ===")
    pred = result["prediction"]
    print(f"  Label      : {pred['label']}")
    print(f"  Confidence : {pred['confidence']}%")
    print(f"  Fake prob  : {pred['fake_prob']}%")
    print(f"  Real prob  : {pred['real_prob']}%")

    print("\n=== Similar Articles ===")
    for art in result["similar"]:
        print(f"  #{art['rank']}  [{art['label']}]  {art['similarity']}%  {art['title'][:60]}")
