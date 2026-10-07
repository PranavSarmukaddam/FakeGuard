# FakeGuard: Fake News Detection and Similar News Retrieval

FakeGuard is an end-to-end Information Retrieval project that classifies news as
fake or real and retrieves the five most similar articles from its indexed corpus.
It combines an explainable TF-IDF and cosine-similarity pipeline with three
text-classification baselines in a Flask web application.

**SPPU B.E. Computer Engineering – Information Retrieval (410245(A)) Mini Project**  
Subject: Information Retrieval | Semester VII | Elective IV | 2019 Pattern

---

## 📋 Project Overview

FakeGuard is a web-based **Fake News Detection and Similar News Retrieval System** that:

1. Accepts a news headline and/or article as input
2. **Preprocesses** text: lowercasing → URL removal → tokenization → stop-word removal → lemmatization
3. Converts text to **TF-IDF vectors** (Vector Space Model)
4. **Classifies** the article as **FAKE** or **REAL** using the best of three ML models
5. **Retrieves top-5 similar articles** using Cosine Similarity
6. Displays **confidence scores**, **evaluation metrics** and **confusion matrices**

---

## 🎯 SPPU IR Syllabus Mapping

| Syllabus Topic | Implementation |
|---|---|
| Tokenization | NLTK `word_tokenize` |
| Stop-word removal | NLTK English stopwords |
| Stemming / Lemmatization | Porter Stemmer + WordNet Lemmatizer |
| TF-IDF Weighting | Scikit-learn `TfidfVectorizer` (bigrams) |
| Vector Space Model | TF-IDF matrix for full corpus |
| Cosine Similarity | `sklearn.metrics.pairwise.cosine_similarity` |
| Document Ranking | Top-5 articles ranked by similarity score |
| Classification | Logistic Regression, Naive Bayes, Linear SVM |
| Precision / Recall / F1 | Computed on 20% held-out test set |
| Confusion Matrix | Interactive chart on Evaluation page |
| Probabilistic Model | Multinomial Naive Bayes |

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Backend | Python 3.8+, Flask 2.3 |
| ML / NLP | Scikit-learn, NLTK |
| Data | Pandas, NumPy, SciPy |
| Frontend | HTML5, Vanilla CSS, Vanilla JS |
| Charts | Chart.js (CDN) |
| Model Persistence | Joblib |

---

## 📁 Project Structure

```
fake-news-ir/
├── app.py                      ← Flask web application
├── requirements.txt            ← Python dependencies
├── README.md                   ← This file
│
├── data/
│   ├── raw/                    ← Place Fake.csv + True.csv here
│   └── processed/              ← Auto-generated preprocessed CSV
│
├── models/
│   ├── tfidf_vectorizer.pkl    ← Saved TF-IDF vectorizer
│   ├── fake_news_model.pkl     ← Best trained classifier
│   ├── corpus_tfidf.npz        ← Corpus TF-IDF matrix
│   ├── corpus_metadata.pkl     ← Corpus document metadata
│   └── evaluation_results.json ← All model metrics
│
├── src/
│   ├── preprocessing.py        ← NLP preprocessing pipeline
│   ├── train_model.py          ← Training + model comparison
│   ├── evaluate.py             ← Evaluation metrics loader
│   └── retrieval.py            ← VSM + cosine similarity retrieval
│
├── templates/
│   ├── base.html               ← Shared layout + navbar
│   ├── index.html              ← Home page
│   ├── detect.html             ← Detection input form
│   ├── result.html             ← Results + similar articles
│   ├── evaluate.html           ← Model metrics + confusion matrix
│   ├── about.html              ← About + IR concepts
│   ├── train.html              ← Web-based training trigger
│   └── error.html              ← 404/500 error pages
│
└── static/
    ├── css/style.css           ← Dark glassmorphism design
    └── js/main.js              ← Navbar, animations, interactions
```

---

## ⚙️ Installation & Setup

### 1. Clone / Download the project
```bash
cd fake-news-ir
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Recommended) Download dataset
Download **Fake.csv** and **True.csv** from:  
🔗 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Place both files in `data/raw/`:
```
data/raw/Fake.csv
data/raw/True.csv
```

> **Without these files**, a small built-in sample dataset (24 articles) is used automatically. Models will still train and the system will work — but accuracy will be limited.

### 5. Train the models
```bash
python src/train_model.py
```

This will:
- Load and preprocess the dataset
- Build TF-IDF feature vectors
- Train Logistic Regression, Naive Bayes and Linear SVM
- Evaluate all models and save the best one
- Save all artefacts to `models/`

### 6. Run the Flask app
```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

---

## 🌐 Application Routes

| Route | Description |
|---|---|
| `GET /` | Home page |
| `GET /detect` | Detection input form |
| `POST /detect` | Run analysis → redirect to results |
| `GET /results` | Show prediction + similar articles |
| `GET /evaluate` | Model metrics, confusion matrix, formulae |
| `GET /about` | Project info, IR concepts, viva prep |
| `GET /train` | Web-based training trigger (dev) |
| `POST /api/predict` | JSON API endpoint |
| `GET /api/status` | Model readiness check |

---

## 🔌 JSON API

**Endpoint:** `POST /api/predict`

**Request:**
```json
{
  "headline": "Scientists discover new cancer treatment",
  "body":     "Researchers at MIT announced..."
}
```

**Response:**
```json
{
  "prediction": {
    "label":      "REAL",
    "confidence": 87.4,
    "fake_prob":  12.6,
    "real_prob":  87.4,
    "processed":  "scientist discover new cancer treatment ..."
  },
  "similar": [
    {
      "rank": 1,
      "title": "...",
      "snippet": "...",
      "label": "REAL",
      "similarity": 72.3
    }
  ]
}
```

---

## 🧠 Algorithms & IR Concepts

### TF-IDF
```
TF-IDF(t, d) = TF(t, d) × log(N / df(t))
```

### Cosine Similarity
```
cos(A, B) = (A · B) / (‖A‖ · ‖B‖)
```

### Classifiers Compared
| Classifier | Type | Notes |
|---|---|---|
| Logistic Regression | Discriminative | Best on large datasets |
| Multinomial Naive Bayes | Probabilistic | Fast, good baseline |
| Linear SVM | Margin-based | Strong on text; wrapped with CalibratedClassifierCV for probabilities |

---

## 📊 Evaluation Metrics

- **Accuracy** = (TP + TN) / Total
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1-Score** = 2 × P × R / (P + R)
- **Confusion Matrix** shown on `/evaluate` page

---

## ⚠️ Disclaimer

This system provides **automated ML predictions** based on learned textual patterns.
It is **not** a definitive fact-checker or replacement for professional journalism verification.
Always consult authoritative sources for important news.

---

## 📚 Reference

Savitribai Phule Pune University, B.E. Computer Engineering (2019 Pattern),
Semester VII, Elective IV – Information Retrieval (410245(A)).  
Prescribed mini-project: *"Develop Fake news detection system"*
