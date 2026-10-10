"""Train and evaluate a spam classifier.

Usage:  python train.py [--data data/emails.csv]

Accepted CSV formats:
  * columns: label,text          (label = spam/ham or 1/0)
  * Kaggle SMS spam format: v1,v2 (latin-1 encoded)
  * columns: Category,Message
"""
import argparse
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent


def resolve_data_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


def load(path):
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    cols = {c.lower(): c for c in df.columns}
    for lab, txt in [("label", "text"), ("v1", "v2"), ("category", "message"), ("class", "text")]:
        if lab in cols and txt in cols:
            df = df[[cols[lab], cols[txt]]]
            df.columns = ["label", "text"]
            break
    else:
        raise SystemExit(f"Unrecognized columns: {list(df.columns)}")
    df = df.dropna()
    df["label"] = df["label"].astype(str).str.lower().map(
        {"spam": 1, "ham": 0, "1": 1, "0": 0, "legitimate": 0})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df


def build(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english",
                                  ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", clf),
    ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/emails.csv")
    args = ap.parse_args()

    data_path = resolve_data_path(args.data)
    if not data_path.exists():
        sample_script = BASE_DIR / "make_sample_data.py"
        if sample_script.exists():
            subprocess.run([sys.executable, str(sample_script)], cwd=str(BASE_DIR), check=True)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

    df = load(str(data_path))
    print(f"Loaded {len(df)} rows ({df.label.sum()} spam, {(df.label == 0).sum()} ham)")
    X_tr, X_te, y_tr, y_te = train_test_split(
        df.text, df.label, test_size=0.2, random_state=42, stratify=df.label)

    candidates = {
        "naive_bayes": build(MultinomialNB(alpha=0.1)),
        "logreg": build(LogisticRegression(max_iter=1000, C=10, class_weight="balanced")),
    }
    best_name, best_model, best_f1 = None, None, -1
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rep = classification_report(y_te, pred, target_names=["ham", "spam"], output_dict=True)
        f1 = rep["spam"]["f1-score"]
        print(f"\n=== {name} (spam F1 = {f1:.4f}) ===")
        print(classification_report(y_te, pred, target_names=["ham", "spam"]))
        print("Confusion matrix [ham, spam]:\n", confusion_matrix(y_te, pred))
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, model, f1

    # Refit best model on all data and save
    best_model.fit(df.text, df.label)
    model_dir = BASE_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_dir / "spam_model.joblib")
    print(f"\nSaved best model ({best_name}) to {model_dir / 'spam_model.joblib'}")
