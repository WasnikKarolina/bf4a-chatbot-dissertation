import json
import os
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

DATA_PATH = os.path.join("data", "training", "intents.csv")
OUT_DIR = os.path.join("backend", "models")

VECTORIZER_PATH = os.path.join(OUT_DIR, "intent_vectorizer.joblib")
MODEL_PATH = os.path.join(OUT_DIR, "intent_model.joblib")
LABELS_PATH = os.path.join(OUT_DIR, "intent_labels.json")
THRESHOLDS_PATH = os.path.join(OUT_DIR, "intent_thresholds.json")


def normalize_text(s: str) -> str:
    # Normalisation step that mirrors the cleanup used by the runtime intent service.
    s = str(s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].astype(str).fillna("").map(normalize_text)
    df["label"] = df["label"].astype(str).fillna("OTHER")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    # Feature setup that combines word and character signals to handle both meaning and messy spelling.
    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(lowercase=False, ngram_range=(1, 2), max_features=60000, sublinear_tf=True)),
        ("char", TfidfVectorizer(lowercase=False, analyzer="char_wb", ngram_range=(3, 5), max_features=80000, sublinear_tf=True)),
    ])
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=6000, C=3.0, class_weight="balanced", solver="lbfgs")
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)

    print("\n Classification Report")
    print(classification_report(y_test, y_pred))

    print("\n Confusion Matrix (labels order)")
    labels_sorted = sorted(df["label"].unique())
    print(labels_sorted)
    print(confusion_matrix(y_test, y_pred, labels=labels_sorted))

    top1 = y_proba.max(axis=1)
    top2 = np.partition(y_proba, -2, axis=1)[:, -2]
    margins = top1 - top2
    y_true_arr = y_test.to_numpy()
    correct = y_pred == y_true_arr

    # Threshold estimation that uses the weaker end of the correct predictions as a runtime baseline.
    if correct.any():
        global_conf = float(np.percentile(top1[correct], 10))
        global_margin = float(np.percentile(margins[correct], 10))
    else:
        global_conf = 0.45
        global_margin = 0.12
    global_conf = float(np.clip(global_conf, 0.30, 0.85))
    global_margin = float(np.clip(global_margin, 0.08, 0.40))

    per_label_conf = {}
    per_label_margin = {}
    for label in labels_sorted:
        mask = (y_true_arr == label) & correct
        if mask.sum() >= 2:
            lbl_conf = float(np.percentile(top1[mask], 10))
            lbl_margin = float(np.percentile(margins[mask], 10))
            per_label_conf[label] = float(np.clip(lbl_conf, 0.25, 0.90))
            per_label_margin[label] = float(np.clip(lbl_margin, 0.08, 0.40))
        else:
            per_label_conf[label] = global_conf
            per_label_margin[label] = global_margin

    min_good_match_conf = float(np.clip(max(0.45, global_conf), 0.45, 0.90))
    min_good_match_margin = float(np.clip(max(0.12, global_margin), 0.12, 0.40))

    thresholds = {
        "global_confidence": global_conf,
        "global_margin": global_margin,
        "min_good_match_confidence": min_good_match_conf,
        "min_good_match_margin": min_good_match_margin,
        "per_label_confidence": per_label_conf,
        "per_label_margin": per_label_margin,
    }

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels_sorted, f, indent=2)
    with open(THRESHOLDS_PATH, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\nSaved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved labels to {LABELS_PATH}")
    print(f"Saved thresholds to {THRESHOLDS_PATH}")
    print(
        f"Thresholds: conf>={global_conf:.3f}, margin>={global_margin:.3f}, "
        f"good_match_conf>={min_good_match_conf:.3f}, good_match_margin>={min_good_match_margin:.3f}"
    )


if __name__ == "__main__":
    main()
