import json
import os
import re
from typing import Tuple

import joblib


MODEL_DIR = os.path.join("backend", "models")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "intent_vectorizer.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "intent_model.joblib")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "intent_thresholds.json")


class IntentService:
    def __init__(self):
        # Model handles that stay empty until the first prediction so startup stays lightweight.
        self.vectorizer = None
        self.model = None
        self.thresholds = {
            "global_confidence": 0.25,
            "global_margin": 0.05,
            "per_label_confidence": {},
            "per_label_margin": {},
        }

    def _normalize_text(self, text: str) -> str:
        # Normalisation step that mirrors the same cleanup used during model training.
        t = str(text or "").strip().lower()
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def load(self):
        if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH)):
            raise FileNotFoundError("Intent model not found.")

        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.model = joblib.load(MODEL_PATH)
        if os.path.exists(THRESHOLDS_PATH):
            with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    self.thresholds.update(loaded)

    def is_loaded(self) -> bool:
        return self.vectorizer is not None and self.model is not None

    def predict(self, text: str) -> Tuple[str, float]:
        if not self.is_loaded():
            self.load()

        norm = self._normalize_text(text)
        X = self.vectorizer.transform([norm])
        proba = getattr(self.model, "predict_proba", None)

        label = self.model.predict(X)[0]

        if proba:
            probs = self.model.predict_proba(X)[0]
            order = probs.argsort()
            top_idx = int(order[-1])
            second_idx = int(order[-2]) if len(order) > 1 else top_idx
            top_conf = float(probs[top_idx])
            second_conf = float(probs[second_idx]) if len(order) > 1 else 0.0
            margin = top_conf - second_conf

            pred_label = self.model.classes_[top_idx]
            conf_floor = float(self.thresholds.get("global_confidence", 0.25))
            margin_floor = float(self.thresholds.get("global_margin", 0.05))
            conf_floor = min(max(conf_floor, 0.15), 0.35)
            margin_floor = min(max(margin_floor, 0.0), 0.10)

            # Confidence rule that sends uncertain predictions back to OTHER instead of forcing a label.
            if top_conf < conf_floor or margin < margin_floor:
                return "OTHER", top_conf
            label = pred_label
            conf = top_conf
        else:
            conf = 1.0

        return label, conf
