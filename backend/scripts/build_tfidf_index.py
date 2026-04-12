import os
import json
import re
import joblib
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from rank_bm25 import BM25Okapi


CHUNKS_PATH = os.path.join("data", "chunks", "chunks.jsonl")
INDEX_DIR = os.path.join("backend", "index")

VECTORIZER_PATH = os.path.join(INDEX_DIR, "tfidf_vectorizer.joblib")
MATRIX_PATH = os.path.join(INDEX_DIR, "tfidf_matrix.joblib")
META_PATH = os.path.join(INDEX_DIR, "chunks_meta.json")

BM25_PATH = os.path.join(INDEX_DIR, "bm25.joblib")
BM25_TOKENS_PATH = os.path.join(INDEX_DIR, "bm25_tokens.joblib")


def _norm_space(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9']+", text)
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]
    return tokens


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    texts_for_indexing = []
    meta = []
    bm25_tokens = []

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            title = _norm_space(row.get("title", "") or "")
            section = _norm_space(row.get("section_title", "") or "")
            heading_path = row.get("heading_path", []) or []
            hp = " > ".join([_norm_space(h) for h in heading_path if _norm_space(h)])
            body = _norm_space(row.get("text", "") or "")

            # Index text that combines the title, section, and body so short headings still contribute.
            index_text = "\n".join([x for x in [title, section, hp, body] if x])
            if not index_text:
                continue

            texts_for_indexing.append(index_text)

            meta.append({
                "id": row["id"],
                "url": row["url"],
                "title": title,
                "chunk_index": row.get("chunk_index", 0),
                "section_title": section,
                "heading_path": heading_path,
            })

            bm25_tokens.append(tokenize(index_text))

    if not texts_for_indexing:
        raise ValueError("No chunk text found. Run chunk_text.py and check data/chunks/chunks.jsonl.")

    # Index builders that create the main TF-IDF matrix and a second BM25 lexical signal.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=60000
    )
    matrix = vectorizer.fit_transform(texts_for_indexing)

    bm25 = BM25Okapi(bm25_tokens)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(matrix, MATRIX_PATH)
    joblib.dump(bm25, BM25_PATH)
    joblib.dump(bm25_tokens, BM25_TOKENS_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Built TF-IDF index with {matrix.shape[0]} chunks")
    print(f"Saved TF-IDF vectorizer/matrix and BM25 index to {INDEX_DIR}")


if __name__ == "__main__":
    main()
