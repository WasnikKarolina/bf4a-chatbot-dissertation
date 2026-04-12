import os
import json
import re
import numpy as np

from sentence_transformers import SentenceTransformer


CHUNKS_PATH = os.path.join("data", "chunks", "chunks.jsonl")
INDEX_DIR = os.path.join("backend", "index")
META_PATH = os.path.join(INDEX_DIR, "chunks_meta.json")

EMB_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
EMB_IDS_PATH = os.path.join(INDEX_DIR, "embeddings_ids.json")


def _norm_space(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _load_chunks_text() -> dict:
    # Loader that rebuilds the chunk text map so the embedding file lines up with the TF-IDF ids.
    by_id = {}
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_id[row["id"]] = {
                "text": row.get("text", ""),
                "title": row.get("title", ""),
                "section_title": row.get("section_title", ""),
                "heading_path": row.get("heading_path", []),
            }
    return by_id


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    if not os.path.exists(META_PATH):
        raise FileNotFoundError("Missing chunks_meta.json. Run build_tfidf_index.py first.")

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    chunks = _load_chunks_text()

    ids = []
    texts = []
    for m in meta:
        cid = m["id"]
        c = chunks.get(cid, {})
        title = _norm_space(c.get("title") or m.get("title") or "")
        section = _norm_space(c.get("section_title") or m.get("section_title") or "")
        hp = c.get("heading_path") or m.get("heading_path") or []
        hp = " > ".join([_norm_space(h) for h in hp if _norm_space(h)])
        body = _norm_space(c.get("text") or "")
        index_text = "\n".join([x for x in [title, section, hp, body] if x])
        ids.append(cid)
        texts.append(index_text)

    # Embedding step that adds an optional dense-search signal when the files are present.
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype=np.float32)

    np.save(EMB_PATH, embs)
    with open(EMB_IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)

    print(f"Saved embeddings: {EMB_PATH} ({embs.shape[0]} vectors, dim={embs.shape[1]})")
    print(f"Saved ids: {EMB_IDS_PATH}")


if __name__ == "__main__":
    main()
