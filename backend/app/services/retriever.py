import json
import os
import re
from typing import Any, Dict, List

import joblib
import numpy as np

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

EXPLAIN_PREFIX_PATTERNS = [
    r"^\s*(?:please\s+)?(?:can|could|would)\s+you\s+",
    r"^\s*(?:please\s+)?(?:do|can|could|would)\s+you\s+offer\s+",
    r"^\s*(?:please\s+)?(?:explain|define|describe)\s+",
    r"^\s*(?:please\s+)?tell\s+me\s+about\s+",
    r"^\s*(?:please\s+)?what\s+is\s+",
    r"^\s*(?:please\s+)?what\s+are\s+",
    r"^\s*(?:please\s+)?who\s+is\s+",
    r"^\s*(?:please\s+)?who\s+are\s+",
]

GENERIC_QUERY_TOKENS = {
    "explain", "what", "is", "are", "tell", "me", "about",
    "who", "can", "could", "would", "please", "do", "you", "offer",
}

PERSONAL_QUERY_TOKENS = {
    "my", "mine", "myself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "son", "daughter", "child", "children", "kid", "kids",
}

PERSONAL_QUERY_PATTERNS = [
    r"\bmy\s+son\b",
    r"\bmy\s+daughter\b",
    r"\bmy\s+child(?:ren)?\b",
    r"\bmy\s+kids?\b",
    r"\b(?:son|daughter|child|children|kid|kids)\b",
    r"\b(?:my|mine|myself)\b",
    r"\b(?:he|him|his|himself)\b",
    r"\b(?:she|her|hers|herself)\b",
]


def _strip_personal_terms(text: str) -> str:
    # Cleanup step that removes family-specific wording so retrieval stays focused on the actual topic.
    t = f" {(text or '').strip()} "
    for pattern in PERSONAL_QUERY_PATTERNS:
        t = re.sub(pattern, " ", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()


class TfidfRetriever:
    def __init__(self, index_dir: str, chunks_path: str):
        # Retriever state that loads the saved index artefacts and combines them at query time.
        self.index_dir = index_dir
        self.chunks_path = chunks_path

        self.vectorizer = None
        self.matrix = None
        self.meta = None
        self.chunk_text_by_id = None

        self.bm25 = None
        self.bm25_tokens = None

        self.embeddings = None
        self.embedding_ids = None
        self._embedder = None

        self.query_expansion_map = None

    def load(self) -> None:
        # Load step that restores the sparse index, metadata, and optional retrieval extras from disk.
        vectorizer_path = os.path.join(self.index_dir, "tfidf_vectorizer.joblib")
        matrix_path = os.path.join(self.index_dir, "tfidf_matrix.joblib")
        meta_path = os.path.join(self.index_dir, "chunks_meta.json")
        bm25_path = os.path.join(self.index_dir, "bm25.joblib")
        bm25_tokens_path = os.path.join(self.index_dir, "bm25_tokens.joblib")
        emb_path = os.path.join(self.index_dir, "embeddings.npy")
        emb_ids_path = os.path.join(self.index_dir, "embeddings_ids.json")

        if not (os.path.exists(vectorizer_path) and os.path.exists(matrix_path) and os.path.exists(meta_path)):
            raise FileNotFoundError("Index files not found. Run build_tfidf_index.py")

        self.vectorizer = joblib.load(vectorizer_path)
        self.matrix = joblib.load(matrix_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        chunk_text_by_id = {}
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                chunk_text_by_id[row["id"]] = row.get("text", "")
        self.chunk_text_by_id = chunk_text_by_id

        if os.path.exists(bm25_path) and os.path.exists(bm25_tokens_path):
            self.bm25 = joblib.load(bm25_path)
            self.bm25_tokens = joblib.load(bm25_tokens_path)

        if os.path.exists(emb_path) and os.path.exists(emb_ids_path):
            # Optional dense artefacts that are only used when the embedding files are present.
            self.embeddings = np.load(emb_path)
            with open(emb_ids_path, "r", encoding="utf-8") as f:
                self.embedding_ids = json.load(f)

        qemap_path = os.path.join("data", "query_expansion.json")
        if os.path.exists(qemap_path):
            with open(qemap_path, "r", encoding="utf-8") as f:
                self.query_expansion_map = json.load(f)

    def is_loaded(self) -> bool:
        return (
            self.vectorizer is not None
            and self.matrix is not None
            and self.meta is not None
            and self.chunk_text_by_id is not None
        )

    def _tokenize(self, text: str) -> List[str]:
        text = (text or "").lower()
        tokens = re.findall(r"[a-z0-9']+", text)
        return [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]

    def _strip_explain_prefix(self, q: str) -> str:
        # Cleanup step that removes generic question prefixes before ranking.
        cleaned = (q or "").strip()
        while True:
            before = cleaned
            for pattern in EXPLAIN_PREFIX_PATTERNS:
                cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip(" .,:;!?")
            if cleaned == before:
                break
        return cleaned

    def _content_tokens(self, tokens: List[str]) -> List[str]:
        # Token filter that removes generic question words before scoring overlap.
        out = [t for t in tokens if t not in GENERIC_QUERY_TOKENS and t not in PERSONAL_QUERY_TOKENS]
        return out or tokens

    def _has_min_words(self, text: str, min_words: int = 4) -> bool:
        return len(re.findall(r"[a-z0-9']+", (text or "").lower())) >= min_words

    def _apply_aliases(self, q: str) -> str:
        # Aliases that map common user wording back to the terms used in the site content.
        text = (q or "")
        grass_target = "we launched a pilot project, grass2grades. This innovative project was increasing"
        grass_phrase_pattern = r"we\s+launched\s+a\s+pilot\s+(?:project|programme|program)\s*,\s*grass2grades\.\s*this\s+innovative\s+(?:project|programme|program)\s+was\s+increasing"
        text = re.sub(r"\bspecial educational needs\b", "send", text, flags=re.IGNORECASE)
        text = re.sub(r"\bapply\b", "register", text, flags=re.IGNORECASE)
        text = re.sub(r"\bview\s+your\b", "see the", text, flags=re.IGNORECASE)
        text = re.sub(r"\bview\s+the\b", "see the", text, flags=re.IGNORECASE)
        text = re.sub(r"\bnewsletters\b", "newsletter", text, flags=re.IGNORECASE)
        text = re.sub(r"(?<!international )\bsummer\s+school\b", "international summer school", text, flags=re.IGNORECASE)
        text = re.sub(r"\binternational\s+summer\s+school\b", "international youth exchange programme", text, flags=re.IGNORECASE)
        text = re.sub(grass_phrase_pattern, "__grass_target__", text, flags=re.IGNORECASE)
        text = re.sub(r"\bgrass2grades\b", "__grass2grades_token__", text, flags=re.IGNORECASE)
        text = re.sub(r"\bprojects?\b", "programme", text, flags=re.IGNORECASE)
        text = re.sub(r"(?<!fun activities and )\bcultural\s+trips\b(?! that help)", "fun activities and cultural trips that help", text, flags=re.IGNORECASE)
        text = re.sub(r"\bgrass\s*2\s*grades\b", "__grass2grades_token__", text, flags=re.IGNORECASE)
        text = re.sub(r"\bmrs\.?\s+karen\s+bryson\b", "Karen Bryson", text, flags=re.IGNORECASE)
        text = re.sub(r"\bkaren\s+bryson\b(?!\s+is\s+an\s+educationalist)", "Karen Bryson is an Educationalist", text, flags=re.IGNORECASE)
        text = text.replace("__grass2grades_token__", grass_target)
        text = text.replace("__grass_target__", grass_target)
        text = re.sub(r"\bleadership\s+programmes?\b", "youth leadership programmes", text, flags=re.IGNORECASE)
        text = re.sub(r"\bleadership\s+programs?\b", "youth leadership programmes", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhome[\s-]?schooling\b", "online homeschooling", text, flags=re.IGNORECASE)
        text = re.sub(r"\bodd[\s-]+girl(?:s)?\b", "odd girls in", text, flags=re.IGNORECASE)
        text = re.sub(r"\bprograme\b", "programme", text, flags=re.IGNORECASE)
        text = re.sub(r"\bprogrammes\b", "programme", text, flags=re.IGNORECASE)
        text = re.sub(r"\bworkshops?\b", "__workshop_bundle__", text, flags=re.IGNORECASE)
        text = re.sub(r"\btalks?\b", "__keynote_bundle__", text, flags=re.IGNORECASE)
        text = re.sub(r"\bspeech(?:es)?\b", "__keynote_bundle__", text, flags=re.IGNORECASE)
        text = re.sub(r"(?:__workshop_bundle__\s*)+", "engaging workshops ", text)
        text = re.sub(r"(?:__keynote_bundle__\s*)+", "engaging and impactful keynote speech ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _normalize_query(self, q: str) -> str:
        q = _strip_personal_terms(q)
        q = re.sub(r"\s+", " ", q)
        q = self._apply_aliases(q)
        q = re.sub(r"\bprograme\b", "programme", q, flags=re.IGNORECASE)
        q = re.sub(r"\bprogrammes\b", "programme", q, flags=re.IGNORECASE)
        q = re.sub(r"\bprogramme\b", "program", q, flags=re.IGNORECASE)
        q = re.sub(r"\bprograms\b", "program", q, flags=re.IGNORECASE)
        q = re.sub(r"\bprogram\b", "program", q, flags=re.IGNORECASE)
        q = re.sub(r"\b(gift\s*-?\s*aid|giftaid)\b", "gift aid", q, flags=re.IGNORECASE)
        q = re.sub(
            r"\b(bf4a|bright\s+futures\s+4\s+all)\b",
            "Bright Futures 4 All",
            q,
            flags=re.IGNORECASE
        )
        stripped = self._strip_explain_prefix(q)
        if stripped:
            q = stripped
        return q

    def _expand_query(self, q: str) -> str:
        q = self._normalize_query(q)
        if not self.query_expansion_map:
            return q

        # Rule that leaves navigation-style queries alone because page opening is handled elsewhere.
        q_low = q.lower()
        nav_triggers = ["where", "find", "open", "go to", "take me", "link", "page"]
        if any(t in q_low for t in nav_triggers):
            return q

        tokens = self._tokenize(q)
        if len(tokens) > 10:
            return q

        expansions = []
        for key, values in self.query_expansion_map.items():
            if key.lower() in q_low:
                expansions.extend(values)

        expansions = expansions[:8]
        if expansions:
            return q + " " + " ".join(expansions)
        return q

    def _lexical_coverage_boost(self, query_tokens: List[str], meta: Dict[str, Any], chunk_text: str) -> float:
        # Coverage boost that rewards chunks matching more of the actual query terms.
        if not query_tokens:
            return 0.0
        qset = set(query_tokens)
        title = (meta.get("title") or "").lower()
        section = (meta.get("section_title") or "").lower()
        heading_path = " ".join(meta.get("heading_path") or []).lower()
        body = (chunk_text or "").lower()
        hay = f" {title} {section} {heading_path} {body} "

        hits = 0
        for tok in qset:
            if f" {tok} " in hay:
                hits += 1

        coverage = hits / max(1, len(qset))
        boost = 0.18 * coverage

        phrase = " ".join(query_tokens).strip().lower()
        if len(query_tokens) >= 2 and phrase:
            if phrase in f"{title} {section} {heading_path}":
                boost += 0.10
            elif phrase in body:
                boost += 0.06

        if hits == 0:
            boost -= 0.04
        return boost

    def _rrf_scores(
        self,
        candidate_indices: np.ndarray,
        tfidf_scores: np.ndarray,
        bm25_scores: np.ndarray | None,
        dense_scores: np.ndarray | None,
        query_tokens: List[str],
    ) -> np.ndarray:
        # Fusion step that combines sparse and dense ranking signals without heavy score calibration.
        k = 60.0
        scores = np.zeros(len(candidate_indices), dtype=np.float32)

        def rank_map(arr: np.ndarray) -> dict[int, int]:
            order = np.argsort(-arr)[:120]
            return {int(idx): int(rank + 1) for rank, idx in enumerate(order)}

        tf_ranks = rank_map(tfidf_scores)
        bm_ranks = rank_map(bm25_scores) if bm25_scores is not None else {}
        de_ranks = rank_map(dense_scores) if dense_scores is not None else {}

        q_len = len(query_tokens)
        if q_len <= 2:
            w_tfidf, w_bm25, w_dense = 0.35, 0.50, 0.15
        elif q_len <= 5:
            w_tfidf, w_bm25, w_dense = 0.30, 0.45, 0.25
        else:
            w_tfidf, w_bm25, w_dense = 0.25, 0.35, 0.40

        if bm25_scores is None:
            w_bm25 = 0.0
        if dense_scores is None:
            w_dense = 0.0

        w_sum = w_tfidf + w_bm25 + w_dense
        if w_sum <= 0:
            w_tfidf = 1.0
            w_sum = 1.0
        w_tfidf /= w_sum
        w_bm25 /= w_sum
        w_dense /= w_sum

        for pos, idx in enumerate(candidate_indices):
            idx_int = int(idx)
            sc = 0.0
            r_tf = tf_ranks.get(idx_int)
            if r_tf is not None:
                sc += w_tfidf * (1.0 / (k + r_tf))
            if bm_ranks:
                r_bm = bm_ranks.get(idx_int)
                if r_bm is not None:
                    sc += w_bm25 * (1.0 / (k + r_bm))
            if de_ranks:
                r_de = de_ranks.get(idx_int)
                if r_de is not None:
                    sc += w_dense * (1.0 / (k + r_de))
            scores[pos] = sc
        return scores

    def _boost(self, query_tokens: List[str], meta: Dict[str, Any]) -> float:
        # Heuristic boosts that help obvious high-value pages win when the query points to them clearly.
        boost = 0.0
        title = (meta.get("title") or "").lower()
        section = (meta.get("section_title") or "").lower()
        heading_path = " ".join(meta.get("heading_path") or []).lower()
        url = (meta.get("url") or "").lower()

        for token in query_tokens:
            if token in title or token in section or token in heading_path:
                boost += 0.006
            if token in url:
                boost += 0.004

        q = " ".join(query_tokens)

        if any(t in q for t in ["contact", "email", "phone", "address", "message", "call", "speak"]):
            if "contact" in url:
                boost += 0.12

        if any(t in q for t in ["about", "mission", "history", "purpose", "registered", "charity"]):
            if "about" in url or "about" in title:
                boost += 0.12

        if any(t in q for t in ["tutor", "tutoring", "mentoring", "therapy", "therapist", "counselling", "counseling", "one-to-one"]):
            if "tutoring" in url or "tutor" in url or "tutors" in title or "therap" in title:
                boost += 0.15

        return min(boost, 0.25)

    def _get_embedder(self):
        # Lazy loader that only creates the embedder when dense retrieval is available.
        if self._embedder is not None:
            return self._embedder
        from sentence_transformers import SentenceTransformer
        self._embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self._embedder

    def _dense_scores(self, query: str) -> np.ndarray:
        # Dense scoring path that returns no scores when the embedding files have not been built.
        if self.embeddings is None or self.embedding_ids is None:
            return None
        embedder = self._get_embedder()
        q_emb = embedder.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)[0]
        emb = self.embeddings
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)
        sims = emb @ q_emb
        scores = np.zeros(len(self.meta), dtype=np.float32)
        id_to_idx = {m["id"]: i for i, m in enumerate(self.meta)}
        for j, cid in enumerate(self.embedding_ids):
            i = id_to_idx.get(cid)
            if i is not None:
                scores[i] = sims[j]
        return scores

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.is_loaded():
            self.load()

        # Query preparation step that normalises and expands the user text before ranking.
        q = self._expand_query(query)
        q_tokens = self._content_tokens(self._tokenize(q))
        q_for_vector = " ".join(q_tokens).strip() or q

        qv = self.vectorizer.transform([q_for_vector])
        tfidf_sims = cosine_similarity(qv, self.matrix).flatten().astype(np.float32)

        bm25_scores = None
        if self.bm25 is not None:
            bm25_scores = np.array(self.bm25.get_scores(q_tokens), dtype=np.float32)

        dense_scores = None
        if self.embeddings is not None and self.embedding_ids is not None:
            dense_scores = self._dense_scores(q_for_vector)

        if bm25_scores is None and dense_scores is None:
            # Fallback ranking path that still works when only TF-IDF is available.
            top_idx = np.argsort(-tfidf_sims)
            results = []
            for i in top_idx:
                idx = int(i)
                score = float(tfidf_sims[idx])
                score += self._boost(q_tokens, self.meta[idx])
                score += self._lexical_coverage_boost(q_tokens, self.meta[idx], self.chunk_text_by_id.get(self.meta[idx]["id"], ""))
                row = self._format_result(idx, score)
                if not self._has_min_words(row.get("text", ""), min_words=4):
                    continue
                results.append(row)
                if len(results) >= top_k:
                    break
            return results

        TOP_CANDIDATES = 60
        candidates = []

        # Candidate selection step that lets each signal nominate strong matches before fusion.
        tf_top = np.argsort(-tfidf_sims)[:TOP_CANDIDATES]
        candidates.append(tf_top)

        if bm25_scores is not None:
            bm_top = np.argsort(-bm25_scores)[:TOP_CANDIDATES]
            candidates.append(bm_top)

        if dense_scores is not None:
            de_top = np.argsort(-dense_scores)[:TOP_CANDIDATES]
            candidates.append(de_top)

        candidate_indices = np.unique(np.concatenate(candidates))
        combined = self._rrf_scores(
            candidate_indices=candidate_indices,
            tfidf_scores=tfidf_sims,
            bm25_scores=bm25_scores,
            dense_scores=dense_scores,
            query_tokens=q_tokens,
        )

        for i, idx in enumerate(candidate_indices):
            combined[i] += self._boost(q_tokens, self.meta[int(idx)])
            chunk_id = self.meta[int(idx)]["id"]
            chunk_text = self.chunk_text_by_id.get(chunk_id, "")
            combined[i] += self._lexical_coverage_boost(q_tokens, self.meta[int(idx)], chunk_text)

        order = np.argsort(-combined)

        results = []
        for pos in order:
            idx = int(candidate_indices[pos])
            row = self._format_result(idx, float(combined[pos]))
            if not self._has_min_words(row.get("text", ""), min_words=4):
                continue
            results.append(row)
            if len(results) >= top_k:
                break
        return results

    def search_filtered(self, query: str, allowed_url_substrings: list[str], top_k: int = 3):
        # Filtered search path that limits results to a known set of page URLs.
        base_results = self.search(query, top_k=60)
        allowed_lower = [s.lower() for s in allowed_url_substrings]

        filtered = []
        for r in base_results:
            url = (r.get("url") or "").lower()
            if any(sub in url for sub in allowed_lower):
                filtered.append(r)

        return filtered[:top_k] if filtered else base_results[:top_k]

    def _format_result(self, i: int, score: float) -> Dict[str, Any]:
        # Formatting step that returns a consistent result shape to the chatbot layer.
        m = self.meta[i]
        chunk_id = m["id"]
        return {
            "score": score,
            "id": chunk_id,
            "url": m["url"],
            "title": m.get("title", ""),
            "chunk_index": m.get("chunk_index", 0),
            "section_title": m.get("section_title", ""),
            "heading_path": m.get("heading_path", []),
            "text": self.chunk_text_by_id.get(chunk_id, "")
        }
