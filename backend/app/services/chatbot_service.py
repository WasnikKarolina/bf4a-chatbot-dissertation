import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from ..models.schemas import Action, Citation, ClarificationOption
from .intent_service import IntentService
from .page_summary_service import PageSummaryService
from .retriever import TfidfRetriever

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

CONFIRM_YES = {"yes"}
CONFIRM_NO = {"no"}
QUIZ_EXIT = {"stop quiz", "quit quiz", "end quiz", "cancel quiz", "stop", "quit"}

CLARIFICATION_REPLY = "I apologise, I’m not completely sure I understood your question. You could try rephrasing it, or choose one of the examples below:"
KB_NOISE_TOKENS = {
    "the", "a", "an", "your", "our", "their",
    "view", "see", "open", "page", "link",
    "what", "is", "are", "tell", "me", "about",
    "can", "could", "would", "do", "you", "please",
}

ResponsePayload = Tuple[
    str,
    List[Citation],
    List[Action],
    List[str],
    float,
    bool,
    str,
    List[ClarificationOption],
    str,
    float,
]
# Type alias that keeps the larger return signatures readable across the service methods.
SessionStateMap = Dict[str, Dict[str, Any]]


def _strip_personal_terms(text: str) -> str:
    # Cleanup step that removes family-specific wording before matching.
    t = f" {(text or '').strip()} "
    for pattern in PERSONAL_QUERY_PATTERNS:
        t = re.sub(pattern, " ", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()


def _augment_program_terms(t: str) -> str:
    if not t:
        return ""
    has_program = re.search(r"\bprogram\b", t, flags=re.IGNORECASE) is not None
    has_programme = re.search(r"\bprogramme\b", t, flags=re.IGNORECASE) is not None
    has_programe = re.search(r"\bprograme\b", t, flags=re.IGNORECASE) is not None
    has_programmes = re.search(r"\bprogrammes\b", t, flags=re.IGNORECASE) is not None
    extra = ""
    if has_program and not (has_programme or has_programe or has_programmes):
        extra = " programme"
    elif (has_programme or has_programe or has_programmes) and not has_program:
        extra = " program"
    t = re.sub(r"\bprograme\b", "programme", t, flags=re.IGNORECASE)
    t = re.sub(r"\bprogrammes\b", "programme", t, flags=re.IGNORECASE)
    t = re.sub(r"\bprogramme\b", "program", t, flags=re.IGNORECASE)
    return (t + extra).strip()


def _apply_query_aliases(text: str) -> str:
    # Aliases that map common wording variations back to the terms used in the site content.
    t = (text or "").strip()
    grass_target = "we launched a pilot project, grass2grades. This innovative project was increasing"
    grass_phrase_pattern = r"we\s+launched\s+a\s+pilot\s+(?:project|programme|program)\s*,\s*grass2grades\.\s*this\s+innovative\s+(?:project|programme|program)\s+was\s+increasing"
    t = re.sub(r"\bspecial educational needs\b", "send", t, flags=re.IGNORECASE)
    t = re.sub(r"\bapply\b", "register", t, flags=re.IGNORECASE)
    t = re.sub(r"\bview\s+your\b", "see the", t, flags=re.IGNORECASE)
    t = re.sub(r"\bview\s+the\b", "see the", t, flags=re.IGNORECASE)
    t = re.sub(r"\bnewsletters\b", "newsletter", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<!international )\bsummer\s+school\b", "international summer school", t, flags=re.IGNORECASE)
    t = re.sub(r"\binternational\s+summer\s+school\b", "international youth exchange programme", t, flags=re.IGNORECASE)
    t = re.sub(grass_phrase_pattern, "__grass_target__", t, flags=re.IGNORECASE)
    t = re.sub(r"\bgrass2grades\b", "__grass2grades_token__", t, flags=re.IGNORECASE)
    t = re.sub(r"\bprojects?\b", "programme", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<!fun activities and )\bcultural\s+trips\b(?! that help)", "fun activities and cultural trips that help", t, flags=re.IGNORECASE)
    t = re.sub(r"\bgrass\s*2\s*grades\b", "__grass2grades_token__", t, flags=re.IGNORECASE)
    t = re.sub(r"\bmrs\.?\s+karen\s+bryson\b", "Karen Bryson", t, flags=re.IGNORECASE)
    t = re.sub(r"\bkaren\s+bryson\b(?!\s+is\s+an\s+educationalist)", "Karen Bryson is an Educationalist", t, flags=re.IGNORECASE)
    t = t.replace("__grass2grades_token__", grass_target)
    t = t.replace("__grass_target__", grass_target)
    t = re.sub(r"\bleadership\s+programmes?\b", "youth leadership programmes", t, flags=re.IGNORECASE)
    t = re.sub(r"\bleadership\s+programs?\b", "youth leadership programmes", t, flags=re.IGNORECASE)
    t = re.sub(r"\bhome[\s-]?schooling\b", "online homeschooling", t, flags=re.IGNORECASE)
    t = re.sub(r"\bodd[\s-]+girl(?:s)?\b", "odd girls in", t, flags=re.IGNORECASE)
    t = re.sub(r"\bprograme\b", "programme", t, flags=re.IGNORECASE)
    t = re.sub(r"\bprogrammes\b", "programme", t, flags=re.IGNORECASE)
    t = re.sub(r"\bsend\s+workshops?\b", "__workshop_bundle__", t, flags=re.IGNORECASE)
    t = re.sub(r"\bworkshops?\b", "__workshop_bundle__", t, flags=re.IGNORECASE)
    t = re.sub(r"\btalks?\b", "__keynote_bundle__", t, flags=re.IGNORECASE)
    t = re.sub(r"\bspeech(?:es)?\b", "__keynote_bundle__", t, flags=re.IGNORECASE)
    t = re.sub(r"(?:__workshop_bundle__\s*)+", "a variety of engaging workshops ", t)
    t = re.sub(r"(?:__keynote_bundle__\s*)+", "engaging and impactful keynote speech ", t)
    return re.sub(r"\s+", " ", t).strip()


def _normalise_query(text: str) -> str:
    t = _strip_personal_terms(text)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(
        r"\b(bf4a|bright\s+futures\s+4\s+all)\b",
        "Bright Futures 4 All",
        t,
        flags=re.IGNORECASE,
    )
    t = _apply_query_aliases(t)
    t = _augment_program_terms(t)
    return t


def _semantic_query_core(text: str) -> str:
    t = _normalise_query(text)
    while True:
        before = t
        for pattern in EXPLAIN_PREFIX_PATTERNS:
            t = re.sub(pattern, "", t, flags=re.IGNORECASE)
        t = t.strip(" .,:;!?")
        if t == before:
            break
    return t or _normalise_query(text)


def _query_content_tokens(text: str) -> list[str]:
    tokens = _tokenize_simple(_semantic_query_core(text))
    filtered = [t for t in tokens if t not in GENERIC_QUERY_TOKENS and t not in PERSONAL_QUERY_TOKENS]
    return filtered or tokens


def _strip_trailing_ellipsis(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"(?:\.{3}|\u2026)+\s*$", "", text)
    return text.strip()


def _apply_text_replacements(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return ""
    t = re.sub(
        r"\bkaren\s+bryson\s+karen\s+bryson\s+is\s+an\s+educationalist\b",
        "Karen Bryson is an Educationalist",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"\bkaren\s+bryson\s+karen\s+bryson\s+is\b", "Karen Bryson is an Educationalist", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()


def _ensure_finished_sentence(text: str) -> str:
    t = _apply_text_replacements(text)
    t = _strip_trailing_ellipsis(t)
    if not t:
        return ""
    if t[-1] in ".!?:": 
        return t
    matches = list(re.finditer(r"[.!?]", t))
    if matches:
        return t[:matches[-1].end()].strip()
    return t + "."


def _ensure_finished_reply(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"(?<!\n)([.!?])\s+(Would you like me to open\b)", r"\1\n\2", text)
    if "\n" not in text and "\r" not in text:
        return _ensure_finished_sentence(text)

    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""

    for idx, line in enumerate(lines):
        lines[idx] = line.strip()

    lines[-1] = _ensure_finished_sentence(lines[-1])
    return "\n".join(lines)


def truncate_to_sentence(text: str, max_chars: int) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    t = _strip_trailing_ellipsis(t)
    if len(t) <= max_chars:
        return _ensure_finished_sentence(t)
    cut = t[:max_chars].rstrip()
    matches = list(re.finditer(r"[.!?]", cut))
    if matches:
        return _ensure_finished_sentence(cut[:matches[-1].end()].strip())
    return _ensure_finished_sentence(cut.rsplit(" ", 1)[0].strip())


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\bincluding\s*:\s+", "including. ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<![.!?])\s+(Book our CEO\b)", r". \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    parts: list[str] = []
    for block in text.split("\n"):
        block = re.sub(r"\s+", " ", block).strip()
        if not block:
            continue
        parts.extend(re.split(r"(?<=[.!?])\s+", block))
    return [p.strip() for p in parts if p and p.strip()]


def _normalise_summary_phrase(text: str) -> str:
    t = (text or "").lower().replace("&", " and ")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _word_count(text: str) -> int:
    return len(_tokenize_simple(text))


BAD_SENTENCE_STARTS = (
    "register", "book now", "top of page", "home", "donate", "log in",
    "menu", "search", "more", "click here"
)

NOISE_CONTAINS = (
    "if you like our work and would like to support us",
    "company no",
    "all rights reserved",
    "copyright bright futures 4 all",
    "tel:",
    "mob:",
    "info@brightfutures4all.com",
    "open this page in a new tab",
)


def _clean_snippet(s: str) -> str:
    s = _apply_text_replacements(s)
    s = _strip_trailing_ellipsis(s)
    s = re.sub(r",?\s+including\.$", ".", s, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).strip()


def _is_noise_sentence(s: str) -> bool:
    lo = (s or "").lower()
    if not lo:
        return True
    if any(x in lo for x in NOISE_CONTAINS):
        return True
    if lo.count("|") >= 2:
        return True
    if re.search(r"\b\d{3,}\s*\d{3,}\s*\d{3,}\b", lo):
        return True
    return False


def _strip_header_lines(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\b[^.\n]{0,120}\|\s*Bright Futures 4 All\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\|\s*Bright Futures 4 All\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"if you like our work and would like to support us, please consider donating", "", text, flags=re.IGNORECASE)
    text = re.sub(r"BRIGHT FUTURES 4 ALL C\.I\.C Company No:[^.\n]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Copyright Bright Futures 4 All[^.\n]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+\|\s+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


POSITIVE_WORDS = {
    "great", "amazing", "fantastic", "brilliant", "wonderful", "helpful", "supportive",
    "inclusive", "kind", "friendly", "excellent", "awesome", "fun", "thank"
}

FIRST_PERSON_MARKERS = [
    " i ", " we ", " my ", " our ", " me ", " us ",
    "my son", "my daughter", "my child", "my kids", "my family"
]

TESTIMONIAL_PHRASES = [
    "first time", "thank you", "so much", "very helpful", "great fun",
    "couldn't have", "really appreciate", "made a difference"
]


def looks_like_testimonial(s: str) -> bool:
    if not s:
        return False
    if '"' in s:
        return True
    lo = s.lower()
    lo_spaced = f" {lo} "
    if any(p in lo for p in TESTIMONIAL_PHRASES):
        return True
    has_first_person = any(m in lo_spaced for m in FIRST_PERSON_MARKERS)
    has_positive = any(w in lo for w in POSITIVE_WORDS)
    if has_first_person and has_positive:
        return True
    if len(lo.split()) <= 12 and has_positive and ("was" in lo or "were" in lo):
        return True
    if any(x in lo for x in ["mum", "mom", "grandmother", "grandma", "parent", "carer"]):
        if has_positive or has_first_person:
            return True
    return False


def user_wants_testimonials(query: str) -> bool:
    q = (query or "").lower()
    triggers = ["testimonial", "testimonials", "success story", "success stories", "feedback", "reviews"]
    return any(t in q for t in triggers)


def _user_asked_fsm(query: str) -> bool:
    q = (query or "").lower()
    return any(x in q for x in ["fsm", "free school meals", "eligib", "who can", "free", "cost", "qualif"])


def _definition_boost(sentence: str, query: str) -> float:
    s = (sentence or "").lower()
    q_core = _semantic_query_core(query).lower()
    q_tokens = set(_tokenize_simple(q_core))
    if not q_tokens:
        return 0.0
    overlap = sum(1 for t in q_tokens if t in s)
    if overlap == 0:
        return 0.0
    boost = 0.0
    boost += min(0.018, 0.004 * overlap)
    if any(x in s for x in [" is ", " are ", "means", "refers to", "we offer", "we provide", "we run", "includes", "programme", "program", "support"]):
        boost += 0.016
    if any(x in s for x in ["we offer", "we provide", "we run", "includes", "programme", "program"]):
        boost += 0.010
    return boost


def _holiday_fsm_penalty(sentence: str, query: str) -> float:
    s = (sentence or "").lower()
    mentions_fsm = ("free school meals" in s) or ("fsm" in s)
    if not mentions_fsm:
        return 0.0
    if _user_asked_fsm(query):
        return 0.045
    return -0.060


def _rank_sentences(retriever: TfidfRetriever, query: str, sentences: list[str]) -> list[tuple[float, str]]:
    query_core = _semantic_query_core(query)
    q_tokens = set(_query_content_tokens(query_core))
    qv = retriever.vectorizer.transform([query_core])
    sv = retriever.vectorizer.transform(sentences)
    scores = (sv @ qv.T).toarray().flatten()
    out: list[tuple[float, str]] = []
    for s, sc in zip(sentences, scores):
        score = float(sc)
        s_tokens = set(_tokenize_simple(s))
        overlap = len(q_tokens.intersection(s_tokens)) if q_tokens else 0
        if q_tokens:
            score += 0.012 * (overlap / max(1, len(q_tokens)))
            if overlap == 0:
                score -= 0.080
        score += _definition_boost(s, query_core)
        score += _holiday_fsm_penalty(s, query)
        out.append((score, s))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def _best_sentences_across_results(
    retriever: TfidfRetriever,
    query: str,
    results: list[dict],
    max_sentences: int = 2,
    max_chars: int = 360
) -> tuple[str, float, list[Citation]]:
    if not results:
        return "", 0.0, []

    allow_testimonials = user_wants_testimonials(query)
    avoid_fsm_lead = not _user_asked_fsm(query)
    query_tokens = set(_query_content_tokens(query))
    query_core = _semantic_query_core(query).lower()
    is_compact_workshop_query = any(
        term in query_core for term in ["workshop", "workshops", "talk", "talks", "speech", "speeches", "keynote"]
    )
    max_sentences_eff = 1 if len(query_tokens) <= 1 or is_compact_workshop_query else max_sentences
    max_chars_eff = min(max_chars, 240) if is_compact_workshop_query else max_chars
    candidates: list[tuple[float, str, str, str, int]] = []

    for r in results:
        text = _strip_header_lines(r.get("text") or "")
        url = r.get("url") or ""
        title = r.get("title") or "Bright Futures 4 All"
        if not text:
            continue
        if _word_count(text) < 4:
            continue
        sents = [_clean_snippet(x) for x in _split_sentences(text)]
        sents = [s for s in sents if len(s) >= 30]
        sents = [s for s in sents if _word_count(s) >= 4]
        sents = [s for s in sents if not any(s.lower().startswith(p) for p in BAD_SENTENCE_STARTS)]
        sents = [s for s in sents if not _is_noise_sentence(s)]
        if not allow_testimonials:
            sents = [s for s in sents if not looks_like_testimonial(s)]
        if not sents:
            continue

        try:
            ranked = _rank_sentences(retriever, query, sents)
        except Exception:
            ranked = [(float(r.get("score", 0.0)), s) for s in sents]

        base_score = float(r.get("score", 0.0))
        for sc, s in ranked[:10]:
            s_tokens = set(_tokenize_simple(s))
            overlap = len(query_tokens.intersection(s_tokens)) if query_tokens else 0
            if query_tokens and overlap == 0:
                continue
            combined = 0.72 * float(sc) + 0.28 * base_score
            candidates.append((combined, s, url, title, overlap))

    if not candidates:
        return "", 0.0, []

    candidates.sort(key=lambda x: (x[0], x[4]), reverse=True)

    chosen: list[str] = []
    seen: set[str] = set()
    citations: list[Citation] = []
    used_urls: set[str] = set()
    primary_url = candidates[0][2]
    top_score = float(candidates[0][0])

    for sc, s, url, title, overlap in candidates:
        key = re.sub(r"\s+", " ", s.lower()).strip()
        if key in seen:
            continue
        if any(_sentence_content_overlap(s, existing) >= 0.72 for existing in chosen):
            continue
        if len(chosen) > 0:
            if url != primary_url and (sc < top_score * 0.90 or overlap < 1):
                continue
            if url == primary_url and sc < top_score * 0.62:
                continue
        if avoid_fsm_lead and len(chosen) == 0 and ("free school meals" in key or "fsm" in key):
            non_fsm_exists = any(("free school meals" not in c[1].lower() and "fsm" not in c[1].lower()) for c in candidates)
            if non_fsm_exists:
                continue
        seen.add(key)
        chosen.append(s)
        if url and url not in used_urls and len(citations) < 1:
            citations.append(Citation(title=title, url=url))
            used_urls.add(url)
        if len(chosen) >= max_sentences_eff:
            break

    if not chosen:
        return "", float(candidates[0][0]), []

    snippet = " ".join(chosen).strip()
    snippet = truncate_to_sentence(snippet, max_chars_eff)
    return snippet, float(candidates[0][0]), citations


def _tokenize_simple(text: str) -> list[str]:
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9']+", text)
    return [t for t in tokens if len(t) > 1]


def _sentence_content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _tokenize_simple(text)
        if len(token) > 3 and token not in GENERIC_QUERY_TOKENS and token not in KB_NOISE_TOKENS
    }


def _sentence_content_overlap(left: str, right: str) -> float:
    left_tokens = _sentence_content_tokens(left)
    right_tokens = _sentence_content_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def _tokenize_kb(text: str) -> list[str]:
    return [t for t in _tokenize_simple(text) if t not in KB_NOISE_TOKENS]


def _is_nav_query(q: str) -> bool:
    lo = (q or "").lower()
    explicit_phrases = [
        "open ",
        "open that page",
        "open the page",
        "open this page",
        "go to ",
        "take me to ",
        "take us to ",
        "navigate to ",
        "show me ",
        "visit ",
        "direct me to ",
    ]
    return any(p in lo for p in explicit_phrases)


def _slug_tokens_from_url(url: str) -> list[str]:
    u = (url or "").lower()
    slug = u.split("://")[-1]
    slug = slug.split("?", 1)[0].split("#", 1)[0]
    slug = slug.split("/", 1)[-1]
    parts = re.split(r"[-_/]+", slug)
    toks = []
    for p in parts:
        p = re.sub(r"[^a-z0-9]+", "", p)
        if p and len(p) > 1:
            toks.append(p)
    return toks


def _score_overlap(qt: set[str], tt: set[str]) -> float:
    if not qt or not tt:
        return 0.0
    inter = qt.intersection(tt)
    return len(inter) / max(1, min(len(qt), len(tt)))


class KBService:
    def __init__(self, path: str):
        self.path = path
        self.entries: list[dict] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            self.entries = []
            return
        with open(self.path, "r", encoding="utf-8") as f:
            self.entries = json.load(f)

    def match_with_score(self, query: str, min_score: float = 0.22) -> Tuple[Optional[dict], float]:
        if not self.entries:
            return None, 0.0
        q = _semantic_query_core(query)
        qt = set(_tokenize_kb(q))
        if not qt:
            return None, 0.0
        best: Optional[dict] = None
        best_score = 0.0
        for e in self.entries:
            tags = e.get("tags") or []
            tag_text = " ".join(tags)
            tt = set(_tokenize_kb(tag_text))
            if not tt:
                continue
            sc = _score_overlap(qt, tt)
            title = (e.get("title") or "").lower()
            if title:
                sc += 0.05 * _score_overlap(qt, set(_tokenize_kb(title)))
            if sc > best_score:
                best_score = sc
                best = e
        if best and best_score >= min_score:
            return best, best_score
        return None, best_score

    def match(self, query: str, min_score: float = 0.22) -> Optional[dict]:
        best, _ = self.match_with_score(query=query, min_score=min_score)
        return best


class ChatbotService:
    def __init__(self):
        # Service setup that wires the models, indices, and page maps together once at startup.
        self.retriever = TfidfRetriever(
            index_dir=os.path.join("backend", "index"),
            chunks_path=os.path.join("data", "chunks", "chunks.jsonl"),
        )
        try:
            self.retriever.load()
        except Exception:
            pass

        self.intent = IntentService()
        try:
            self.intent.load()
        except Exception:
            pass

        self.kb = KBService(os.path.join("data", "kb", "kb.json"))
        self.page_summary = PageSummaryService(os.path.join("data", "pages_text"))
        self.kb_intent_guard_ids = {"policies", "newsletters", "photo_gallery"}

        # URL map that keeps the fixed replies and navigation rules consistent.
        self.page_urls = {
            "home": "https://www.brightfutures4all.com/",
            "reports_case_study": "https://www.brightfutures4all.com/",
            "register": "https://app.teachngo.com/leads/add/11035",
            "team": "https://www.brightfutures4all.com/copy-of-meet-our-team-trustees",
            "newsletter": "https://www.brightfutures4all.com/blog",
            "photo_gallery": "https://www.brightfutures4all.com/general-clean",
            "policies": "https://www.brightfutures4all.com/general-5",
            "volunteers_vacancies": "https://www.brightfutures4all.com/projects-3",
            "after_school_activities": "https://www.brightfutures4all.com/copy-of-send-after-school-club",
            "holiday_clubs": "https://www.brightfutures4all.com/holiday-clubs",
            "homeschooling": "https://www.brightfutures4all.com/copy-of-homeschooling-1",
            "curriculum": "https://www.brightfutures4all.com/copy-of-homeschooling-1",
            "tutors_mentors_therapists": "https://www.brightfutures4all.com/copy-of-tutoring-1",
            "testimonials": "https://www.brightfutures4all.com/copy-of-tutoring-1",
            "youth_leadership": "https://www.brightfutures4all.com/youth-leadership",
            "talks_workshops": "https://www.brightfutures4all.com/copy-of-after-school-activities",
            "donate": "https://www.brightfutures4all.com/donate",
            "contact": "https://www.brightfutures4all.com/contact-10",
            "about": "https://www.brightfutures4all.com/copy-of-about-us",
            "send_support": "https://www.brightfutures4all.com/copy-of-send-after-school-club",
            "tutoring": "https://www.brightfutures4all.com/copy-of-tutoring-1",
        }

        self.nav_targets = [
            {
                "id": "home",
                "title": "Home",
                "url": self.page_urls["home"],
                "keywords": ["home", "homepage", "main page", "about", "about page"],
            },
            {
                "id": "reports_case_study",
                "title": "Reports/Case Study",
                "url": self.page_urls["reports_case_study"],
                "keywords": ["report", "reports", "case study", "case studies", "read our reports"],
            },
            {
                "id": "register",
                "title": "Register",
                "url": self.page_urls["register"],
                "keywords": ["register", "register interest", "registration", "sign up", "enrol"],
            },
            {
                "id": "team",
                "title": "The Team",
                "url": self.page_urls["team"],
                "keywords": ["the team", "team", "trustees", "staff"],
            },
            {
                "id": "newsletter",
                "title": "Newsletter",
                "url": self.page_urls["newsletter"],
                "keywords": ["newsletter", "newsletters", "blog"],
            },
            {
                "id": "photo_gallery",
                "title": "Photo Gallery",
                "url": self.page_urls["photo_gallery"],
                "keywords": ["photo gallery", "gallery", "photos", "images"],
            },
            {
                "id": "policies",
                "title": "Policies",
                "url": self.page_urls["policies"],
                "keywords": ["policies", "policy", "safeguarding", "privacy"],
            },
            {
                "id": "volunteers_vacancies",
                "title": "Volunteers and Vacancies",
                "url": self.page_urls["volunteers_vacancies"],
                "keywords": ["volunteer", "volunteers", "volounteer", "vacancy", "vacancies", "jobs", "roles"],
            },
            {
                "id": "after_school_activities",
                "title": "After School Activities",
                "url": self.page_urls["after_school_activities"],
                "keywords": ["after school", "after school activities", "send club"],
            },
            {
                "id": "holiday_clubs",
                "title": "Holiday Clubs",
                "url": self.page_urls["holiday_clubs"],
                "keywords": ["holiday club", "holiday clubs", "school holiday"],
            },
            {
                "id": "homeschooling",
                "title": "Homeschooling",
                "url": self.page_urls["homeschooling"],
                "keywords": ["homeschooling", "home schooling", "home-schooling", "online homeschooling", "home education"],
            },
            {
                "id": "curriculum",
                "title": "Curriculum",
                "url": self.page_urls["curriculum"],
                "keywords": ["curriculum"],
            },
            {
                "id": "tutors_mentors_therapists",
                "title": "Tutors Mentors and Therapists",
                "url": self.page_urls["tutors_mentors_therapists"],
                "keywords": ["tutors", "mentors", "therapists", "tutoring", "therapy", "mentoring"],
            },
            {
                "id": "testimonials",
                "title": "Testimonials",
                "url": self.page_urls["testimonials"],
                "keywords": ["testimonials", "feedback", "reviews"],
            },
            {
                "id": "youth_leadership",
                "title": "Youth Development Leadership and Advocacy",
                "url": self.page_urls["youth_leadership"],
                "keywords": ["youth development", "leadership", "advocacy", "youth leadership"],
            },
            {
                "id": "talks_workshops",
                "title": "Talks and Workshops",
                "url": self.page_urls["talks_workshops"],
                "keywords": ["talks", "workshops", "talks and workshops", "keynote"],
            },
            {
                "id": "donate",
                "title": "Donate",
                "url": self.page_urls["donate"],
                "keywords": ["donate", "donation", "donate page"],
            },
            {
                "id": "contact",
                "title": "Contact",
                "url": self.page_urls["contact"],
                "keywords": [
                    "contact", "contact page"
                ],
            },

            {
                "id": "about",
                "title": "About BF4A",
                "url": self.page_urls["about"],
                "keywords": ["about", "about bf4a", "about bright futures 4 all", "mission", "who are you", "what is bf4a", "history", "background"],
            }
        ]

        self.intent_to_urls = {
            "REGISTER_CALL": ["contact-10", "register", "teachngo.com/leads/add/11035"],
            "REGISTER_FORM": ["teachngo.com/leads/add/11035", "register"],
            "CONTACT_INFO": ["contact-10", "contact"],
            "DONATE": ["donate"],
            "NEWSLETTERS": ["blog", "newsletter"],
            "REPORT_CASE_STUDY": ["home", "report", "case-study"],
            "VOLUNTEER_VACANCIES": ["projects-3", "vacanc", "volunteer"],
            "THE_TEAM": ["copy-of-meet-our-team-trustees", "team", "trustee"],
            "PHOTO_GALLERY": ["general-clean", "gallery", "photo"],
            "POLICIES": ["general-5", "policy", "privacy", "safeguard"],
            "CURRICULUM": ["copy-of-homeschooling-1", "curriculum", "homeschooling"],
            "ABOUT_BF4A": ["copy-of-about-us", "about-us", "about"],
            "HISTORY_BACKGROUND": ["copy-of-about-us", "about"],
        }

        self.topic_to_allowed = {
            "donate": ["donate"],
            "contact": ["contact"],
            "holiday_clubs": ["holiday-clubs", "holiday"],
            "send_support": ["send", "copy-of-send-after-school-club"],
            "tutoring": ["tutor", "tutoring", "mentor", "therap", "copy-of-tutoring-1"],
            "about": ["copy-of-about-us", "/about-us", "/about"],
            "testimonials": ["testimonial", "tutoring", "copy-of-tutoring-1"],
            "curriculum": ["homeschool", "curriculum"],
            "policies": ["general-5", "policy", "privacy", "safeguard"],
            "history_background": ["copy-of-about-us", "about"],
        }

        self.session_state: SessionStateMap = {}

    def _build_clarification_options(self) -> List[ClarificationOption]:
        # Clarification options that give the user a small set of safe topic shortcuts.
        return [
            ClarificationOption(label="About BF4A", payload={"topic": "about"}),
            ClarificationOption(label="Register", payload={"topic": "register"}),
            ClarificationOption(label="Programmes & Activities", payload={"topic": "programmes_activities"}),
            ClarificationOption(label="Contact", payload={"topic": "contact"}),
            ClarificationOption(label="Donate", payload={"topic": "donate"}),
            ClarificationOption(label="Quiz me", payload={"topic": "quiz"}),
            ClarificationOption(label="Volunteer & Vacancies", payload={"topic": "volunteer_vacancies"}),
            ClarificationOption(label="Therapy & Mentors", payload={"topic": "therapy_mentors"}),
            ClarificationOption(label="The Team", payload={"topic": "team"}),
            ClarificationOption(label="Policies", payload={"topic": "policies"}),
            ClarificationOption(label="Photo Gallery", payload={"topic": "photo_gallery"}),
            ClarificationOption(label="Newsletters", payload={"topic": "newsletters"}),
        ]

    def _response(
        self,
        session_id: str,
        last_query: str,
        reply: str,
        citations: Optional[List[Citation]] = None,
        actions: Optional[List[Action]] = None,
        quick: Optional[List[str]] = None,
        confidence: Optional[float] = None,
        needs_clarification: bool = False,
        clarification_question: str = "",
        clarification_options: Optional[List[ClarificationOption]] = None,
        verification_status: Optional[str] = None,
        evidence_score: Optional[float] = None,
        source: str = "fixed",
        topic: str = "",
        awaiting_open_confirmation: bool = False,
        awaiting_retrieval_confirmation: bool = False,
        pending_open_payload: Optional[dict] = None,
        pending_open_title: str = "",
        pending_open_url: str = "",
        quiz_state: Optional[dict] = None,
    ) -> ResponsePayload:
        # Response builder that packages the final reply together with any follow-up state.
        citations = citations or []
        actions = actions or []
        quick = quick or []
        clarification_options = clarification_options or []
        confidence_val = float(confidence) if confidence is not None else 0.0

        reply_text = _ensure_finished_reply(reply)
        pending_payload = dict(pending_open_payload or {}) if awaiting_open_confirmation else {}
        if awaiting_open_confirmation:
            quick = ["Yes", "No"]
        elif awaiting_retrieval_confirmation:
            # Retrieval follow-up that always checks whether the returned snippet was actually useful.
            if "does this answer your question" not in reply_text.lower():
                reply_text = f"{reply_text} Does this answer your question?"
            quick = ["Yes", "No"]

        # Session state that stores whatever the next turn needs, such as quiz progress or a pending page open.
        self.session_state[session_id] = {
            "awaiting_open_confirmation": awaiting_open_confirmation,
            "awaiting_retrieval_confirmation": awaiting_retrieval_confirmation,
            "pending_open_payload": pending_payload,
            "pending_open_title": pending_open_title if awaiting_open_confirmation else "",
            "pending_open_url": pending_open_url if awaiting_open_confirmation else "",
            "last_query": last_query,
            "last_topic": topic,
            "last_source": source,
            "quiz_state": dict(quiz_state or {}),
        }

        return (
            reply_text,
            citations,
            actions,
            quick,
            confidence_val,
            needs_clarification,
            clarification_question or "",
            clarification_options,
            verification_status,
            evidence_score,
        )

    def _clarification_response(
        self,
        session_id: str,
        last_query: str,
        results: list[dict],
        best_score: float,
        reply: str,
    ) -> ResponsePayload:
        # Clarification path that falls back to guided topic choices when the evidence is weak.
        clarification_options = self._build_clarification_options()

        return self._response(
            session_id=session_id,
            last_query=last_query,
            reply=CLARIFICATION_REPLY,
            citations=[],
            actions=[],
            quick=[],
            confidence=best_score,
            needs_clarification=True,
            clarification_question="Which area do you mean?",
            clarification_options=clarification_options,
            verification_status="weak_evidence",
            evidence_score=best_score,
            source="clarification",
        )

    def _topic_from_text(self, text: str) -> Optional[str]:
        # Topic guesser that helps retrieval and KB fallback stay in the right area.
        q = (text or "").lower()
        topic_keywords = {
            "donate": ["donate", "donation", "gift aid", "giftaid"],
            "contact": ["contact", "email", "phone", "address", "opening hours"],
            "holiday_clubs": ["holiday", "holiday clubs", "fsm", "free school meals"],
            "send_support": ["send", "neuro", "additional needs"],
            "tutoring": ["tutor", "tutoring", "mentor", "mentoring", "therapy", "therapist", "counselling"],
            "about": ["about", "mission", "history", "purpose", "aim", "vision", "what is bf4a", "bright futures 4 all"],
            "testimonials": ["testimonial", "feedback", "reviews"],
            "curriculum": ["curriculum", "home schooling", "homeschooling"],
            "policies": ["policy", "policies", "privacy", "safeguard"],
            "history_background": ["history", "background", "founded", "started", "began", "origin", "when did", "how did"],
        }
        for topic, kws in topic_keywords.items():
            if any(k in q for k in kws):
                return topic
        return None

    def _forced_intent_label(self, raw_text: str, normalised_text: str) -> Optional[str]:
        # Intent overrides that keep a few important queries stable even if the classifier is unsure.
        raw = (raw_text or "").lower().strip()
        norm = (normalised_text or "").lower().strip()
        if not raw and not norm:
            return None

        chatbot_identity_queries = {
            "who are you",
            "what are you",
            "what is this chatbot",
            "what is your name",
            "who is this chatbot",
            "tell me about this chatbot",
            "what can this chatbot help with",
            "how does this chatbot work",
        }
        semantic_core = _semantic_query_core(raw_text).lower().strip()
        if raw.rstrip("?.! ") in chatbot_identity_queries or semantic_core in chatbot_identity_queries:
            return "ABOUT_CHATBOT"

        bf4a_terms = [
            "bf4a",
            "bright futures 4 all",
            "bright futures for all",
        ]
        about_triggers = [
            "about",
            "what is",
            "who is",
            "tell me about",
            "explain",
            "what does",
        ]
        blockers = [
            "address",
            "email",
            "phone",
            "contact",
            "opening hours",
            "history",
            "background",
            "founded",
            "founded by",
            "started",
            "created",
            "when was",
            "who founded",
            "who started",
            "who created",
        ]

        haystack = f"{raw} {norm}".strip()
        if not any(term in haystack for term in bf4a_terms):
            return None
        if any(blocker in haystack for blocker in blockers):
            return None

        core = semantic_core
        if core in {"bright futures 4 all", "bf4a", "bright futures for all"}:
            return "ABOUT_BF4A"
        if any(trigger in raw for trigger in about_triggers):
            return "ABOUT_BF4A"
        return None

    def _kb_guard_topic(self, query: str) -> str:
        # Guard check that protects a few curated KB topics from being hijacked by the wrong intent.
        kb_hit, kb_score = self.kb.match_with_score(query, min_score=0.24)
        if not kb_hit or kb_score < 0.24:
            return ""
        kb_id = str(kb_hit.get("id") or "")
        if kb_id in self.kb_intent_guard_ids:
            return kb_id
        return ""

    def _is_confirmation_yes(self, text: str) -> bool:
        # Confirmation check for positive follow-up replies such as page opening or retrieval confirmation.
        return (text or "").strip().lower() in CONFIRM_YES

    def _is_confirmation_no(self, text: str) -> bool:
        # Confirmation check for negative follow-up replies.
        return (text or "").strip().lower() in CONFIRM_NO

    def _should_confirm_open(self, reply: str, url: str) -> bool:
        # Reply check that only enables the open-page confirmation state when a real URL is available.
        if not url:
            return False
        reply_text = (reply or "").lower()
        return "would you like me to open" in reply_text

    def _is_page_summary_query(self, text: str) -> bool:
        # Summary query check that stays simple so ordinary questions do not slip into the summary flow.
        q = (text or "").lower()
        has_summary_word = any(
            phrase in q for phrase in [
                "summarise",
                "summarize",
                "summary",
            ]
        )
        return has_summary_word and "page" in q

    def _is_quiz_request(self, text: str) -> bool:
        # Quiz matcher that accepts a wider range of casual phrasings.
        q = re.sub(r"[^a-z0-9']+", " ", (text or "").strip().lower())
        q = re.sub(r"\s+", " ", q).strip()
        if not q or q in {"continue quiz", "finish quiz", "stop quiz", "end quiz", "quit quiz", "cancel quiz"}:
            return False

        quiz_phrases = {
            "quiz",
            "quizes",
            "quizzes",
            "quiz me",
            "give me a quiz",
            "start a quiz",
            "start quiz",
            "bf4a quiz",
            "test me",
            "i want a quiz",
            "i want to do a quiz",
            "i want to play a quiz",
            "i want to answer some quiz questions",
            "let's do a quiz",
            "lets do a quiz",
            "let's play a quiz",
            "lets play a quiz",
            "give me quiz questions",
            "ask me quiz questions",
            "test my knowledge",
            "give me some quizzes",
            "give me some quizes",
            "can we do a quiz",
            "give me a quick quiz",
            "i'd like to try a quiz",
            "id like to try a quiz",
            "ask me some questions",
            "quiz time",
            "can you quiz me",
            "can you give me a quiz",
            "can you ask me quiz questions",
        }
        if q in quiz_phrases:
            return True

        quiz_patterns = [
            r"\bquiz me\b",
            r"\b(?:give|start|do|try)\s+(?:me\s+)?(?:a\s+|some\s+|quick\s+)?quiz(?:es|zes)?\b",
            r"\b(?:i|we)\s+(?:want|would like|d like|want to)\s+(?:to\s+do\s+|to\s+try\s+|to\s+play\s+|)\s+(?:a\s+|some\s+)?quiz(?:es|zes)?\b",
            r"\blet'?s\s+play\s+a\s+quiz(?:es|zes)?\b",
            r"\bcan you\s+quiz me\b",
            r"\bcan you\s+give me\s+(?:a\s+|some\s+|quick\s+)?quiz(?:es|zes)?\b",
            r"\bcan you\s+ask me\s+(?:some\s+)?quiz questions\b",
            r"\bask me\s+(?:some\s+)?quiz questions\b",
            r"\bgive me\s+(?:some\s+)?quiz questions\b",
            r"\bquiz time\b",
            r"\btest my knowledge\b",
            r"\bask me some questions\b",
        ]
        return any(re.search(pattern, q) for pattern in quiz_patterns)

    def _quiz_question_bank(self) -> List[dict]:
        # Quiz question bank that stores the fixed BF4A multiple-choice questions.
        return [
            {
                "id": 1,
                "prompt": "What does BF4A stand for?",
                "options": ["A. Bright Future Academics", "B. Bright Futures 4 All", "C. Better Futures Alliance", "D. British Families 4 Achievement"],
                "answer_key": "B",
            },
            {
                "id": 2,
                "prompt": "Who founded Bright Futures 4 All?",
                "options": ["A. Karen Bryson", "B. David Cameron", "C. Sarah Williams", "D. Andrew Law"],
                "answer_key": "A",
            },
            {
                "id": 3,
                "prompt": "In which year was Bright Futures 4 All originally founded?",
                "options": ["A. 1999", "B. 2005", "C. 2007", "D. 2015"],
                "answer_key": "C",
            },
            {
                "id": 4,
                "prompt": "Where was Bright Futures 4 All originally founded?",
                "options": ["A. Croydon", "B. Islington", "C. Camden", "D. Hackney"],
                "answer_key": "B",
            },
            {
                "id": 5,
                "prompt": "Which group does BF4A primarily support?",
                "options": ["A. Only university students", "B. Only teachers", "C. Neurodiverse and disadvantaged children", "D. Professional athletes"],
                "answer_key": "C",
            },
            {
                "id": 6,
                "prompt": "What type of organisation is Bright Futures 4 All?",
                "options": ["A. Private company", "B. Community Interest Company (CIC)", "C. Government agency", "D. University department"],
                "answer_key": "B",
            },
            {
                "id": 7,
                "prompt": "Which of the following is one of BF4A's services?",
                "options": ["A. Online gaming tournaments", "B. After-school activities", "C. Airline training", "D. Driving lessons"],
                "answer_key": "B",
            },
            {
                "id": 8,
                "prompt": "Which programme focuses on mentoring and leadership development?",
                "options": ["A. Youth Development, Leadership & Advocacy", "B. Online Gaming Club", "C. Business Internship Scheme", "D. City Leadership Course"],
                "answer_key": "A",
            },
            {
                "id": 9,
                "prompt": "What type of learning support does BF4A offer to students?",
                "options": ["A. Military training", "B. Tutoring and mentoring", "C. Flight school", "D. Engineering apprenticeships"],
                "answer_key": "B",
            },
            {
                "id": 10,
                "prompt": "Which club is described as Croydon's first neuro-diverse club?",
                "options": ["A. Future Leaders Club", "B. Odd Girl In", "C. Tech Skills Club", "D. STEM Champions"],
                "answer_key": "B",
            },
            {
                "id": 11,
                "prompt": "What type of activities are offered in BF4A holiday clubs?",
                "options": ["A. Only exam preparation", "B. Sports, science, arts and crafts", "C. Only computer programming", "D. Only language lessons"],
                "answer_key": "B",
            },
            {
                "id": 12,
                "prompt": "What is the aim of the Homeschooling Hub?",
                "options": ["A. Training teachers", "B. Supporting home-schooled and school-refusing children", "C. Teaching university students", "D. Providing office space"],
                "answer_key": "B",
            },
            {
                "id": 13,
                "prompt": "Which programme combines sports and maths learning?",
                "options": ["A. Smart Scholars", "B. Grass2Grades", "C. Future Leaders", "D. Maths Champions"],
                "answer_key": "B",
            },
            {
                "id": 14,
                "prompt": "What percentage of UK school pupils have Special Educational Needs or Disabilities (SEND)?",
                "options": ["A. 5%", "B. 10%", "C. 18.4%", "D. 40%"],
                "answer_key": "C",
            },
            {
                "id": 15,
                "prompt": "What is one goal of Bright Futures 4 All?",
                "options": ["A. Promote social mobility through education", "B. Train athletes for the Olympics", "C. Build housing developments", "D. Provide bank loans"],
                "answer_key": "A",
            },
            {
                "id": 16,
                "prompt": "Which type of club is Odd Girl In?",
                "options": ["A. Coding club", "B. Neuro-diverse girls club", "C. Debate club", "D. Science research club"],
                "answer_key": "B",
            },
            {
                "id": 17,
                "prompt": "BF4A programmes aim to help young people improve their:",
                "options": ["A. Driving ability", "B. Confidence, wellbeing, and educational outcomes", "C. Investment skills", "D. Political leadership"],
                "answer_key": "B",
            },
            {
                "id": 18,
                "prompt": "Where is BF4A mainly operating now?",
                "options": ["A. Manchester", "B. Croydon and South London boroughs", "C. Liverpool", "D. Birmingham"],
                "answer_key": "B",
            },
            {
                "id": 19,
                "prompt": "Bright Futures 4 All focuses on creating opportunities for:",
                "options": ["A. Only university graduates", "B. Children and young people in the community", "C. Only teachers", "D. Only businesses"],
                "answer_key": "B",
            },
            {
                "id": 20,
                "prompt": "What is the overall mission of Bright Futures 4 All?",
                "options": ["A. Promote international tourism", "B. Support children and families through inclusive education and wellbeing services", "C. Provide business investment training", "D. Train government officials"],
                "answer_key": "B",
            },
        ]

    def _sample_quiz_questions(self, asked_ids: Optional[List[int]] = None, count: int = 6) -> List[dict]:
        # Sampling step that avoids repeating quiz questions that were already used in the same session.
        asked = set(asked_ids or [])
        available = [question for question in self._quiz_question_bank() if question["id"] not in asked]
        if not available:
            return []
        if len(available) <= count:
            return available
        return random.sample(available, count)

    def _build_quiz_questions(self, asked_ids: Optional[List[int]] = None, count: int = 6) -> List[dict]:
        # Small wrapper that keeps the quiz question selection logic in one place.
        return self._sample_quiz_questions(asked_ids=asked_ids, count=count)

    def _quiz_correct_option(self, question: dict) -> str:
        # Answer lookup that expands the stored answer key back into the full option text.
        answer_key = (question.get("answer_key") or "").upper()
        for option in question.get("options") or []:
            if option.upper().startswith(f"{answer_key}."):
                return option
        return ""

    def _active_quiz_state(self, session_id: str) -> dict:
        # State lookup that returns the active quiz state for the current session when one exists.
        state = self.session_state.get(session_id, {})
        quiz_state = state.get("quiz_state") or {}
        if quiz_state.get("active"):
            return quiz_state
        return {}

    def _quiz_question_response(
        self,
        session_id: str,
        last_query: str,
        quiz_state: dict,
        intro: str = "",
    ) -> ResponsePayload:
        # Question response that shows the current quiz item and the answer buttons.
        index = int(quiz_state.get("current_index", 0))
        questions = quiz_state.get("questions") or []
        if index >= len(questions):
            return self._quiz_continue_response(session_id, last_query, quiz_state)

        question = questions[index]
        prefix = f"{intro}\n\n" if intro else ""
        reply = f"{prefix}Question {index + 1} of {len(questions)}: {question['prompt']}\nYou can stop the quiz by typing \"stop quiz\"."
        return self._response(
            session_id=session_id,
            last_query=last_query,
            reply=reply,
            actions=[],
            quick=list(question.get("options") or []),
            confidence=1.0,
            needs_clarification=False,
            clarification_question="",
            clarification_options=[],
            verification_status="verified",
            evidence_score=1.0,
            source="fixed",
            topic="quiz",
            quiz_state=quiz_state,
        )

    def _quiz_continue_response(
        self,
        session_id: str,
        last_query: str,
        quiz_state: dict,
        intro: str = "",
    ) -> ResponsePayload:
        # Continue prompt that appears after a batch of six quiz questions has been completed.
        total_answered = int(quiz_state.get("total_answered", 0))
        score = int(quiz_state.get("score", 0))
        remaining = len(self._sample_quiz_questions(asked_ids=quiz_state.get("asked_ids") or [], count=20))
        if remaining <= 0:
            return self._quiz_finish_response(session_id, last_query, quiz_state)
        prefix = f"{intro}\n\n" if intro else ""
        reply = f"{prefix}You have answered {total_answered} questions and scored {score} so far. Would you like to continue the quiz or finish it?"
        next_state = {
            **quiz_state,
            "awaiting_continue": True,
        }
        return self._response(
            session_id=session_id,
            last_query=last_query,
            reply=reply,
            actions=[],
            quick=["Continue quiz", "Finish quiz"],
            confidence=1.0,
            needs_clarification=False,
            clarification_question="",
            clarification_options=[],
            verification_status="verified",
            evidence_score=1.0,
            source="fixed",
            topic="quiz",
            quiz_state=next_state,
        )

    def _quiz_finish_response(
        self,
        session_id: str,
        last_query: str,
        quiz_state: dict,
    ) -> ResponsePayload:
        # Finish response that reports the final score and resets the quiz state.
        total_answered = int(quiz_state.get("total_answered", 0))
        score = int(quiz_state.get("score", 0))
        reply = f"You scored {score} out of {total_answered} on the BF4A quiz. If you want, I can quiz you again or help with another question."
        return self._response(
            session_id=session_id,
            last_query=last_query,
            reply=reply,
            actions=[],
            quick=["Quiz me again"],
            confidence=1.0,
            needs_clarification=False,
            clarification_question="",
            clarification_options=[],
            verification_status="verified",
            evidence_score=1.0,
            source="fixed",
            topic="quiz",
            quiz_state={},
        )

    def _start_quiz_response(
        self,
        session_id: str,
        msg_norm: str,
    ) -> ResponsePayload:
        # Start response that creates a fresh batch of quiz questions for the current session.
        questions = self._build_quiz_questions()
        if not questions:
            return self._response(
                session_id=session_id,
                last_query=msg_norm,
                reply="I do not have enough BF4A FAQ information loaded to start a quiz yet.",
                actions=[],
                quick=[],
                confidence=0.0,
                needs_clarification=False,
                clarification_question="",
                clarification_options=[],
                verification_status="weak_evidence",
                evidence_score=0.0,
                source="fixed",
                topic="quiz",
                quiz_state={},
            )

        quiz_state = {
            "active": True,
            "questions": questions,
            "current_index": 0,
            "score": 0,
            "total_answered": 0,
            "asked_ids": [question["id"] for question in questions],
            "awaiting_continue": False,
        }
        intro = "Here is a short BF4A quiz."
        return self._quiz_question_response(session_id, msg_norm, quiz_state, intro=intro)

    def _handle_quiz_turn(
        self,
        session_id: str,
        msg_norm: str,
        quiz_state: dict,
    ) -> ResponsePayload:
        # Quiz turn handler that processes stop commands, answers, and continue or finish choices.
        if msg_norm.lower() in QUIZ_EXIT:
            return self._response(
                session_id=session_id,
                last_query=msg_norm,
                reply="Okay. I have ended the quiz. What would you like to ask next?",
                actions=[],
                quick=[],
                confidence=1.0,
                needs_clarification=False,
                clarification_question="",
                clarification_options=[],
                verification_status="verified",
                evidence_score=1.0,
                source="fixed",
                topic="quiz",
                quiz_state={},
            )

        if quiz_state.get("awaiting_continue"):
            # Continue state that waits for the user to choose whether to extend the quiz or stop.
            if msg_norm.lower() == "continue quiz":
                next_questions = self._build_quiz_questions(asked_ids=quiz_state.get("asked_ids") or [])
                if not next_questions:
                    return self._quiz_finish_response(session_id, msg_norm, quiz_state)
                next_state = {
                    **quiz_state,
                    "active": True,
                    "questions": next_questions,
                    "current_index": 0,
                    "asked_ids": list(quiz_state.get("asked_ids") or []) + [question["id"] for question in next_questions],
                    "awaiting_continue": False,
                }
                return self._quiz_question_response(session_id, msg_norm, next_state, intro="Okay, here are 6 more questions.")
            if msg_norm.lower() == "finish quiz":
                return self._quiz_finish_response(session_id, msg_norm, quiz_state)
            return self._response(
                session_id=session_id,
                last_query=msg_norm,
                reply="Please choose whether you want to continue the quiz or finish it.",
                actions=[],
                quick=["Continue quiz", "Finish quiz"],
                confidence=1.0,
                needs_clarification=False,
                clarification_question="",
                clarification_options=[],
                verification_status="verified",
                evidence_score=1.0,
                source="fixed",
                topic="quiz",
                quiz_state=quiz_state,
            )

        questions = quiz_state.get("questions") or []
        index = int(quiz_state.get("current_index", 0))
        if index >= len(questions):
            return self._quiz_continue_response(session_id, msg_norm, quiz_state)

        question = questions[index]
        options = question.get("options") or []
        correct = self._quiz_correct_option(question)
        correct_key = (question.get("answer_key") or "").upper()

        # Answer parsing that accepts either the button text or just the A/B/C/D letter.
        matched_option = ""
        for option in options:
            if _normalise_query(option).lower() == msg_norm.lower():
                matched_option = option
                break
        if not matched_option and msg_norm.strip().upper() in {"A", "B", "C", "D"}:
            letter = msg_norm.strip().upper()
            matched_option = next((option for option in options if option.upper().startswith(f"{letter}.")), "")
        if not matched_option:
            # Validation branch that asks for one of the listed options when the answer cannot be matched.
            return self._response(
                session_id=session_id,
                last_query=msg_norm,
                reply="Please choose one of the quiz answers below or type stop quiz.",
                actions=[],
                quick=list(options),
                confidence=1.0,
                needs_clarification=False,
                clarification_question="",
                clarification_options=[],
                verification_status="verified",
                evidence_score=1.0,
                source="fixed",
                topic="quiz",
                quiz_state=quiz_state,
            )

        is_correct = matched_option.upper().startswith(f"{correct_key}.")
        next_index = index + 1
        next_score = int(quiz_state.get("score", 0)) + (1 if is_correct else 0)
        total_answered = int(quiz_state.get("total_answered", 0)) + 1
        # Feedback text that is sent before the next question or the continue prompt.
        feedback = "Correct." if is_correct else f"Not quite. The correct answer was {correct}."

        if next_index >= len(questions):
            next_quiz_state = {
                **quiz_state,
                "active": True,
                "questions": questions,
                "current_index": next_index,
                "score": next_score,
                "total_answered": total_answered,
                "awaiting_continue": False,
            }
            return self._quiz_continue_response(session_id, msg_norm, next_quiz_state, intro=feedback)

        next_quiz_state = {
            "active": True,
            "questions": questions,
            "current_index": next_index,
            "score": next_score,
            "total_answered": total_answered,
            "asked_ids": list(quiz_state.get("asked_ids") or []),
            "awaiting_continue": False,
        }
        return self._quiz_question_response(session_id, msg_norm, next_quiz_state, intro=feedback)

    def _summary_page_match(self, query: str) -> Optional[dict]:
        # Summary page matcher that first checks known aliases before using overlap scoring.
        q = _normalise_query(query)
        q = re.sub(r"\bsummaris[sz]e\b", " ", q, flags=re.IGNORECASE)
        q = re.sub(r"\bsummary\s+of\b", " ", q, flags=re.IGNORECASE)
        q = re.sub(r"\bsummary\b", " ", q, flags=re.IGNORECASE)
        q = re.sub(r"\bthe\b", " ", q, flags=re.IGNORECASE)
        q = re.sub(r"\bthis\b", " ", q, flags=re.IGNORECASE)
        q = re.sub(r"\bthat\b", " ", q, flags=re.IGNORECASE)
        q = re.sub(r"\bpage\b", " ", q, flags=re.IGNORECASE)
        q = re.sub(r"\s+", " ", q).strip().lower()
        if not q:
            return None

        q_compact = _normalise_summary_phrase(q)

        summary_aliases = [
            ("homeschooling", [
                "homeschooling",
                "home schooling",
                "home-schooling",
                "online homeschooling",
                "homeschooling hub",
                "home schooling hub",
                "home education",
            ]),
            ("volunteers_vacancies", [
                "volunteers and vacancies",
                "volunteer and vacancies",
                "volunteers vacancies",
                "volunteer vacancies",
                "volounteer and vacancies",
                "volounteers and vacancies",
                "vacancies",
                "volunteers",
                "volunteer roles",
            ]),
            ("tutors_mentors_therapists", [
                "tutors mentors and therapists",
                "tutors mentors therapists",
                "tutors and mentors",
                "mentors and therapists",
                "tutors mentors",
                "tutors therapists",
                "tutors mentors and psychotherapists",
                "tutors mentors therapists and counselling",
                "therapy mentors",
                "therapists",
            ]),
        ]
        for target_id, phrases in summary_aliases:
            if any(_normalise_summary_phrase(phrase) in q_compact for phrase in phrases):
                return next((target for target in self.nav_targets if target.get("id") == target_id), None)

        if all(token in q_compact.split() for token in ["tutors", "mentors", "therapists"]):
            return next((target for target in self.nav_targets if target.get("id") == "tutors_mentors_therapists"), None)
        if "vacancies" in q_compact and any(token in q_compact for token in ["volunteer", "volunteers", "volounteer", "volounteers"]):
            return next((target for target in self.nav_targets if target.get("id") == "volunteers_vacancies"), None)
        if "home" in q_compact and "schooling" in q_compact:
            return next((target for target in self.nav_targets if target.get("id") == "homeschooling"), None)
        if q_compact in {"home", "homepage", "home page", "main page"}:
            return next((target for target in self.nav_targets if target.get("id") == "home"), None)

        qt = set(_tokenize_simple(q))
        best = None
        best_score = 0.0

        for target in self.nav_targets:
            keywords = set()
            for keyword in target.get("keywords") or []:
                keywords.update(_tokenize_simple(keyword))
            keywords.update(_slug_tokens_from_url(target.get("url") or ""))
            if not keywords:
                continue

            score = _score_overlap(qt, keywords)
            title_tokens = set(_tokenize_simple(target.get("title") or ""))
            score += 0.35 * _score_overlap(qt, title_tokens)
            if target.get("id") and target["id"].replace("_", " ") in q:
                score += 0.20

            if score > best_score:
                best_score = score
                best = target

        if best and best_score >= 0.45:
            return best
        return None

    def _page_summary_response(
        self,
        session_id: str,
        msg_norm: str,
        target: dict,
    ) -> Optional[ResponsePayload]:
        # Summary rule that reuses the about page text when the user asks for the home page summary.
        summary_source_url = target.get("url") or ""
        if target.get("id") == "home":
            summary_source_url = self.page_urls["about"]

        summary = self.page_summary.summarize_url(summary_source_url)
        if not summary:
            return None

        title = target.get("title") or "Bright Futures 4 All"
        # Open payload that is stored in case the user says yes to opening the summarised page.
        payload = self._build_open_payload(target)
        if summary:
            summary = re.sub(r"^[A-Z][A-Z0-9]{1,}(?=\b)", lambda m: m.group(0).lower(), summary, count=1)
            if summary[:1].isalpha():
                summary = summary[:1].lower() + summary[1:]
        reply = f"Summarising, {summary}\nWould you like me to open that page for you?"

        return self._response(
            session_id=session_id,
            last_query=msg_norm,
            reply=reply,
            citations=[Citation(title=title, url=target["url"])],
            actions=[],
            quick=[],
            confidence=1.0,
            needs_clarification=False,
            clarification_question="",
            clarification_options=[],
            verification_status="verified",
            evidence_score=1.0,
            source="fixed",
            topic=target.get("id") or "",
            awaiting_open_confirmation=True,
            pending_open_payload=payload,
            pending_open_title=title,
            pending_open_url=target.get("url") or "",
        )

    def _build_open_payload(self, target: dict) -> dict:
        # Open payload builder that keeps the frontend page-opening action format consistent.
        return {"url": target["url"]}

    def _nav_match(self, query: str) -> Optional[dict]:
        # Navigation matcher that only runs for explicit open or go-to style queries.
        q = _normalise_query(query)
        qt = set(_tokenize_simple(q))
        ql = q.lower()

        has_nav_trigger = _is_nav_query(ql)
        if not has_nav_trigger:
            return None

        best: Optional[dict] = None
        best_score = 0.0

        for t in self.nav_targets:
            # Score made up of title, slug, and keyword overlap with a few exact keyword bonuses.
            keywords = t.get("keywords") or []
            title_toks = set(_tokenize_simple(t.get("title") or ""))
            slug_toks = set(_slug_tokens_from_url(t.get("url") or ""))
            kw_toks = set(_tokenize_simple(" ".join(keywords)))

            overlap = (
                0.45 * _score_overlap(qt, title_toks)
                + 0.35 * _score_overlap(qt, slug_toks)
                + 0.20 * _score_overlap(qt, kw_toks)
            )
            score = overlap
            keyword_hits = 0

            for kw in keywords:
                if kw in ql:
                    keyword_hits += 1
                    score += 0.9 if " " in kw else 0.75

            if has_nav_trigger:
                score += 0.08
            if keyword_hits > 0 and len(qt) <= 4:
                score += 0.12

            if score > best_score:
                best_score = score
                best = {
                    **t,
                    "score": score,
                    "keyword_hits": keyword_hits,
                }

        if not best:
            return None

        min_score = 0.75 if best.get("keyword_hits", 0) > 0 else 0.52
        # Threshold that is stricter when the match relies only on overlap rather than direct keyword hits.
        if best_score >= min_score:
            return best
        return None

    def _fixed_flow(self, query: str, intent_label: str, intent_conf: float) -> Optional[dict]:
        # Fixed-flow replies that are used when a curated answer is more reliable than retrieval.
        min_intent_conf = 0.25
        thresholds = getattr(self.intent, "thresholds", {}) or {}
        min_intent_conf = float(thresholds.get("global_confidence", min_intent_conf))
        if not intent_label or intent_label == "OTHER" or intent_conf < min_intent_conf:
            return None

        # Fixed response map that links a high-confidence intent to a curated reply and optional page.
        flows = {
            "REGISTER_CALL": {
                "reply": "Please call our office to enquire or register: 0207 062 7123 or 07835 878283",
                "url": "",
                "title": "",
                "quick": [],
            },
            "HISTORY_BACKGROUND": {
                "reply": (
                    "Bright Futures 4 All (BF4A) was founded in 2007 by Mrs. Karen Bryson as a not-for-profit, community-led initiative to support children, young people and families through inclusive education, mentoring and wellbeing programmes, and it has since grown to provide tutoring, SEND support, holiday clubs and youth development activities designed to improve confidence, wellbeing and opportunities in the local community."
                    "For the full background and story, you can read the About BF4A page.\n"
                    "Would you like me to open the About page for you?"
                ),
                "url": self.page_urls["about"],
                "title": "About - Bright Futures 4 All",
                "quick": [],
            },
            "ABOUT_BF4A": {
                "reply": (
                    "Bright Futures 4 All (BF4A) is a community-focused charity that supports children, young people of school age and their families through education and wellbeing programmes. "
                    "Our goal is to improve confidence, opportunities and support for families who need additional help in their communities.\n"
                    "Would you like me to open the About BF4A page for you?"
                ),
                "url": self.page_urls["about"],
                "title": "About - Bright Futures 4 All",
                "quick": [],
            },
            "PROGRAMMES": {
                "reply": (
                    "Bright Futures 4 All runs a range of excellent, free and affordable programmes and activities for children, young people of school age and their families who need additional support. "
                    "These include holiday clubs, the free SEND after-school club, tutoring and mentoring, youth leadership development, "
                    "as well as talks and workshops for schools and communities."
                ),
                "url": "",
                "title": "",
                "quick": [ "Holiday Clubs", "SEND After-School Club", "Tutoring", "Grass2Grades", "Workshops", "Therapy & Mentoring", "Youth Leadership Development",  "Homeschooling", "Odd Girl In"],
            },
            "REGISTER_FORM": {
                "reply": "Please complete our form to enquire or register your interest in any of our programs. Would you like me to open the form for you?",
                "url": self.page_urls["register"],
                "title": "Register - Bright Futures 4 All",
                "quick": [],
            },
            "ABOUT_CHATBOT": {
                "reply": (
                    "I’m the Bright Futures 4 All chatbot. I can help you find information about "
                    "the charity, including programmes, workshops, mentoring, tutoring, "
                    "SEND support, contact details, registration, policies, newsletters and more. I can also summarise and open pages or quiz you! "
                    "Example of questions you can ask me:"
                ),
                "url": "",
                "title": "",
                "quick": ["Open the donate page", "How do I register?", "What is your email address?", "What programmes do you have?"]
            },
            
            "HELLO": {
                "reply": "Hello. How can i help you?",
                "url": "",
                "title": "",
                "quick": ["Contact", "Donate", "About BF4A", "Register", "Programmes & Activities"],
            },
            "GOODBYE": {
                "reply": "If you have any other queries feel free to ask. Thank you, goodbye!",
                "url": "",
                "title": "",
                "quick": [],
            },
            "CONTACT_INFO": {
                "reply": "Here are the BF4A contact details:\n"
            "• Email: info@brightfutures4all.com\n"
            "• Phone: 0207 062 7123 / 07835 878283\n"
            "• Address: Heavers Farm Primary School, 58 Dinsdale Gardens, London SE25 6LT\n"
            "• Opening times: Mon–Sat, 9:30am–5:30pm\n"
            "Would you like me to open the contact form for you?",
                "url": self.page_urls["contact"],
                "title": "Contact - Bright Futures 4 All",
                "quick": [],
            },
            "DONATE": {
                "reply": "If you’d like to support BF4A, you can donate directly on our website.\n"
            "Would you like me to open that page for you?",
                "url": self.page_urls["donate"],
                "title": "Donate - Bright Futures 4 All",
                "quick": [],
            },
            "REPORT_CASE_STUDY": {
                "reply": "If you want to see more of BF4A’s progress, impact, reports and case studies, you can read about them on our website. For example, BF4A shares short impact stories like Cara’s, who built confidence in English over two years and improved her GCSE prediction from grade 3 to grade 5. \n"
            "Would you like me to open the reports/case studies section for you?",
                "url": self.page_urls["reports_case_study"],
                "title": "Reports and Case Study - Bright Futures 4 All",
                "quick": [],
            },
            "THERAPY_MENTORS": {
                "reply": (
                    "BF4A provides access to tutors, mentors and therapists who support children and young people with learning, emotional wellbeing and personal development. "
                    "This support can include mentoring, tutoring, counselling and specialist guidance designed to help build confidence, improve learning and support overall wellbeing.\n"
                    "Would you like me to open the Tutors, Mentors and Therapists page for more information?"
                ),
                "url": self.page_urls["tutors_mentors_therapists"],
                "title": "Tutors, Mentors and Therapists - Bright Futures 4 All",
                "quick": [],
            },
            "VOLUNTEER_VACANCIES": {
                "reply": "If you’d like to volunteer or you’re looking for roles with BF4A, the latest opportunities are listed on the Volunteers and Vacancies page.\n"
            "Would you like me to open that page for you?",
                "url": self.page_urls["volunteers_vacancies"],
                "title": "Volunteers and Vacancies - Bright Futures 4 All",
                "quick": [],
            },
            "THE_TEAM": {
                "reply": "You can meet the BF4A team and trustees on the Team page.\n"
            "Would you like me to open that page for you?",
                "url": self.page_urls["team"],
                "title": "The Team - Bright Futures 4 All",
                "quick": [],
            },
            "CURRICULUM": {
                "reply": "Our curriculum values every subject and challenges students to reach their full potential. Subjects taught include: English, mathematics, science, RE, arts, and social studies. For more details on our curriculum you can read about it on our curriculum page.\n"
            "Would you like me to open that page for you?",
                "url": self.page_urls["curriculum"],
                "title": "Curriculum - Bright Futures 4 All",
                "quick": [],
            },
        }

        return flows.get(intent_label)

    def _kb_fallback(
        self,
        session_id: str,
        msg_norm: str,
        best_score: float,
        topic: str,
    ) -> Optional[ResponsePayload]:
        # KB fallback that gives short curated answers for topics where precision matters more than recall.
        kb_hit, kb_score = self.kb.match_with_score(msg_norm, min_score=0.22)
        if not kb_hit or kb_score < 0.24:
            return None

        url = kb_hit.get("url") or ""
        title = kb_hit.get("title") or "Bright Futures 4 All"
        reply = kb_hit.get("answer") or "The most accurate details are on the linked page."
        citations = [Citation(title=title, url=url)] if url else []

        conf = max(float(kb_score), float(best_score))

        return self._response(
            session_id=session_id,
            last_query=msg_norm,
            reply=reply,
            citations=citations,
            actions=[],
            quick=[],
            confidence=conf,
            needs_clarification=False,
            clarification_question="",
            clarification_options=[],
            verification_status="verified",
            evidence_score=float(kb_score),
            source="kb",
            topic=topic,
        )

    def _run_retrieval(
        self,
        session_id: str,
        msg_norm: str,
        intent_label: str,
        forced_allowed: Optional[List[str]] = None,
        topic: str = "",
    ) -> ResponsePayload:
        # Retrieval path that handles open factual questions after the more explicit features are ruled out.
        allowed = forced_allowed if forced_allowed is not None else self.intent_to_urls.get(intent_label)
        retrieval_query = _semantic_query_core(msg_norm)

        try:
            if allowed:
                results = self.retriever.search_filtered(retrieval_query, allowed_url_substrings=allowed, top_k=8)
            else:
                results = self.retriever.search(retrieval_query, top_k=8)
        except FileNotFoundError:
            kb_result = self._kb_fallback(
                session_id=session_id,
                msg_norm=msg_norm,
                best_score=0.0,
                topic=topic,
            )
            if kb_result:
                return kb_result
            return self._response(
                session_id=session_id,
                last_query=msg_norm,
                reply="The search index has not been built yet.",
                confidence=0.0,
                needs_clarification=False,
                clarification_question="",
                clarification_options=[],
                verification_status="weak_evidence",
                evidence_score=0.0,
                source="retrieval",
                topic=topic,
            )

        results = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
        best = results[0] if results else None
        best_score = float(best.get("score", 0.0)) if best else 0.0

        LOW_CONF = 0.07
        AMBIG_DELTA = 0.012
        STRONG_TOP_SCORE = 0.14

        # Ambiguity check that backs off into KB or clarification when the retrieval signal is too weak.
        ambiguous = False
        if len(results) >= 2:
            delta = float(results[0].get("score", 0.0)) - float(results[1].get("score", 0.0))
            if delta < AMBIG_DELTA:
                ambiguous = True
        if best_score >= STRONG_TOP_SCORE:
            ambiguous = False

        if (not best) or (best_score < LOW_CONF) or ambiguous:
            kb_result = self._kb_fallback(
                session_id=session_id,
                msg_norm=msg_norm,
                best_score=best_score,
                topic=topic,
            )
            if kb_result:
                return kb_result
            return self._clarification_response(
                session_id=session_id,
                last_query=msg_norm,
                results=results,
                best_score=best_score,
                reply=CLARIFICATION_REPLY,
            )

        top_results = results[:5]
        snippet, sent_score, sent_citations = _best_sentences_across_results(
            self.retriever,
            retrieval_query,
            top_results,
            max_sentences=2,
        )

        evidence_score = float(sent_score)
        VERIFY_MIN_SENT_SCORE = 0.10

        if not snippet or evidence_score < VERIFY_MIN_SENT_SCORE:
            kb_result = self._kb_fallback(
                session_id=session_id,
                msg_norm=msg_norm,
                best_score=best_score,
                topic=topic,
            )
            if kb_result:
                return kb_result
            return self._clarification_response(
                session_id=session_id,
                last_query=msg_norm,
                results=results,
                best_score=best_score,
                reply=CLARIFICATION_REPLY,
            )

        citations = sent_citations if sent_citations else [
            Citation(title=best.get("title") or "Bright Futures 4 All", url=best.get("url") or "")
        ]
        citations = [c for c in citations if c.url]

        return self._response(
            session_id=session_id,
            last_query=msg_norm,
            reply=snippet,
            citations=citations,
            actions=[],
            quick=["Donate", "Contact", "Ask another question"],
            confidence=min(max(evidence_score, 0.0), 1.0),
            needs_clarification=False,
            clarification_question="",
            clarification_options=[],
            verification_status="verified",
            evidence_score=evidence_score,
            source="retrieval",
            topic=topic,
            awaiting_retrieval_confirmation=True,
        )

    def respond(
        self,
        session_id: str,
        message: str,
    ) -> ResponsePayload:
        # Main routing method that gives priority to the most explicit feature that matches the query.
        msg = (message or "").strip()
        if not msg:
            return self._response(
                session_id=session_id,
                last_query="",
                reply="Please type a question.",
                confidence=0.0,
                needs_clarification=False,
                clarification_question="",
                clarification_options=[],
                verification_status=None,
                evidence_score=None,
                source="fixed",
            )

        msg_norm = _normalise_query(msg)

        state = self.session_state.get(session_id, {})
        active_quiz = self._active_quiz_state(session_id)
        # Quiz-first rule that prevents an active quiz from being interrupted by normal retrieval logic.
        if active_quiz:
            if self._is_quiz_request(msg):
                return self._start_quiz_response(session_id, msg_norm)
            return self._handle_quiz_turn(session_id, msg_norm, active_quiz)

        if self._is_quiz_request(msg):
            return self._start_quiz_response(session_id, msg_norm)

        if state.get("awaiting_retrieval_confirmation"):
            # Confirmation branch that resolves the previous retrieval answer before starting a new topic.
            if self._is_confirmation_yes(msg_norm):
                return self._response(
                    session_id=session_id,
                    last_query=state.get("last_query", ""),
                    reply="Great! Is there anything else I can help you with?",
                    confidence=1.0,
                    needs_clarification=False,
                    clarification_question="",
                    clarification_options=[],
                    verification_status="verified",
                    evidence_score=1.0,
                    source="fixed",
                    topic=state.get("last_topic", ""),
                )
            if self._is_confirmation_no(msg_norm):
                return self._clarification_response(
                    session_id=session_id,
                    last_query=state.get("last_query", msg_norm),
                    results=[],
                    best_score=0.0,
                    reply="Please choose a topic so I can narrow it down.",
                )

        if state.get("awaiting_open_confirmation"):
            # Confirmation branch that handles follow-up page-opening prompts from summary and fixed intent flows.
            if self._is_confirmation_yes(msg_norm):
                payload = state.get("pending_open_payload") or {}
                url = state.get("pending_open_url", "")
                title = state.get("pending_open_title") or "Bright Futures 4 All"
                citations = [Citation(title=title, url=url)] if url else []
                actions = [Action(type="OPEN_URL", payload=payload)] if payload.get("url") else []
                return self._response(
                    session_id=session_id,
                    last_query=state.get("last_query", ""),
                    reply="Opening that page for you.",
                    citations=citations,
                    actions=actions,
                    confidence=1.0,
                    needs_clarification=False,
                    clarification_question="",
                    clarification_options=[],
                    verification_status="verified",
                    evidence_score=1.0,
                    source="nav",
                    topic=state.get("last_topic", ""),
                )
            if self._is_confirmation_no(msg_norm):
                return self._response(
                    session_id=session_id,
                    last_query=state.get("last_query", msg_norm),
                    reply="No problem. Is there anything else I can help you with?",
                    confidence=1.0,
                    needs_clarification=False,
                    clarification_question="",
                    clarification_options=[],
                    verification_status="verified",
                    evidence_score=1.0,
                    source="fixed",
                    topic=state.get("last_topic", ""),
                )

        nav = self._nav_match(msg_norm)
        # Navigation branch that opens a page straight away when the user clearly asks to go somewhere.
        if nav:
            payload = self._build_open_payload(nav)
            return self._response(
                session_id=session_id,
                last_query=msg_norm,
                reply="Opening that page for you.",
                citations=[Citation(title=nav.get("title") or "Bright Futures 4 All", url=nav["url"])],
                actions=[Action(type="OPEN_URL", payload=payload)],
                quick=["Open page", "Donate", "Contact", "About BF4A"],
                confidence=1.0,
                needs_clarification=False,
                clarification_question="",
                clarification_options=[],
                verification_status="verified",
                evidence_score=1.0,
                source="nav",
                topic=nav.get("id") or "",
            )

        if self._is_page_summary_query(msg_norm):
            # Summary branch that matches the requested page and then generates the summary response.
            summary_target = self._summary_page_match(msg_norm)
            if summary_target:
                summary_result = self._page_summary_response(
                    session_id=session_id,
                    msg_norm=msg_norm,
                    target=summary_target,
                )
                if summary_result:
                    return summary_result

        intent_label, intent_conf = "OTHER", 0.0
        try:
            # Intent prediction still uses the raw message with personal wording removed rather than the retrieval form.
            intent_input = _strip_personal_terms(msg) or msg
            intent_label, intent_conf = self.intent.predict(intent_input)
        except Exception:
            intent_label, intent_conf = "OTHER", 0.0

        forced_intent = self._forced_intent_label(msg, msg_norm)
        # Forced intents override the model output for a small number of important queries.
        if forced_intent:
            intent_label, intent_conf = forced_intent, max(float(intent_conf), 1.0)

        kb_guard_topic = self._kb_guard_topic(msg_norm)
        # KB guard stops a few curated topics from being incorrectly absorbed by chatbot or contact intents.
        if kb_guard_topic and intent_label in {"ABOUT_CHATBOT", "CONTACT_INFO"}:
            intent_label, intent_conf = "OTHER", 0.0

        fixed = self._fixed_flow(msg_norm, intent_label, float(intent_conf))
        # Fixed-flow branch returns a curated reply when the final intent is strong enough.
        if fixed is not None:
            url = fixed.get("url") or ""
            title = fixed.get("title") or "Bright Futures 4 All"
            payload = {"url": url} if url else {}

            citations = [Citation(title=title, url=url)] if url else []
            should_confirm_open = self._should_confirm_open(fixed.get("reply") or "", url)

            return self._response(
                session_id=session_id,
                last_query=msg_norm,
                reply=fixed.get("reply") or "",
                citations=citations,
                actions=[],
                quick=[] if should_confirm_open else (fixed.get("quick") or []),
                confidence=1.0,
                needs_clarification=False,
                clarification_question="",
                clarification_options=[],
                verification_status="verified",
                evidence_score=1.0,
                source="fixed",
                topic=self._topic_from_text(msg_norm) or "",
                awaiting_open_confirmation=should_confirm_open,
                pending_open_payload=payload if should_confirm_open else None,
                pending_open_title=title,
                pending_open_url=url,
            )

        topic_hint = self._topic_from_text(msg_norm) or ""
        # Retrieval branch acts as the final answer path for questions not handled earlier.
        return self._run_retrieval(
            session_id=session_id,
            msg_norm=msg_norm,
            intent_label=intent_label,
            forced_allowed=None,
            topic=topic_hint,
        )
