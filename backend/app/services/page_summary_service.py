import json
import os
import re
from typing import Optional


SUMMARY_KEYWORDS = {
    "children": 2.2,
    "young people": 2.1,
    "young adults": 1.7,
    "families": 2.1,
    "family": 1.8,
    "support": 2.0,
    "programme": 1.8,
    "programmes": 1.8,
    "activities": 1.5,
    "send": 2.1,
    "mentoring": 1.8,
    "mentor": 1.8,
    "therapy": 1.7,
    "tutoring": 1.7,
    "education": 1.7,
    "wellbeing": 1.7,
    "community": 1.6,
    "leadership": 1.5,
    "workshops": 1.4,
    "volunteer": 1.4,
    "volunteers": 1.4,
    "vacancy": 1.8,
    "vacancies": 1.8,
    "roles": 1.5,
    "jobs": 1.5,
    "freelance": 1.4,
    "apply": 1.2,
    "donate": 1.8,
    "donation": 1.8,
    "therapists": 1.6,
    "therapeutic": 1.5,
    "counselling": 1.4,
    "academic": 1.4,
    "confidence": 1.3,
    "skills": 1.3,
    "holiday": 1.8,
    "club": 1.6,
    "clubs": 1.6,
    "homeschooling": 2.0,
    "curriculum": 1.9,
    "lessons": 1.4,
    "teachers": 1.3,
    "maths": 1.3,
    "english": 1.3,
    "science": 1.3,
    "trips": 1.4,
    "outdoor": 1.2,
    "cultural": 1.3,
    "inclusive": 1.4,
}

NOISE_PATTERNS = [
    r"cookie",
    r"privacy preference",
    r"accept all",
    r"manage cookies",
    r"all rights reserved",
    r"copyright bright futures 4 all",
    r"company no",
    r"registration number",
    r"tel:\s*[0-9 ]+",
    r"mob:\s*[0-9 ]+",
    r"info@brightfutures4all\.com",
]

CTA_PATTERNS = [
    r"register now",
    r"book now",
    r"apply now",
    r"make a donation",
    r"donate now",
    r"click here",
    r"contact us for further information",
    r"if you like our work and would like to support us",
]

NAV_LINES = {
    "about us",
    "the team",
    "newsletters",
    "photo gallery",
    "policies",
    "vacancies",
    "projects",
    "after school activities",
    "holiday clubs",
    "tutors, mentors & therapists",
    "youth development, leadership & advocacy",
    "talks and workshops",
    "more",
    "home",
    "top of page",
}

LOW_VALUE_SENTENCE_PATTERNS = [
    r"\bwhat our stakeholders say\b",
    r"\bour stories\b",
    r"\bbook our ceo\b",
    r"\bthought leader\b",
    r"\bupcoming events?\b",
    r"\bshare (?:her|his|their) insights\b",
    r"\bfeel less isolated\b",
    r"\bfeel empowered\b",
    r"\bfocus group\b",
    r"\bcontact us for further information\b",
    r"\bdonate time, resources\b",
    r"\bclosing the loop\b",
    r"\bregister now\b",
    r"\bmake a donation\b",
    r"\bjoin us\b",
    r"\bthank you so much\b",
]

ADMIN_PATTERNS = [
    r"\bcontact the manager\b",
    r"\bfor all job enquiries\b",
    r"\bhow to apply\b",
    r"\bcontract\s*:",
    r"\bsalary\s*:",
    r"\bclosing date\s*:",
    r"\blocation\s*:",
    r"\bday rate\s*:",
    r"\boffice:\b",
    r"\bmobile:\b",
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is",
    "it", "its", "of", "on", "or", "that", "the", "their", "this", "to", "was", "we", "with",
    "our", "your", "they", "them", "these", "those", "all", "into", "during",
}

ALLOWED_NEW_SUMMARY_TOKENS = {
    "bf4a", "bright", "futures", "page", "supports", "provides", "offers", "runs", "helps",
    "families", "children", "young", "people", "community", "services", "programmes", "activities",
    "clubs", "holiday", "after", "school", "workshops", "talks", "send", "wellbeing", "mentoring",
    "therapy", "tutoring", "donations", "donation", "vacancies", "roles", "homeschooling", "curriculum",
    "lessons", "teachers", "trips", "volunteer", "volunteers", "support", "education", "inclusive",
}


class PageSummaryService:
    def __init__(self, pages_dir: str):
        # Model state that stays unloaded until the summarisation feature is actually used.
        self.pages_dir = pages_dir
        self._summarizer = None
        self._summarizer_tokenizer = None

    def summarize_url(self, url: str, max_input_chars: int = 2600) -> Optional[str]:
        # Summary flow that starts from saved page text rather than a live web request.
        page = self._load_page_by_url(url)
        if not page:
            return None

        title, sentences, source = self._prepare_summary(page, max_input_chars=max_input_chars)
        if not source or not sentences:
            return None

        # Generation step that gives the model the first attempt before using the grounded fallback.
        generated = self._ai_rewrite_summary(source, title)
        cleaned = self._clean_summary(generated)
        if cleaned and self._is_acceptable_generated_summary(cleaned, source, title):
            return cleaned

        fallback = self._clean_summary(" ".join(sentences[:3]))
        if fallback:
            return fallback
        return None

    def _load_page_by_url(self, url: str) -> Optional[dict]:
        # URL matching rule that accounts for the two stored forms of the home page.
        if not os.path.isdir(self.pages_dir):
            return None

        target = (url or "").rstrip("/").lower()
        target_aliases = {target}
        if target == "https://www.brightfutures4all.com":
            target_aliases.add("https://www.brightfutures4all.com/home")
        elif target == "https://www.brightfutures4all.com/home":
            target_aliases.add("https://www.brightfutures4all.com")

        for name in os.listdir(self.pages_dir):
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.pages_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    row = json.load(f)
            except Exception:
                continue
            candidate = str(row.get("url") or "").rstrip("/").lower()
            if candidate in target_aliases:
                return row
        return None

    def _prepare_summary(self, page: dict, max_input_chars: int = 2600) -> tuple[str, list[str], str]:
        # Preparation step that cleans the page text and turns the strongest lines into model notes.
        title = self._clean_title(page.get("title") or "")
        url = page.get("url") or ""
        lines = self._clean_lines(page.get("text") or "", title)
        if not lines:
            return title, [], ""

        specialised = self._prepare_specialised_summary(title, url, lines)
        if specialised is not None:
            sentences, source = specialised
            if sentences and source:
                return title, sentences, source[:max_input_chars].strip()

        blocks = self._build_blocks(lines)
        if not blocks:
            return title, [], ""

        selected_blocks = self._select_blocks(blocks, title, url)
        sentences = self._select_sentences(selected_blocks, title, url)
        if not sentences:
            return title, [], ""

        source = self._build_source_from_sentences(title, sentences[:6])
        return title, sentences, source[:max_input_chars].strip()

    def _build_source_from_sentences(self, title: str, sentences: list[str]) -> str:
        # Source builder that keeps the prompt simple and grounded in selected sentences.
        source_parts = [f"Page title: {title or 'Bright Futures 4 All'}"]
        source_parts.extend(f"- {sentence}" for sentence in sentences if sentence)
        return "\n".join(source_parts).strip()

    def _page_kind(self, title: str, url: str) -> str:
        # Page classifier that routes a few page types into specialised summary preparation.
        haystack = f"{title} {url}".lower()
        if "vacanc" in haystack or "projects-3" in haystack:
            return "vacancies"
        if "tutors" in haystack or "mentors" in haystack or "therapists" in haystack or "tutoring" in haystack:
            return "tutoring"
        if "homeschool" in haystack:
            return "homeschooling"
        return ""

    def _prepare_specialised_summary(self, title: str, url: str, lines: list[str]) -> Optional[tuple[list[str], str]]:
        # Specialised summary path that handles pages with more awkward or noisy content structures.
        page_kind = self._page_kind(title, url)
        if page_kind == "vacancies":
            return self._prepare_vacancies_summary(title, lines)
        if page_kind == "tutoring":
            return self._prepare_tutoring_summary(title, lines)
        if page_kind == "homeschooling":
            return self._prepare_homeschooling_summary(title, lines)
        return None

    def _prepare_vacancies_summary(self, title: str, lines: list[str]) -> Optional[tuple[list[str], str]]:
        # Summary builder that pulls the main roles and purpose out of a vacancy page.
        role_headings = []
        for line in lines:
            if len(line.split()) > 6:
                continue
            low = line.lower()
            if any(re.search(pattern, low, flags=re.IGNORECASE) for pattern in ADMIN_PATTERNS):
                continue
            if low in NAV_LINES:
                continue
            if any(char.isdigit() for char in line):
                continue
            role_headings.append(line)
            if len(role_headings) == 5:
                break

        role_summary = ""
        if role_headings:
            role_summary = "This page lists BF4A volunteer opportunities and vacancies, including " + ", ".join(role_headings[:-1])
            if len(role_headings) > 1:
                role_summary += f", and {role_headings[-1]}."
            else:
                role_summary += f"{role_headings[0]}."

        support_line = self._find_best_line(
            lines,
            ["supporting neurodiverse", "dedicated to supporting", "inclusive education", "well-being and advisory services"],
        )
        responsibilities_line = self._find_best_line(
            lines,
            ["fundraising initiatives", "community relationships", "after-school projects", "holiday clubs", "donor stewardship", "support duties"],
        )
        fundraising_line = self._find_best_line(
            lines,
            ["fundraising initiatives", "community relationships", "donor stewardship", "resource acquisition", "communications"],
        )

        support_summary = ""
        if support_line:
            support_summary = "BF4A is a charitable community hub supporting neurodiverse and disadvantaged children and their families."

        responsibilities_summary = ""
        if responsibilities_line or fundraising_line:
            parts = []
            if fundraising_line:
                parts.append("fundraising, communications and community engagement")
            if responsibilities_line:
                parts.append("after-school projects and holiday clubs")
            if parts:
                if len(parts) == 1:
                    responsibilities_summary = f"The roles support {parts[0]}."
                else:
                    responsibilities_summary = f"The roles support {parts[0]} as well as {parts[1]}."

        sentences = [part for part in [role_summary, support_summary, responsibilities_summary] if part]
        if not sentences:
            return None

        sentences = sentences[:4]
        return sentences, self._build_source_from_sentences(title or "Vacancies", sentences)

    def _prepare_tutoring_summary(self, title: str, lines: list[str]) -> Optional[tuple[list[str], str]]:
        # Summary builder that keeps the result centred on tutoring, mentoring, and therapy support.
        overview_line = self._find_best_line(
            lines,
            ["customized academic assistance", "specialized support for emotional well-being", "one-on-one guidance", "comprehensive support for our community"],
        )
        tutoring_line = self._find_best_line(
            lines,
            ["english, maths, science and reasoning", "sats", "11+ exam preparation", "gcse", "highly qualified"],
        )
        therapy_line = self._find_best_line(
            lines,
            ["range of psychotherapists", "group therapy", "emotional support", "therapeutic", "social skills"],
        )

        overview_summary = ""
        if overview_line:
            overview_summary = (
                "BF4A offers tutoring, mentoring and therapeutic support, combining academic help, one-to-one guidance, and support for emotional wellbeing and social skills."
            )

        tutoring_summary = ""
        if tutoring_line:
            tutoring_summary = (
                "Tutoring covers English, maths, science and reasoning, including SATS, 11+ and GCSE preparation for school-aged learners."
            )

        therapy_summary = ""
        if therapy_line:
            therapy_summary = (
                "The page also highlights therapy and psychotherapist support for emotional wellbeing, resilience and wider personal development."
            )

        sentences = [part for part in [overview_summary, tutoring_summary, therapy_summary] if part]
        if not sentences:
            return None

        sentences = sentences[:4]
        return sentences, self._build_source_from_sentences(title or "Tutors, Mentors & Therapists", sentences)

    def _prepare_homeschooling_summary(self, title: str, lines: list[str]) -> Optional[tuple[list[str], str]]:
        # Summary builder that keeps home schooling summaries focused on lessons and support.
        support_line = self._find_best_line(
            lines,
            ["quality support in maths, english, and science", "support in maths, english, and science", "maths, english, and science"],
        )
        trips_line = self._find_best_line(
            lines,
            ["students participate monthly in cultural trips and outdoor learning", "cultural trips and outdoor learning", "cultural outings"],
        )
        curriculum_line = self._find_best_line(
            lines,
            ["lessons with trained teachers", "please see our curriculum below", "holistic development of each student", "subjects taught include"],
        )

        support_summary = ""
        if support_line:
            support_summary = (
                "BF4A's home schooling provision supports school-aged children and parents with maths, English and science."
            )

        trips_summary = ""
        if trips_line:
            trips_summary = "Students also take part in cultural trips and outdoor learning."

        curriculum_summary = ""
        if curriculum_line:
            curriculum_summary = (
                "The page highlights lessons with trained teachers and a curriculum focused on wider personal development."
            )

        sentences = [part for part in [support_summary, trips_summary, curriculum_summary] if part]
        if not sentences:
            return None

        sentences = sentences[:4]
        return sentences, self._build_source_from_sentences(title or "Homeschooling", sentences)

    def _find_best_line(self, lines: list[str], keywords: list[str]) -> str:
        # Scoring helper that picks the strongest grounded line around a target idea.
        best_line = ""
        best_score = 0.0
        for line in lines:
            low = line.lower()
            score = 0.0
            if any(re.search(pattern, low, flags=re.IGNORECASE) for pattern in ADMIN_PATTERNS):
                score -= 4.0
            if '"' in line:
                score -= 2.0
            if re.search(r"\b(mother|story|testimonial|confidence over the last|predicted grade)\b", low):
                score -= 2.0
            for keyword in keywords:
                if keyword in low:
                    score += 2.2
            if len(line.split()) < 8:
                score -= 0.8
            if len(line.split()) > 45:
                score -= 0.4
            if score > best_score:
                best_score = score
                best_line = line
        return best_line if best_score > 0 else ""

    def _get_summarizer(self):
        # Model loader that keeps the feature local rather than relying on an external API.
        if self._summarizer is not None and self._summarizer_tokenizer is not None:
            return self._summarizer, self._summarizer_tokenizer

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name = "google/flan-t5-base"
        self._summarizer_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._summarizer = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return self._summarizer, self._summarizer_tokenizer

    def _ai_rewrite_summary(self, text: str, title: str) -> str:
        # Prompt builder that asks for a short factual rewrite from the selected notes.
        source = re.sub(r"\s+", " ", str(text or "")).strip()
        if not source:
            return ""

        try:
            model, tokenizer = self._get_summarizer()
        except Exception:
            return ""

        prompt = (
            "Summarise this Bright Futures 4 All webpage in two or three short factual sentences. "
            "Use only the information in the notes. Keep it clear and natural. "
            "Do not include testimonials, quotes, cookies, navigation, contact details, or calls to action unless they are the page's main topic.\n"
            f"{source}"
        )
        try:
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=640,
            )
            output = model.generate(
                **encoded,
                max_new_tokens=110,
                num_beams=5,
                do_sample=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.12,
                length_penalty=1.0,
                early_stopping=True,
            )
            return tokenizer.decode(output[0], skip_special_tokens=True).strip()
        except Exception:
            return ""

    def _clean_title(self, title: str) -> str:
        # Title cleanup that removes common noise from the saved page titles.
        cleaned = re.sub(r"\|\s*Bright Futures 4 All\b", "", str(title or ""), flags=re.IGNORECASE)
        cleaned = cleaned.replace("(old)", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -|")
        return cleaned.title() if cleaned.isupper() else cleaned

    def _normalise_line(self, line: str) -> str:
        # Line cleanup that fixes leftover extraction characters before scoring.
        cleaned = str(line or "")
        cleaned = cleaned.replace("\u2019", "'").replace("\u2018", "'")
        cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
        cleaned = cleaned.replace("â€œ", '"').replace("â€", '"').replace("â€™", "'")
        cleaned = cleaned.replace("&nbsp;", " ")
        cleaned = cleaned.replace("\u200b", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _looks_like_heading(self, line: str) -> bool:
        # Heading check that stays simple because the earlier cleaning already removes most of the worst noise.
        if not line:
            return False
        if len(line) > 85:
            return False
        if len(line.split()) > 10:
            return False
        if re.search(r"[.!?]", line):
            return False
        return True

    def _is_noise_line(self, line: str, title: str) -> bool:
        # Noise filter that removes boilerplate before any scoring or generation happens.
        low = line.lower().strip()
        if not low:
            return True
        if low == title.lower().strip():
            return True
        if low in NAV_LINES:
            return True
        if any(re.search(pattern, low, flags=re.IGNORECASE) for pattern in NOISE_PATTERNS):
            return True
        if any(re.search(pattern, low, flags=re.IGNORECASE) for pattern in CTA_PATTERNS):
            return True
        if re.fullmatch(r"[^\w]*", low):
            return True
        if len(low.split()) <= 2 and not self._looks_like_heading(line):
            return True
        return False

    def _clean_lines(self, text: str, title: str) -> list[str]:
        raw_lines = [self._normalise_line(line) for line in str(text or "").replace("\r", "\n").split("\n")]
        lines = []
        seen = set()
        prev = ""

        for line in raw_lines:
            if self._is_noise_line(line, title):
                continue
            if prev and line.lower() == prev.lower():
                continue
            key = line.lower()
            if key in seen:
                continue
            if len(line.split()) <= 3 and not self._looks_like_heading(line):
                continue
            lines.append(line)
            seen.add(key)
            prev = line
        return lines

    def _build_blocks(self, lines: list[str]) -> list[dict]:
        blocks = []
        heading = ""
        current: list[str] = []

        def flush():
            nonlocal heading, current
            text = re.sub(r"\s+", " ", " ".join(current)).strip()
            if text and len(text.split()) >= 8:
                blocks.append({"heading": heading, "text": text})
            heading = ""
            current = []

        for line in lines:
            if self._looks_like_heading(line):
                if current:
                    flush()
                heading = line
                continue
            current.append(line)
        flush()
        return blocks

    def _score_block(self, block: dict, title: str, url: str) -> float:
        text = f"{block.get('heading') or ''} {block.get('text') or ''}".lower()
        score = 0.0

        for phrase, weight in SUMMARY_KEYWORDS.items():
            if phrase in text:
                score += weight

        title_tokens = [token for token in re.findall(r"[a-z0-9']+", title.lower()) if len(token) > 2]
        url_tokens = [token for token in re.findall(r"[a-z0-9']+", url.lower()) if len(token) > 2]
        for token in set(title_tokens + url_tokens):
            if token in text:
                score += 0.7

        if block.get("heading"):
            score += 0.6
        if len(block.get("text", "").split()) >= 18:
            score += 0.5
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in LOW_VALUE_SENTENCE_PATTERNS):
            score -= 2.5
        if '"' in text:
            score -= 1.2
        return score

    def _select_blocks(self, blocks: list[dict], title: str, url: str) -> list[dict]:
        scored = []
        for idx, block in enumerate(blocks):
            score = self._score_block(block, title, url)
            if idx == 0:
                score += 0.5
            scored.append((score, idx, block))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)

        selected = []
        seen = set()
        for score, idx, block in scored[:5]:
            key = block["text"].lower()
            if key in seen:
                continue
            selected.append((idx, block))
            seen.add(key)
        selected.sort(key=lambda item: item[0])
        return [block for _, block in selected[:5]]

    def _split_sentences(self, text: str) -> list[str]:
        cleaned = re.sub(r"(including)\s*:\s+", r"\1. ", str(text or ""), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        sentences = []
        for part in parts:
            sentence = part.strip(" -")
            if len(sentence.split()) < 6:
                continue
            if sentence:
                sentences.append(sentence)
        return sentences

    def _score_sentence(self, sentence: str, heading: str, title: str, url: str, index: int) -> float:
        text = sentence.lower()
        score = 0.0
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in LOW_VALUE_SENTENCE_PATTERNS):
            return -10.0
        if re.search(r"\b(i|me|my|mine)\b", text):
            score -= 2.0
        if '"' in sentence or "testimonial" in text or "our stories" in text:
            score -= 1.8
        for phrase, weight in SUMMARY_KEYWORDS.items():
            if phrase in text:
                score += weight
        if heading:
            heading_low = heading.lower()
            for phrase, weight in SUMMARY_KEYWORDS.items():
                if phrase in heading_low:
                    score += weight * 0.3
        title_tokens = [token for token in re.findall(r"[a-z0-9']+", title.lower()) if len(token) > 2]
        url_tokens = [token for token in re.findall(r"[a-z0-9']+", url.lower()) if len(token) > 2]
        for token in set(title_tokens + url_tokens):
            if token in text:
                score += 0.7
        if index == 0:
            score += 0.4
        if 10 <= len(sentence.split()) <= 30:
            score += 0.6
        return score

    def _select_sentences(self, blocks: list[dict], title: str, url: str) -> list[str]:
        scored = []
        seen = set()
        for block_index, block in enumerate(blocks):
            for sentence_index, sentence in enumerate(self._split_sentences(block.get("text") or "")):
                key = sentence.lower()
                if key in seen:
                    continue
                seen.add(key)
                score = self._score_sentence(sentence, block.get("heading") or "", title, url, sentence_index)
                if block_index == 0:
                    score += 0.2
                scored.append((score, block_index, sentence_index, sentence))

        if not scored:
            return []

        scored.sort(key=lambda item: (item[0], -item[1], -item[2]), reverse=True)
        selected = []
        for item in scored:
            sentence = item[3]
            if any(self._sentence_overlap(sentence, existing[3]) >= 0.72 for existing in selected):
                continue
            selected.append(item)
            if len(selected) == 4:
                break

        ordered = sorted(selected, key=lambda item: (item[1], item[2]))
        return [sentence for _, _, _, sentence in ordered]

    def _clean_summary(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        cleaned = re.sub(r"(?:\.{3}|\u2026)+\s*$", "", cleaned)
        if not cleaned:
            return ""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s and s.strip()]
        limited = []
        seen = set()
        total_chars = 0
        for sentence in sentences[:3]:
            candidate = sentence.strip()
            key = candidate.lower().rstrip(".!?")
            if not key or key in seen:
                continue
            seen.add(key)
            if candidate[-1] not in ".!?":
                candidate += "."
            next_total = total_chars + len(candidate) + (1 if limited else 0)
            if limited and next_total > 480:
                break
            limited.append(candidate)
            total_chars = next_total
        return " ".join(limited).strip()

    def _is_acceptable_generated_summary(self, summary: str, source: str, title: str) -> bool:
        # Validation rule that rejects summaries which drift too far from the selected source notes.
        candidate = self._clean_summary(summary)
        if not candidate:
            return False
        low = candidate.lower()
        page_kind = self._page_kind(title, "")
        sentence_count = len([part for part in re.split(r"(?<=[.!?])\s+", candidate) if part.strip()])
        if title and low == title.lower():
            return False
        if len(candidate.split()) < 12:
            return False
        if any(re.search(pattern, low, flags=re.IGNORECASE) for pattern in LOW_VALUE_SENTENCE_PATTERNS):
            return False
        if page_kind == "vacancies":
            if sentence_count < 2:
                return False
            if not any(token in low for token in ["vacanc", "volunteer", "roles", "jobs"]):
                return False
            if not any(token in low for token in ["fundraising", "holiday", "project", "support assistant", "play workers", "communications"]):
                return False
        if page_kind == "tutoring":
            if sentence_count < 2:
                return False
            if not any(token in low for token in ["tutor", "mentor", "therap", "support"]):
                return False
            if not any(token in low for token in ["english", "maths", "science", "gcse", "emotional", "social skills", "psychotherapist"]):
                return False
        if page_kind == "homeschooling":
            if sentence_count < 2:
                return False
            if not any(token in low for token in ["maths", "english", "science", "curriculum", "lessons"]):
                return False
            if not any(token in low for token in ["trips", "outdoor", "cultural", "teachers", "students"]):
                return False

        source_tokens = self._content_tokens(source)
        candidate_tokens = self._content_tokens(candidate)
        if not source_tokens or not candidate_tokens:
            return False

        overlap = len(source_tokens & candidate_tokens) / max(1, min(len(source_tokens), len(candidate_tokens)))
        if overlap < 0.35:
            return False

        extra_tokens = candidate_tokens - source_tokens - self._content_tokens(title) - ALLOWED_NEW_SUMMARY_TOKENS
        if len(extra_tokens) > 4:
            return False
        return True

    def _sentence_overlap(self, left: str, right: str) -> float:
        left_tokens = self._content_tokens(left)
        right_tokens = self._content_tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        intersection = len(left_tokens & right_tokens)
        baseline = min(len(left_tokens), len(right_tokens))
        if baseline == 0:
            return 0.0
        return intersection / baseline

    def _content_tokens(self, text: str) -> set[str]:
        tokens = set()
        for token in re.findall(r"[a-z0-9']+", str(text or "").lower()):
            if len(token) < 4:
                continue
            if token in STOPWORDS:
                continue
            tokens.add(token)
        return tokens
