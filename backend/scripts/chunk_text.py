import os
import json
import glob
import re


TEXT_DIR = os.path.join("data", "pages_text")
OUT_PATH = os.path.join("data", "chunks", "chunks.jsonl")

TARGET_WORDS = 180
MAX_WORDS = 240

MIN_WORDS_DEFAULT = 60
MIN_WORDS_SHORT_PAGE = 20
MIN_WORDS_ABSOLUTE = 10
SHORT_PAGE_URL_HINTS = ["contact", "donate", "privacy", "safeguard", "gdpr", "policy"]


def _norm_space(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def is_heading(block: str) -> bool:
    # Heading check that uses a few simple rules to separate section titles from body text.
    b = _norm_space(block)
    if not b:
        return False
    if len(b) > 90:
        return False
    if "." in b:
        return False
    if len(b.split()) > 10:
        return False
    lo = b.lower()
    bad = ["cookie", "privacy", "terms", "menu", "log in", "sign up", "home", "top of page"]
    if any(x in lo for x in bad):
        return False
    return True


def split_blocks(text: str) -> list[str]:
    text = re.sub(r"\r", "\n", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    return blocks


def words_count(s: str) -> int:
    return len((s or "").split())


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    files = sorted(glob.glob(os.path.join(TEXT_DIR, "*.json")))
    total = 0

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                page = json.load(f)

            url = page.get("url") or ""
            title = page.get("title") or ""
            text = page.get("text") or ""
            blocks = split_blocks(text)
            if not blocks:
                continue

            heading_path = []
            current_section = ""
            buffer = []
            buffer_words = 0
            chunk_index = 0
            carry_text = ""

            url_lower = (url or "").lower()
            min_words = MIN_WORDS_SHORT_PAGE if any(h in url_lower for h in SHORT_PAGE_URL_HINTS) else MIN_WORDS_DEFAULT

            def flush(force: bool = False):
                nonlocal total, chunk_index, buffer, buffer_words, carry_text
                if not buffer:
                    return
                answer_text = _norm_space(" ".join(buffer))
                if carry_text:
                    answer_text = _norm_space(f"{carry_text} {answer_text}")
                    carry_text = ""

                n_words = words_count(answer_text)
                # Carry-over rule that keeps short but useful sections from disappearing between chunks.
                if n_words < min_words and not force:
                    carry_text = _norm_space(f"{carry_text} {answer_text}") if carry_text else answer_text
                    buffer = []
                    buffer_words = 0
                    return
                if n_words < MIN_WORDS_ABSOLUTE:
                    buffer = []
                    buffer_words = 0
                    return
                record = {
                    "id": f"{os.path.basename(fp)}::{chunk_index}",
                    "url": url,
                    "title": title,
                    "chunk_index": chunk_index,
                    "section_title": current_section,
                    "heading_path": heading_path[:],
                    "text": answer_text,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_index += 1
                total += 1
                buffer = []
                buffer_words = 0

            for b in blocks:
                b_norm = _norm_space(b)
                if not b_norm:
                    continue

                if is_heading(b_norm):
                    flush()
                    current_section = b_norm
                    heading_path = [b_norm]
                    continue

                w = words_count(b_norm)
                if buffer_words + w > MAX_WORDS:
                    flush()

                buffer.append(b_norm)
                buffer_words += w

                if buffer_words >= TARGET_WORDS:
                    flush()

            flush(force=True)
            if carry_text and words_count(carry_text) >= MIN_WORDS_ABSOLUTE:
                record = {
                    "id": f"{os.path.basename(fp)}::{chunk_index}",
                    "url": url,
                    "title": title,
                    "chunk_index": chunk_index,
                    "section_title": current_section,
                    "heading_path": heading_path[:],
                    "text": _norm_space(carry_text),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_index += 1
                total += 1

    print(f"Wrote {total} chunks to {OUT_PATH}")


if __name__ == "__main__":
    main()
