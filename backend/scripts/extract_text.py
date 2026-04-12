import os
import re
import json
import time
from urllib.parse import urlparse, urldefrag, urlunparse

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

URL_LIST_PATH = os.path.join("data", "sitemaps", "pages_urls.txt")
OUT_DIR = os.path.join("data", "html")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.json")

ALLOWED_DOMAIN = "www.brightfutures4all.com"
REQUEST_DELAY_SEC = 0.2
MAX_PAGES = 200

SKIP_PATH_CONTAINS = [
    "/projects-3",
    "/general-clean",
    "/post/",
    "/login",
]

NAV_TIMEOUT_MS = 20000
POST_LOAD_WAIT_MS = 600


def should_skip(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(s in path for s in SKIP_PATH_CONTAINS)


def is_allowed(url: str) -> bool:
    try:
        u = urlparse(url)
        return u.netloc == ALLOWED_DOMAIN and u.scheme in ("http", "https")
    except Exception:
        return False


def normalise(url: str) -> str:
    url, _ = urldefrag(url)
    u = urlparse(url)
    u = u._replace(query="")
    return urlunparse(u).rstrip("/")


def safe_filename(url: str) -> str:
    p = urlparse(url).path.strip("/")
    if not p:
        p = "home"
    p = re.sub(r"[^a-zA-Z0-9]+", "-", p).strip("-").lower()
    if not p:
        p = "page"
    return f"{p}.html"


def load_urls() -> list[str]:
    if not os.path.exists(URL_LIST_PATH):
        raise FileNotFoundError(f"Missing URL list: {URL_LIST_PATH}")

    urls = []
    with open(URL_LIST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if not u:
                continue
            u = normalise(u)
            if not is_allowed(u):
                continue
            if should_skip(u):
                continue
            urls.append(u)

    seen = set()
    out = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)

    return out[:MAX_PAGES]


def save_manifest(items: list[dict]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    urls = load_urls()
    print(f"Crawling {len(urls)} URLs (max={MAX_PAGES})")

    manifest = []
    used_files = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="bf4a-dissertation-bot/1.0 (educational)",
            viewport={"width": 1280, "height": 720},
        )

        def block_heavy(route):
            # Route filter that skips heavy assets because only the page HTML is needed.
            rtype = route.request.resource_type
            if rtype in ("image", "media", "font"):
                return route.abort()
            return route.continue_()

        context.route("*/", block_heavy)

        page = context.new_page()
        page.set_default_navigation_timeout(NAV_TIMEOUT_MS)
        page.set_default_timeout(NAV_TIMEOUT_MS)

        for i, url in enumerate(urls, start=1):
            print(f"[{i}/{len(urls)}] {url}")

            fn = safe_filename(url)
            if fn in used_files:
                base = fn[:-5]
                k = 2
                while f"{base}-{k}.html" in used_files:
                    k += 1
                fn = f"{base}-{k}.html"
            used_files.add(fn)

            out_path = os.path.join(OUT_DIR, fn)

            status = "ok"
            err = ""
            html = ""

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
                page.wait_for_timeout(POST_LOAD_WAIT_MS)
                html = page.content()
            except PlaywrightTimeoutError:
                status = "timeout"
                try:
                    html = page.content()
                except Exception:
                    html = ""
            except Exception as e:
                status = "error"
                err = str(e)
                html = ""

            if html:
                with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(html)

            manifest.append({"url": url, "file": fn, "status": status, "error": err})

            if i % 5 == 0:
                # Checkpoint save that reduces the impact of the crawl stopping halfway through.
                save_manifest(manifest)

            time.sleep(REQUEST_DELAY_SEC)

        save_manifest(manifest)
        context.close()
        browser.close()

    ok = sum(1 for x in manifest if x.get("status") == "ok")
    print(f"Done. Crawled {len(manifest)} pages. ok={ok}. Saved HTML to {OUT_DIR} and manifest to {MANIFEST_PATH}")


if __name__ == "_main_":
    main()
