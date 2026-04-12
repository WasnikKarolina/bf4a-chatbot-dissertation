import os
import json
import time
import asyncio
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError


DATA_DIR = os.path.join("data", "pages_html")
URLS_PATH = os.path.join("data", "urls.json")
MAX_PAGES = 200
CONCURRENCY = 4
GOTO_TIMEOUT_MS = 30000
TOTAL_TIMEOUT_S = 900
RETRIES = 2


def _safe_filename(url: str) -> str:
    # Filename helper that only needs to produce a stable and readable local name.
    p = urlparse(url)
    path = p.path.strip("/").replace("/", "_")
    if not path:
        path = "home"
    if len(path) > 160:
        path = path[:160]
    return f"{path}.html"


async def _fetch_one(context, url: str, attempt: int) -> Optional[str]:
    page = await context.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=GOTO_TIMEOUT_MS)
        await page.wait_for_timeout(800)
        html = await page.content()
        return html
    except PlaywrightTimeoutError:
        return None
    except Exception:
        return None
    finally:
        try:
            await page.close()
        except Exception:
            pass


async def _worker(name: str, sem: asyncio.Semaphore, context, url: str) -> Dict[str, Any]:
    async with sem:
        # Retry loop that helps with pages which load more slowly on Wix.
        for attempt in range(RETRIES + 1):
            html = await _fetch_one(context, url, attempt)
            if html and len(html) > 500:
                return {"url": url, "ok": True, "html": html, "attempt": attempt}
        return {"url": url, "ok": False, "html": None, "attempt": RETRIES}


async def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(URLS_PATH):
        raise FileNotFoundError(f"Missing {URLS_PATH}. Generate urls.json first.")

    with open(URLS_PATH, "r", encoding="utf-8") as f:
        urls = json.load(f)

    if not isinstance(urls, list):
        raise ValueError("data/urls.json must be a JSON list of URLs")

    urls = [u for u in urls if isinstance(u, str) and u.startswith("http")]
    urls = urls[:MAX_PAGES]

    print(f"Crawling {len(urls)} URLs (max={MAX_PAGES}, concurrency={CONCURRENCY})")

    start = time.time()
    sem = asyncio.Semaphore(CONCURRENCY)

    async with async_playwright() as p:
        # Crawl setup that captures raw HTML snapshots for the later text-processing pipeline.
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        )
        context.set_default_timeout(GOTO_TIMEOUT_MS)

        tasks = []
        for i, url in enumerate(urls, start=1):
            tasks.append(_worker(f"W{i}", sem, context, url))

        completed = 0
        ok = 0
        failed = 0

        for coro in asyncio.as_completed(tasks, timeout=TOTAL_TIMEOUT_S):
            res = await coro
            completed += 1
            if res["ok"]:
                ok += 1
                fp = os.path.join(DATA_DIR, _safe_filename(res["url"]))
                with open(fp, "w", encoding="utf-8") as out:
                    out.write(res["html"])
            else:
                failed += 1

            if completed % 5 == 0 or completed == len(tasks):
                elapsed = time.time() - start
                print(f"Progress: {completed}/{len(tasks)} | ok={ok} | failed={failed} | {elapsed:.1f}s")

        await context.close()
        await browser.close()

    print(f"Done. ok={ok}, failed={failed}, saved_dir={DATA_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
