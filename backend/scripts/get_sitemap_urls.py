import re
import requests

SITEMAP_INDEX = "https://www.brightfutures4all.com/sitemap.xml"

def fetch(url: str) -> str:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text

def extract_locs(xml: str) -> list[str]:
    return [u.strip() for u in re.findall(r"<loc>(.*?)</loc>", xml) if u.strip()]

import os

OUT_PATH = os.path.join("data", "sitemaps", "pages_urls.txt")

def main():
    index_xml = fetch(SITEMAP_INDEX)
    sitemap_urls = extract_locs(index_xml)

    all_pages: list[str] = []

    for sm in sitemap_urls:
        # Filter that keeps only the page sitemap rather than every sitemap Wix exposes.
        if "pages-sitemap.xml" not in sm:
            continue
        pages_xml = fetch(sm)
        page_urls = extract_locs(pages_xml)
        all_pages.extend(page_urls)

    seen = set()
    all_pages = [u for u in all_pages if not (u in seen or seen.add(u))]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for u in all_pages:
            f.write(u.strip() + "\n")

    print("Saved", len(all_pages), "URLs to", OUT_PATH)
    print("First 10:")
    for u in all_pages[:10]:
        print(u)

if __name__ == "__main__":
    main()
