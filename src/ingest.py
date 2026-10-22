# Step 1: scrape the Python tutorial and cut it into clean chunks.
#
# The whole trick is HOW you cut the pages up. I chunk by the docs' own
# <section> blocks instead of blindly chopping every N characters, so a chunk
# never mixes two topics and each one keeps its heading + URL anchor (that's
# what lets the final answer cite a real source). Long sections get split into
# overlapping windows, but I break on paragraph boundaries so code examples
# don't get sliced in half.
#
# Writes data/chunks.jsonl, one JSON per line: {id, text, section_title, url, heading_path}

import json
import os
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

HEADERS = {"User-Agent": "Mozilla/5.0 (RAG task ingest)"}


def fetch(url):
    """Download a page and force UTF-8 so special characters don't get garbled."""
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.encoding = "utf-8"
    return r.text


def clean_heading(el):
    """Readable heading: space-separate parts, drop the ¶ anchor symbol, tidy spaces."""
    t = el.get_text(" ", strip=True).replace("¶", "")
    return re.sub(r"\s+", " ", t).strip()


def get_page_urls():
    """Collect every tutorial page URL from the tutorial index's table of contents."""
    html = fetch(config.TUTORIAL_INDEX)
    soup = BeautifulSoup(html, "lxml")
    urls = []
    for a in soup.select("div.toctree-wrapper a.reference.internal"):
        href = a.get("href", "")
        if href.endswith(".html"):
            full = urljoin(config.BASE_URL, href.split("#")[0])
            if full not in urls:
                urls.append(full)
    # fallback: at least the index itself
    return urls or [config.TUTORIAL_INDEX]


def clean_text(node):
    """Readable text from a section, keeping code blocks as fenced blocks."""
    parts = []
    for el in node.find_all(["p", "pre", "li"], recursive=True):
        if el.name == "pre":
            parts.append("```\n" + el.get_text().strip() + "\n```")
        else:
            t = el.get_text(" ", strip=True)
            if t:
                parts.append(t)
    text = "\n\n".join(parts)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def window_split(text, size, overlap):
    """Split long text into overlapping windows, breaking on paragraph boundaries."""
    if len(text) <= size:
        return [text]
    paras = text.split("\n\n")
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= size:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            # start new window, carry a little overlap from the end of the last one
            tail = cur[-overlap:] if cur else ""
            cur = (tail + "\n\n" + p).strip() if tail else p
    if cur:
        chunks.append(cur)
    return chunks


def chunk_page(url):
    """Parse one tutorial page into section-based chunks with metadata."""
    html = fetch(url)
    soup = BeautifulSoup(html, "lxml")
    out = []
    page_title_el = soup.find(["h1"])
    page_title = clean_heading(page_title_el) if page_title_el else url

    for sec in soup.select("section[id]"):
        heading = sec.find(["h1", "h2", "h3", "h4"])
        if not heading:
            continue
        title = clean_heading(heading)
        anchor = sec.get("id")
        sec_url = f"{url}#{anchor}"
        # only this section's own content (exclude nested subsections' text to avoid duplication)
        for child in sec.find_all("section"):
            child.extract()
        text = clean_text(sec)
        if len(text) < 40:   # skip near-empty sections
            continue
        for i, piece in enumerate(window_split(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            out.append({
                "text": piece,
                "section_title": title,
                "url": sec_url,
                "heading_path": f"{page_title} > {title}",
                "part": i,
            })
    return out


def main():
    os.makedirs(os.path.dirname(config.CHUNKS_PATH), exist_ok=True)
    urls = get_page_urls()
    print(f"Found {len(urls)} tutorial pages.")
    all_chunks = []
    for u in urls:
        try:
            chunks = chunk_page(u)
            all_chunks.extend(chunks)
            print(f"  {u.split('/')[-1]:<22} -> {len(chunks)} chunks")
            time.sleep(0.3)   # be polite to the server
        except Exception as e:
            print(f"  FAILED {u}: {e}")

    for idx, c in enumerate(all_chunks):
        c["id"] = f"chunk_{idx:04d}"

    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_chunks)} chunks to {config.CHUNKS_PATH}")
    print(f"Unique sections: {len(set(c['section_title'] for c in all_chunks))}")


if __name__ == "__main__":
    main()
