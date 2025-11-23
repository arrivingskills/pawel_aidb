import re
import sys
from typing import Iterable, Optional

import requests
from bs4 import BeautifulSoup, NavigableString, Tag


URLS = [
    "https://thepeoplesvoice.tv/",
    "https://www.justfactsdaily.com/50-examples-of-fake-news-in-2024",
]


def fetch_html(url: str, timeout: int = 20) -> Optional[str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/119.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = resp.encoding or resp.apparent_encoding
        return resp.text
    except Exception as e:
        print(f"[newsScraper] Failed to fetch {url}: {e}", file=sys.stderr)
        return None


def _strip_boilerplate(soup: BeautifulSoup) -> None:
    for sel in [
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "nav",
        "footer",
        "header",
        "aside",
        "form",
    ]:
        for tag in soup.select(sel):
            tag.decompose()


def _best_container(candidates: Iterable[Tag]) -> Optional[Tag]:
    best = None
    best_score = -1
    for node in candidates:
        # Score by number of <p> tags and text length
        p_tags = node.find_all("p")
        text_len = sum(len(p.get_text(strip=True)) for p in p_tags)
        score = len(p_tags) * 100 + text_len
        if score > best_score:
            best_score = score
            best = node
    return best


def _is_hidden(tag: Tag) -> bool:
    """Heuristic to skip visually hidden elements."""
    if tag.has_attr("hidden"):
        return True
    if tag.get("aria-hidden") == "true":
        return True
    style = tag.get("style", "")
    if isinstance(style, str) and ("display:none" in style.replace(" ", "").lower() or "visibility:hidden" in style.replace(" ", "").lower()):
        return True
    return False


_EN_STOPWORDS = {
    "the",
    "and",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "is",
    "are",
    "this",
    "that",
    "a",
    "an",
    "as",
    "at",
    "by",
    "from",
    "it",
    "be",
}


def _ascii_letter_ratio(text: str) -> float:
    letters = sum(ch.isalpha() for ch in text)
    ascii_letters = sum(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in text)
    if letters == 0:
        return 0.0
    return ascii_letters / letters


def _looks_english(text: str, soup: BeautifulSoup) -> bool:
    # Prefer explicit html lang
    doc_lang = ""
    if soup.html and soup.html.has_attr("lang"):
        doc_lang = str(soup.html.get("lang") or "").lower()
    if doc_lang.startswith("en"):
        # If declared English, accept unless text is clearly non-English
        return _ascii_letter_ratio(text) >= 0.5

    # Otherwise, apply heuristics
    ratio = _ascii_letter_ratio(text)
    if ratio < 0.6:
        return False
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    stop_hits = sum(tok in _EN_STOPWORDS for tok in tokens)
    return stop_hits >= 1 or ratio >= 0.85


def extract_headings_text(html: str, url: str = "") -> str:
    """Extract only heading texts (h1–h6) that look like English, ordered and deduplicated."""
    soup = BeautifulSoup(html, "html.parser")
    _strip_boilerplate(soup)

    headings: list[str] = []
    seen: set[str] = set()
    for level in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        for h in soup.find_all(level):
            if _is_hidden(h):
                continue
            text = h.get_text(" ", strip=True)
            if not text:
                continue
            # Basic noise filters
            text = re.sub(r"\s+", " ", text).strip()
            # Drop very short or excessively long headings
            if len(text) < 3 or len(text) > 160:
                continue
            # Drop menu-like or boilerplate items
            low = text.lower()
            if any(k in low for k in [
                "cookie",
                "newsletter",
                "subscribe",
                "privacy",
                "terms",
                "related posts",
                "read more",
                "advert",
                "comments",
            ]):
                continue
            if not _looks_english(text, soup):
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            headings.append(text)

    # Build output
    body = "\n".join(headings) if headings else "(no English headings found)"
    if url:
        body = f"Source: {url}\n" + body
    return body


def main(urls: Iterable[str]) -> None:
    outputs: list[str] = []
    for url in urls:
        html = fetch_html(url)
        if not html:
            continue
        heading_text = extract_headings_text(html, url=url)
        if heading_text:
            outputs.append(heading_text)

    if not outputs:
        print("[newsScraper] No articles extracted.", file=sys.stderr)
        return

    with open("articles.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(outputs) + "\n")


if __name__ == "__main__":
    main(URLS)