from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import re
import time

import requests
from bs4 import BeautifulSoup, Comment


DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/119.0.0.0 Safari/537.36"
)


@dataclass
class FetchOptions:
    timeout: int = 15
    retries: int = 2
    backoff: float = 0.6  # seconds, exponential backoff
    user_agent: str = DEFAULT_UA


def _build_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s


def _extract_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()

    for tag in soup([
        "script",
        "style",
        "noscript",
        "template",
        "iframe",
        "canvas",
        "svg",
        "meta",
        "link",
        "header",
        "footer",
        "nav",
        "aside",
        "form",
        "button",
        "input",
    ]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = _sanitize_text(text)
    return text


def _sanitize_text(text: str) -> str:
    """Normalize whitespace and remove specific unwanted characters.

    Currently removes the bullet character '•' and then collapses whitespace.
    """
    # Replace the bullet with a space so words don't concatenate, then collapse spaces
    text = text.replace("•", " ")
    text = " ".join(text.split())
    return text


def _split_news_items(text: str) -> List[str]:
    """Split a raw page text into a list of individual news items.

    This is a heuristic splitter suitable for typical news homepages:
    - First split on line breaks and common inline separators like "|" or bullets.
    - Then split remaining chunks into sentences by punctuation boundaries.
    - Filter out overly short fragments and deduplicate while preserving order.
    """
    if not text:
        return []

    # Primary split on newlines or common inline separators
    primary_parts = re.split(r"[\n\r]+|\s[|•]\s", text)

    chunks: List[str] = []
    for part in primary_parts:
        p = part.strip()
        if not p:
            continue
        # Secondary split on sentence boundaries: ., ?, ! followed by space and capital/number
        chunks.extend(re.split(r"(?<=[\.\?\!])\s+(?=[A-Z0-9])", p))

    items: List[str] = []
    for ch in chunks:
        cc = " ".join(ch.split())
        # Keep fragments that look like actual items (length/word-count heuristic)
        if len(cc) >= 40 or len(cc.split()) >= 6:
            items.append(cc)

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: List[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            result.append(it)
    return result


def _fetch_one(url: str, session: requests.Session, opts: FetchOptions) -> Tuple[str, str]:
    attempt = 0
    last_exc: Exception | None = None
    while attempt <= opts.retries:
        try:
            resp = session.get(url, timeout=opts.timeout, allow_redirects=True)
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "").lower()
            if "text/html" not in ctype and "application/xhtml+xml" not in ctype and \
               not ctype.startswith("text/"):
                text = resp.text
                text = _sanitize_text(text)
                return resp.url, text

            text = _extract_visible_text(resp.text)
            return resp.url, text
        except Exception as e:  # noqa: BLE001 – retry on any error
            last_exc = e
            if attempt >= opts.retries:
                break
            sleep_for = opts.backoff * (2 ** attempt)
            time.sleep(sleep_for)
            attempt += 1
    assert last_exc is not None
    raise last_exc


def download_pages_text(
    urls: Iterable[str],
    *,
    concurrency: int = 8,
    timeout: int = 15,
    retries: int = 2,
    backoff: float = 0.6,
    user_agent: str = DEFAULT_UA,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:

    urls_list: List[str] = list(urls)
    opts = FetchOptions(timeout=timeout, retries=retries, backoff=backoff, user_agent=user_agent)
    texts: Dict[str, List[str]] = {}
    errors: Dict[str, str] = {}

    if not urls_list:
        return texts, errors

    session = _build_session(opts.user_agent)

    def task(original_url: str):
        try:
            final_url, text = _fetch_one(original_url, session=session, opts=opts)
            # CHANGED: store a list of individual news items instead of a single string
            texts[original_url] = _split_news_items(text)
        except Exception as e:  # noqa: BLE001
            errors[original_url] = f"{type(e).__name__}: {e}"

    # Concurrency guard
    max_workers = max(1, int(concurrency))
    if max_workers == 1 or len(urls_list) == 1:
        for u in urls_list:
            task(u)
        return texts, errors

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="dl") as ex:
        futures = {ex.submit(task, u): u for u in urls_list}
        for _ in as_completed(futures):
            # Results are stored in `texts` / `errors` within task()
            pass

    return texts, errors


__all__ = [
    "download_pages_text",
]

urls = [
    "https://cnn.com",
    "https://www.bbc.com/"
]
texts, errors = download_pages_text(urls, concurrency=8)

