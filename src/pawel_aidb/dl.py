from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
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
    text = " ".join(text.split())
    return text


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
                text = " ".join(text.split())
                return (resp.url, text)

            text = _extract_visible_text(resp.text)
            return (resp.url, text)
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
) -> Tuple[Dict[str, str], Dict[str, str]]:

    urls_list: List[str] = list(urls)
    opts = FetchOptions(timeout=timeout, retries=retries, backoff=backoff, user_agent=user_agent)
    texts: Dict[str, str] = {}
    errors: Dict[str, str] = {}

    if not urls_list:
        return texts, errors

    session = _build_session(opts.user_agent)

    def task(original_url: str):
        try:
            final_url, text = _fetch_one(original_url, session=session, opts=opts)
            texts[original_url] = text
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
print(texts)
