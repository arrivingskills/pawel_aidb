import argparse
import random
import textwrap
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import feedparser
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

@dataclass
class NewsItem:
    title: str
    url: str
    summary: str = ""
    content: str = ""  # full fetched article text

DEFAULT_FEEDS = [
    # Major English-language outlets (RSS endpoints)
    "http://feeds.bbci.co.uk/news/rss.xml",  # BBC
    "https://www.reutersagency.com/feed/?best-topics=top-news&post_type=best",  # Reuters Top News
    "https://www.aljazeera.com/xml/rss/all.xml",  # Al Jazeera
    "https://www.npr.org/rss/rss.php?id=1001",  # NPR Top Stories
    "https://apnews.com/hub/apf-topnews?utm_source=apnews.com&utm_medium=referral&utm_campaign=apnews_rss",  # AP Top News
    "https://www.dw.com/en/top-stories/s-9097/rss",  # DW
]


def fetch_rss_entries(feed_urls: Sequence[str], limit_per_feed: int = 20) -> List[NewsItem]:
    """
    Download and parse RSS feeds. Returns a list of NewsItem with title, link, and summary.

    We gather a modest number of items from several feeds to later select a random one.
    """
    items: List[NewsItem] = []
    for url in feed_urls:
        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries[:limit_per_feed]:
                title = getattr(entry, "title", "(no title)")
                link = getattr(entry, "link", "")
                summary = getattr(entry, "summary", "")
                if link:
                    items.append(NewsItem(title=title, url=link, summary=summary))
        except Exception:
            # tolerate individual feed errors
            continue
    return items


def pick_random_item(items: Sequence[NewsItem]) -> Optional[NewsItem]:
    if not items:
        return None
    return random.choice(list(items))

def fetch_article_text(url: str, timeout: int = 15) -> str:
    """
    Fetch the article HTML and extract main text with a simple heuristic using BeautifulSoup.

    Note: This is intentionally lightweight. For production-grade extraction, you may consider
    libraries like newspaper3k or trafilatura. Here we stick to widely available dependencies.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; pawel-aidb/0.1)"}
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
    except Exception:
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # Try common containers first
    for selector in [
        "article",
        "div[itemprop='articleBody']",
        "div.m-article__body",
        "div.article-body",
        "div#content",
        "main",
    ]:
        node = soup.select_one(selector)
        if node:
            text = node.get_text(" ", strip=True)
            if len(text.split()) > 80:
                return text

    # Fallback: concatenate paragraph texts
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = " ".join([p for p in paragraphs if p])
    return text


def normalize_text(text: str) -> str:
    # Minimal normalization. TF–IDF will handle tokenization/stop-words internally.
    return " ".join(text.replace("\n", " ").split())


class FakeNewsDetector:
    """
    Lightweight text classifier for fake-news likelihood using classical NLP:

    Pipeline: TF–IDF vectorizer -> Logistic Regression classifier.

    Why this approach here?
    - Efficient: small and fast, no GPUs needed. Good for a demo CLI.
    - Interpretable: we can inspect coefficients to explain which words push
      the prediction toward "fake" vs "real".

    Training data in this demo:
    - A tiny, embedded sample (few dozen short texts) purely for demonstration.
      In real usage, replace `train_on_embedded()` with `fit(X, y)` on a proper
      labeled dataset (e.g., Kaggle Fake News, LIAR, GossipCop/PolitiFact).

    IMPORTANT: With such a small synthetic dataset, predictions are illustrative
    and not production-grade. The code is structured so you can plug in a better
    dataset without changing the CLI.
    """

    def __init__(self) -> None:
        # TF–IDF turns raw text into sparse numeric features based on term frequency,
        # scaled by inverse document frequency to down-weight common words.
        # Logistic Regression is a linear classifier producing calibrated probabilities.
        self.model: Pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        stop_words="english",
                        ngram_range=(1, 2),  # include unigrams and bigrams for context
                        min_df=2,  # terms must appear in at least 2 docs to reduce noise
                        max_features=20000,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(max_iter=200, class_weight="balanced"),
                ),
            ]
        )

        self._is_trained = False

    def read_data(self, inputs):
        news = {}
        for path_input in inputs:
            with open(path_input, "r") as f:
                contents = f.read()
                contents.split("\n")
                news[f.name] = contents
        return news

    def train_on_embedded(self, verbose: bool = False) -> None:
        """
        Train on a small, embedded sample.

        The examples below are short paraphrases intended for demo only. Replace
        with your actual dataset for meaningful accuracy.
        """
        news = self.read_data([r"data\Fake\compressedFake.csv",r"data\True\compressedTrue.csv"])
        print(news)
        X: List[str] = news['compressedFake']+news['compressedTrue']
        y: List[int] = [[0]*news['compressedTrue'], [1]*news['compressedFake']]
        # 0 = real
        # 1 = fake

        # Split for a tiny internal check (still not representative)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)
        self._is_trained = True

        if verbose:
            preds = self.model.predict(X_test)
            print("\n[Embedded training quality check — NOT a benchmark]")
            print(classification_report(y_test, preds, target_names=["real", "fake"]))

    def predict_proba(self, texts: Sequence[str]) -> List[float]:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        # Probability of class 1 (fake)
        probs = self.model.predict_proba(list(texts))
        class_index = list(self.model.classes_).index(1)
        return [float(p[class_index]) for p in probs]

    def explain_top_features(self, text: str, top_k: int = 10) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Provide simple, local explanation: which n-grams in this text push toward fake vs real.

        Mechanism:
        - Extract the TF–IDF vector for this specific text.
        - Multiply each active feature by the logistic regression coefficient
          for the fake class. Positive weights push toward fake; negative toward real.

        Returns two lists of (feature, weight):
        - top_fake: features with strongest positive contribution
        - top_real: features with strongest negative contribution
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before explanation.")

        # Unpack the steps
        tfidf: TfidfVectorizer = self.model.named_steps["tfidf"]
        clf: LogisticRegression = self.model.named_steps["clf"]

        # Vectorize single document
        vec = tfidf.transform([text])  # scipy sparse

        # Get feature names aligned with columns
        feat_names = tfidf.get_feature_names_out()

        # Get coefficients for class 1 (fake). LogisticRegression.coef_ shape: [n_classes, n_features]
        # For binary, coef_[0] corresponds to class 1 vs class 0.
        coefs = clf.coef_[0]

        # Compute contribution = value * coefficient for nonzero entries only
        # vec is CSR format; iterate through indices
        contributions: List[Tuple[str, float]] = []
        indptr = vec.indptr
        indices = vec.indices
        data = vec.data
        for i in range(indptr[0], indptr[1]):
            j = indices[i]
            val = data[i]
            contributions.append((feat_names[j], float(val * coefs[j])))

        # Split into those pushing fake (positive) and real (negative)
        pos = sorted([(f, w) for f, w in contributions if w > 0], key=lambda x: x[1], reverse=True)[:top_k]
        neg = sorted([(f, -w) for f, w in contributions if w < 0], key=lambda x: x[1], reverse=True)[:top_k]
        return pos, neg


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pawel-aidb",
        description=(
            "Fetch a random news article from popular RSS feeds and assess the\n"
            "likelihood that the text is fake using a small, interpretable AI model.\n\n"
            "NOTE: Model is trained on a tiny embedded dataset for demonstration only."
        ),
    )

    parser.add_argument(
        "--feeds",
        nargs="*",
        default=DEFAULT_FEEDS,
        help="RSS feed URLs to sample from (default: a built-in list of major outlets)",
    )
    parser.add_argument(
        "--limit-per-feed",
        type=int,
        default=20,
        help="Max items to read per feed during sampling (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--verbose-train",
        action="store_true",
        help="Print a tiny internal quality check after embedded training",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.seed is not None:
        random.seed(args.seed)

    print("Collecting news items from RSS feeds...")
    items = fetch_rss_entries(args.feeds, limit_per_feed=args.limit_per_feed)
    item = pick_random_item(items)
    if not item:
        print("Could not collect any news items. Try different feeds or check your internet connection.")
        return 2

    print(f"Selected: {item.title}")
    print(f"URL: {item.url}")

    article_text = fetch_article_text(item.url)
    combined_text = normalize_text(" ".join([item.title or "", item.summary or "", article_text or ""]))
    if not combined_text or len(combined_text.split()) < 30:
        print("Warning: Unable to extract substantial article text; assessment may be unreliable.")

    # Train and predict
    detector = FakeNewsDetector()
    detector.train_on_embedded(verbose=args.verbose_train)
    prob_fake = detector.predict_proba([combined_text])[0]

    # Provide a small local explanation
    top_fake, top_real = detector.explain_top_features(combined_text, top_k=8)

    print("\n--- AI Assessment (Demo) ---")
    print(f"Fake likelihood (0.0–1.0): {prob_fake:.3f}")
    label = "FAKE" if prob_fake >= 0.5 else "REAL"
    print(f"Predicted label: {label}")

    def fmt_feats(feats: List[Tuple[str, float]]) -> str:
        return ", ".join([f"{f} ({w:+.3f})" for f, w in feats]) or "(none)"

    print("\nFeatures pushing toward FAKE (local):")
    print(textwrap.fill(fmt_feats(top_fake), width=100))

    print("\nFeatures pushing toward REAL (local):")
    print(textwrap.fill(fmt_feats(top_real), width=100))

    print(
        "\nNote: This model is intentionally tiny and trained on a synthetic sample.\n"
        "For serious use, train on a large labeled dataset and consider more\n"
        "robust content extraction."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())