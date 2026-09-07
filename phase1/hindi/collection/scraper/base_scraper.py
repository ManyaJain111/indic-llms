"""
Hindi News Scraper — Base Scraper
Provides the abstract base class that all site-specific scrapers inherit from.
"""

import hashlib
import logging
import random
import time
import unicodedata
from abc import ABC, abstractmethod
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# Desktop-only User-Agent pool (mobile UAs cause sites to serve stripped mobile pages)
_DESKTOP_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
]


def _random_ua() -> str:
    """Return a random desktop-only User-Agent string."""
    return random.choice(_DESKTOP_UAS)





# ── Article data model ─────────────────────────────────────────────────────────
class Article:
    """Structured container for a scraped news article."""

    def __init__(
        self,
        url: str,
        source: str,
        headline: str = "",
        body: str = "",
        author: str = "",
        published_date: str = "",
        category: str = "",
        tags: list[str] = None,
    ):
        self.url = url
        self.url_hash = hashlib.sha256(url.encode()).hexdigest()
        self.content_hash = hashlib.sha256(body.encode()).hexdigest() if body else ""
        self.source = source
        self.headline = headline
        self.body = body
        self.author = author
        self.published_date = published_date
        self.category = category
        self.tags = tags or []

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "url_hash": self.url_hash,
            "content_hash": self.content_hash,
            "source": self.source,
            "headline": self.headline,
            "body": self.body,
            "author": self.author,
            "published_date": self.published_date,
            "category": self.category,
            "tags": self.tags,
        }

    def is_valid(self) -> bool:
        """Returns True if the article has at minimum a headline and non-empty body."""
        return bool(self.headline.strip()) and len(self.body.strip()) > 50

    def has_hindi(self) -> bool:
        """Returns True if the article body contains Devanagari script characters."""
        return any("\u0900" <= ch <= "\u097F" for ch in self.body + self.headline)


# ── Base scraper ───────────────────────────────────────────────────────────────
class BaseScraper(ABC):
    """
    Abstract base class for all site scrapers.

    Subclasses must implement:
      - get_article_links(category_url) -> list[str]
      - parse_article(url) -> Optional[Article]
    """

    def __init__(self, site_config: dict, global_config: dict):
        self.site_cfg = site_config
        self.global_cfg = global_config
        self.site_name = site_config["name"]
        self.base_url = site_config["base_url"]
        self.delay_min = site_config.get("delay", global_config.get("request_delay_min", 1.5))
        self.delay_max = global_config.get("request_delay_max", 3.0)
        self.timeout = global_config.get("timeout", 15)
        self.session = self._build_session()

    # ── Session setup ──────────────────────────────────────────────────────────
    def _build_session(self) -> requests.Session:
        sess = requests.Session()
        sess.headers.update(self._build_headers())
        return sess

    def _build_headers(self) -> dict:
        return {
            "User-Agent": _random_ua(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "hi,en-US;q=0.7,en;q=0.3",
            # Intentionally omit 'br' (Brotli) — requests does not natively
            # decompress Brotli, and sites returning it would yield garbled HTML
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    def _rotate_ua(self):
        """Rotate user-agent between requests."""
        self.session.headers.update({"User-Agent": _random_ua()})

    # ── Polite delay ───────────────────────────────────────────────────────────
    def _sleep(self):
        delay = random.uniform(self.delay_min, self.delay_max)
        logger.debug(f"[{self.site_name}] Sleeping {delay:.2f}s")
        time.sleep(delay)

    # ── HTTP fetch with retry ──────────────────────────────────────────────────
    @retry(
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch a URL and return a BeautifulSoup object.
        Returns None on non-200 status or parse failure.
        """
        self._rotate_ua()
        try:
            logger.debug(f"[{self.site_name}] GET {url}")
            resp = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            if resp.status_code == 200:
                resp.encoding = resp.apparent_encoding or "utf-8"
                return BeautifulSoup(resp.text, "lxml")
            elif resp.status_code == 429:
                logger.warning(f"[{self.site_name}] Rate-limited (429) on {url}, backing off...")
                time.sleep(random.uniform(10, 20))
                return None
            elif resp.status_code in (403, 404):
                logger.debug(f"[{self.site_name}] HTTP {resp.status_code} for {url}")
                return None
            else:
                logger.warning(f"[{self.site_name}] HTTP {resp.status_code} for {url}")
                return None
        except requests.Timeout:
            logger.warning(f"[{self.site_name}] Timeout fetching {url}")
            raise
        except requests.ConnectionError as e:
            logger.warning(f"[{self.site_name}] Connection error: {e}")
            raise

    # ── Text utilities ─────────────────────────────────────────────────────────
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Normalize whitespace, remove zero-width characters,
        and apply Unicode NFC normalization for Devanagari.
        """
        if not text:
            return ""
        # NFC normalization for proper Devanagari rendering
        text = unicodedata.normalize("NFC", text)
        # Remove zero-width / invisible chars
        text = "".join(ch for ch in text if unicodedata.category(ch) not in ("Cf",))
        # Normalize whitespace
        text = " ".join(text.split())
        return text.strip()

    @staticmethod
    def extract_text(soup: BeautifulSoup, selector: str) -> str:
        """Try multiple comma-separated CSS selectors, return first match text."""
        for sel in [s.strip() for s in selector.split(",")]:
            el = soup.select_one(sel)
            if el:
                return el.get_text(separator=" ", strip=True)
        return ""

    @staticmethod
    def extract_all_text(soup: BeautifulSoup, selector: str) -> str:
        """Concatenate text from all matching elements (for article body paragraphs)."""
        paragraphs = []
        for sel in [s.strip() for s in selector.split(",")]:
            els = soup.select(sel)
            if els:
                paragraphs = [el.get_text(separator=" ", strip=True) for el in els]
                break
        return "\n\n".join(p for p in paragraphs if p)

    @staticmethod
    def extract_list(soup: BeautifulSoup, selector: str) -> list[str]:
        """Return list of text from all matching elements (e.g., tags)."""
        results = []
        for sel in [s.strip() for s in selector.split(",")]:
            els = soup.select(sel)
            if els:
                results = [el.get_text(strip=True) for el in els if el.get_text(strip=True)]
                break
        return results

    @staticmethod
    def resolve_url(base: str, href: str) -> str:
        """Resolve relative URLs against the site base URL."""
        if href.startswith("http"):
            return href
        return urljoin(base, href)

    @staticmethod
    def is_same_domain(base_url: str, url: str) -> bool:
        """Check that url is on the same domain as base_url."""
        base_host = urlparse(base_url).netloc.replace("www.", "")
        url_host = urlparse(url).netloc.replace("www.", "")
        return base_host in url_host

    # ── Abstract methods ───────────────────────────────────────────────────────
    @abstractmethod
    def get_article_links(self, category_url: str) -> list[str]:
        """
        Given a category/listing page URL, return a list of article URLs.
        """
        raise NotImplementedError

    @abstractmethod
    def parse_article(self, url: str) -> Optional[Article]:
        """
        Fetch and parse a single article page.
        Returns an Article instance or None on failure.
        """
        raise NotImplementedError

    # ── Main scrape entry point ────────────────────────────────────────────────
    def scrape(self, limit: Optional[int] = None, seen_url_hashes: Optional[set] = None) -> list[Article]:
        """
        Scrape all configured categories for this site.
        Returns a list of valid Article objects.
        """
        articles = []
        categories = self.site_cfg.get("categories", [])

        for cat in categories:
            cat_url = cat["url"]
            cat_name = cat["name"]
            logger.info(f"[{self.site_name}] Scraping category: {cat_name} ({cat_url})")

            try:
                links = self.get_article_links(cat_url)
                logger.info(f"[{self.site_name}] Found {len(links)} article links in '{cat_name}'")
            except Exception as e:
                logger.error(f"[{self.site_name}] Failed to get links from {cat_url}: {e}")
                continue

            for link in links:
                if limit and len(articles) >= limit:
                    logger.info(f"[{self.site_name}] Reached limit of {limit} articles.")
                    return articles

                link_hash = hashlib.sha256(link.encode()).hexdigest()
                if seen_url_hashes and link_hash in seen_url_hashes:
                    logger.debug(f"[{self.site_name}] Skipping already seen URL: {link}")
                    continue

                self._sleep()
                try:
                    article = self.parse_article(link)
                    if article:
                        article.category = article.category or cat_name
                        if article.is_valid() and article.has_hindi():
                            articles.append(article)
                            logger.info(
                                f"[{self.site_name}] ✓ {article.headline[:60]}..."
                                if len(article.headline) > 60
                                else f"[{self.site_name}] ✓ {article.headline}"
                            )
                        else:
                            logger.debug(
                                f"[{self.site_name}] Skipped (invalid/no Hindi): {link}"
                            )
                except Exception as e:
                    logger.error(f"[{self.site_name}] Error parsing {link}: {e}")
                    continue

        return articles
