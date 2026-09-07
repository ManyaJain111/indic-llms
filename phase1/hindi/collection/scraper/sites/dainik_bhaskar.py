"""
Dainik Bhaskar scraper — https://www.bhaskar.com
India's largest Hindi newspaper by circulation.
"""

import logging
from typing import Optional
from urllib.parse import urljoin

from ..base_scraper import Article, BaseScraper

logger = logging.getLogger(__name__)


class BhaskarScraper(BaseScraper):
    """Scraper for Dainik Bhaskar (bhaskar.com)."""

    SOURCE = "Dainik Bhaskar"

    def get_article_links(self, category_url: str) -> list[str]:
        """Extract article URLs from a Bhaskar category page."""
        soup = self.fetch_page(category_url)
        if not soup:
            return []

        selectors = self.site_cfg["selectors"]["article_links"]
        links = set()

        for sel in [s.strip() for s in selectors.split(",")]:
            for tag in soup.select(sel):
                href = tag.get("href", "")
                if not href:
                    continue
                full_url = self.resolve_url(self.base_url, href)
                # Bhaskar articles have paths like /category/city/news-title/
                if (
                    self.is_same_domain(self.base_url, full_url)
                    and "/news/" in full_url
                    and full_url != category_url
                ):
                    links.add(full_url)

        # Also try to scrape a second page (pagination)
        for page_num in [2, 3]:
            page_url = f"{category_url}?page={page_num}"
            self._sleep()
            page_soup = self.fetch_page(page_url)
            if not page_soup:
                break
            for sel in [s.strip() for s in selectors.split(",")]:
                for tag in page_soup.select(sel):
                    href = tag.get("href", "")
                    if not href:
                        continue
                    full_url = self.resolve_url(self.base_url, href)
                    if (
                        self.is_same_domain(self.base_url, full_url)
                        and "/news/" in full_url
                        and full_url not in links
                    ):
                        links.add(full_url)

        return list(links)

    def parse_article(self, url: str) -> Optional[Article]:
        """Parse a single Bhaskar article page into an Article object."""
        soup = self.fetch_page(url)
        if not soup:
            return None

        sel = self.site_cfg["selectors"]

        headline = self.clean_text(self.extract_text(soup, sel["headline"]))
        body = self.clean_text(self.extract_all_text(soup, sel["body"]))
        author = self.clean_text(self.extract_text(soup, sel["author"]))
        date = self.clean_text(self.extract_text(soup, sel["date"]))
        category = self.clean_text(self.extract_text(soup, sel["category"]))
        tags = [self.clean_text(t) for t in self.extract_list(soup, sel["tags"])]

        # Fallback: try to grab all <p> in article if body is empty
        if not body:
            article_tag = soup.find("article")
            if article_tag:
                body = self.clean_text(
                    article_tag.get_text(separator="\n\n", strip=True)
                )

        return Article(
            url=url,
            source=self.SOURCE,
            headline=headline,
            body=body,
            author=author,
            published_date=date,
            category=category,
            tags=[t for t in tags if t],
        )
