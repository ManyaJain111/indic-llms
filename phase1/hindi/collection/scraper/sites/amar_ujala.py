"""
Amar Ujala scraper — https://www.amarujala.com
One of India's largest Hindi daily newspapers.
"""

import logging
from typing import Optional

from ..base_scraper import Article, BaseScraper

logger = logging.getLogger(__name__)


class AmarUjalaScraper(BaseScraper):
    """Scraper for Amar Ujala (amarujala.com)."""

    SOURCE = "Amar Ujala"

    def get_article_links(self, category_url: str) -> list[str]:
        """Extract article URLs from an Amar Ujala listing page."""
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
                # Exclude nav/search links that have ?src= or ?text= query params
                # Valid AU article URLs: /category/slug-YYYY-MM-DD (no ?src=)
                if (
                    self.is_same_domain(self.base_url, full_url)
                    and full_url != category_url
                    and "?src=" not in full_url
                    and "?text=" not in full_url
                    and len(full_url) > len(self.base_url) + 15
                    and full_url.count("/") >= 4  # at least /category/slug
                ):
                    links.add(full_url)

        # Pagination — Amar Ujala uses ?page=N
        for page_num in [2, 3]:
            sep = "&" if "?" in category_url else "?"
            page_url = f"{category_url}{sep}page={page_num}"
            self._sleep()
            page_soup = self.fetch_page(page_url)
            if not page_soup:
                break
            new_links_found = False
            for sel in [s.strip() for s in selectors.split(",")]:
                for tag in page_soup.select(sel):
                    href = tag.get("href", "")
                    if not href:
                        continue
                    full_url = self.resolve_url(self.base_url, href)
                    if (
                        self.is_same_domain(self.base_url, full_url)
                        and full_url not in links
                        and "?src=" not in full_url
                        and "?text=" not in full_url
                        and len(full_url) > len(self.base_url) + 15
                        and full_url.count("/") >= 4
                    ):
                        links.add(full_url)
                        new_links_found = True
            if not new_links_found:
                break

        return list(links)

    def parse_article(self, url: str) -> Optional[Article]:
        """Parse a single Amar Ujala article."""
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

        # Amar Ujala sometimes uses <time datetime="..."> for structured date
        if not date:
            time_tag = soup.find("time")
            if time_tag:
                date = time_tag.get("datetime", "") or time_tag.get_text(strip=True)

        # Fallback body extraction
        if not body:
            for cls in ["detail-body", "article-body", "article-content"]:
                container = soup.find(class_=lambda c: c and cls in c.split())
                if container:
                    body = self.clean_text(container.get_text(separator="\n\n", strip=True))
                    break

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
