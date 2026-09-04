"""
Hindustan scraper — https://www.livehindustan.com
Hindi daily from Hindustan Media Ventures (HT Group).
"""

import logging
import re
from typing import Optional

from ..base_scraper import Article, BaseScraper

logger = logging.getLogger(__name__)


class HindustanScraper(BaseScraper):
    """Scraper for Live Hindustan (livehindustan.com)."""

    SOURCE = "Hindustan"

    def get_article_links(self, category_url: str) -> list[str]:
        """Extract article links from a Hindustan listing page."""
        soup = self.fetch_page(category_url)
        if not soup:
            return []

        selectors = self.site_cfg["selectors"]["article_links"]
        links = set()

        def _is_hindustan_article(url: str) -> bool:
            """
            Hindustan article URLs end in a 15+ digit numeric ID followed by .html
            e.g. /national/big-move-...-201786640012031.html
            """
            return bool(re.search(r"-\d{15,}\.html$", url))

        for sel in [s.strip() for s in selectors.split(",")]:
            for tag in soup.select(sel):
                href = tag.get("href", "")
                if not href:
                    continue
                full_url = self.resolve_url(self.base_url, href)
                if (
                    self.is_same_domain(self.base_url, full_url)
                    and _is_hindustan_article(full_url)
                    and full_url != category_url
                ):
                    links.add(full_url)

        # Hindustan uses /page/N/ path pagination
        for page_num in [2, 3]:
            # Strip trailing slash, then append page
            base = category_url.rstrip("/")
            page_url = f"{base}/page/{page_num}/"
            self._sleep()
            page_soup = self.fetch_page(page_url)
            if not page_soup:
                break
            new_found = False
            for sel in [s.strip() for s in selectors.split(",")]:
                for tag in page_soup.select(sel):
                    href = tag.get("href", "")
                    if not href:
                        continue
                    full_url = self.resolve_url(self.base_url, href)
                    if (
                        self.is_same_domain(self.base_url, full_url)
                        and _is_hindustan_article(full_url)
                        and full_url not in links
                    ):
                        links.add(full_url)
                        new_found = True
            if not new_found:
                break

        return list(links)

    def parse_article(self, url: str) -> Optional[Article]:
        """Parse a single Hindustan article page."""
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

        # Fallback body
        if not body:
            for cls_fragment in ["detail-body", "story-details", "article-body"]:
                container = soup.find(class_=lambda c: c and cls_fragment in c.split())
                if container:
                    body = self.clean_text(
                        "\n\n".join(
                            p.get_text(strip=True) for p in container.find_all("p")
                            if p.get_text(strip=True)
                        )
                    )
                    break

        # Fallback date
        if not date:
            time_el = soup.find("time")
            if time_el:
                date = time_el.get("datetime", "") or time_el.get_text(strip=True)

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
