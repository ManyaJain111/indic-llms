"""
Navbharat Times scraper — https://navbharattimes.indiatimes.com
Hindi digital news portal by Bennett Coleman (Times Group).
"""

import logging
from typing import Optional

from ..base_scraper import Article, BaseScraper

logger = logging.getLogger(__name__)


class NavbharatTimesScraper(BaseScraper):
    """Scraper for Navbharat Times (navbharattimes.indiatimes.com)."""

    SOURCE = "Navbharat Times"

    def get_article_links(self, category_url: str) -> list[str]:
        """Extract article links from an NBT category page."""
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
                # NBT articles use /articleshow/ in URL
                if (
                    self.is_same_domain(self.base_url, full_url)
                    and "articleshow" in full_url
                    and full_url != category_url
                ):
                    links.add(full_url)

        # NBT pagination via ?curpg=N
        for page_num in [2, 3]:
            sep = "&" if "?" in category_url else "?"
            page_url = f"{category_url}{sep}curpg={page_num}"
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
                        and "articleshow" in full_url
                        and full_url not in links
                    ):
                        links.add(full_url)
                        new_found = True
            if not new_found:
                break

        return list(links)

    def parse_article(self, url: str) -> Optional[Article]:
        """Parse a single Navbharat Times article page."""
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

        # NBT often stores full article text in a single div — try fallbacks
        if not body:
            for cls_fragment in ["articlestorycontent", "article_content", "artText"]:
                container = soup.find("div", class_=lambda c: c and cls_fragment in c)
                if container:
                    body = self.clean_text(
                        "\n\n".join(p.get_text(strip=True) for p in container.find_all("p") if p.get_text(strip=True))
                    )
                    break

        # Structured date from <time>
        if not date:
            time_el = soup.find("time")
            if time_el:
                date = time_el.get("datetime", time_el.get_text(strip=True))

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
