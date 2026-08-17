import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup

# Headers to mimic a real browser and request Hindi content
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "hi,hi-IN;q=0.9,en-US;q=0.8,en;q=0.7"
}

# --- CONFIGURATION ---
# PRID is a single global auto-incrementing ID across ALL PIB releases and
# languages -- it's not tied to a per-day or per-region listing. Rather than
# discovering IDs from the homepage's small "latest releases" widget (which
# only ever exposes ~15 items), we iterate a PRID range directly and let
# scrape_pib_hindi_release() skip anything invalid or not translated to Hindi.
#
# Set START_PRID to a recent, known-valid PRID (check today's homepage --
# https://www.pib.gov.in/indexd.aspx?reg=3&lang=2 -- for a current example)
# and NUM_TO_SCRAPE to how far back you want to walk.
START_PRID = 2299325
NUM_TO_SCRAPE = 2000   # walking backwards this many PRIDs from START_PRID


def clean_hindi_text(text: str) -> str:
    """Cleans raw scraped text while preserving Devanagari characters & punctuation."""
    if not text:
        return ""
    text = re.sub(r'[\r\t]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')


def is_hindi_text(text: str, min_devanagari_chars: int = 20) -> bool:
    """
    Heuristic Hindi check.

    PIB's `lang=2` query param does NOT reliably force Hindi content --
    each release is filed under a PRID tied to whatever language its
    originating regional office published it in, and consecutive PRIDs
    interleave across many languages (English, Gujarati, Assamese, etc.
    all showed up within a five-PRID stretch during testing). We verify
    actual Devanagari script content instead of trusting the URL param.
    """
    if not text:
        return False
    return len(_DEVANAGARI_RE.findall(text)) >= min_devanagari_chars


def scrape_pib_hindi_release(prid: int) -> dict | None:
    """
    Fetches and cleans a single PIB press release by PRID, returning it only
    if its actual content is in Hindi (verified via is_hindi_text -- the
    lang=2 URL param does not reliably control this). Returns None for
    invalid PRIDs, missing pages, or releases that turn out not to be Hindi.
    """
    url = f"https://pib.gov.in/PressReleasePage.aspx?PRID={prid}&lang=2"

    try:
        response = requests.get(url, headers=HEADERS, timeout=12)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.content, "html.parser")

        for element in soup(["script", "style", "nav", "footer", "header", "iframe"]):
            element.decompose()

        title_tag = soup.find("h2") or soup.find("div", class_="ReleaseTitle")
        title = clean_hindi_text(title_tag.get_text()) if title_tag else ""

        if not title:  # Invalid PRID, or no Hindi translation exists for it
            return None

        meta_div = soup.find("div", class_="ReleaseDate") or soup.find("div", class_="innner-content")
        meta_text = meta_div.get_text() if meta_div else ""

        date_match = re.search(r'प्रविष्टि तिथि:\s*([^\n]+)', meta_text)
        date_str = date_match.group(1).strip() if date_match else ""

        ministry_tag = soup.find("div", class_="MinistryName")
        ministry = clean_hindi_text(ministry_tag.get_text()) if ministry_tag else ""

        body_div = soup.find("div", class_="ReleaseText") or soup.find("div", id="PRDetail")
        if not body_div:
            body_div = soup.find("div", class_="innner-content")

        if body_div:
            for meta in body_div.find_all(class_=["ReleaseDate", "MinistryName", "social-share"]):
                meta.decompose()
            body_text = clean_hindi_text(body_div.get_text())
        else:
            body_text = ""

        # PRID + lang=2 is not a reliable Hindi filter (see is_hindi_text
        # docstring) -- verify the actual script content before accepting.
        if not is_hindi_text(title + " " + body_text):
            return None

        return {
            "prid": prid,
            "url": url,
            "title": title,
            "ministry": ministry,
            "date_posted": date_str,
            "body": body_text,
            "language": "hi"
        }

    except Exception as e:
        print(f"[x] Error fetching PRID {prid}: {e}")
        return None


# --- Main Execution ---
if __name__ == "__main__":
    prid_range = range(START_PRID, START_PRID - NUM_TO_SCRAPE, -1)
    print(f"[*] Walking PRIDs {START_PRID} down to {START_PRID - NUM_TO_SCRAPE + 1} "
          f"({NUM_TO_SCRAPE} candidates)...\n")

    scraped_data = []
    hits = 0
    misses = 0

    for prid in prid_range:
        data = scrape_pib_hindi_release(prid)
        if data:
            scraped_data.append(data)
            hits += 1
            print(f"    [OK {prid}] {data['title'][:50]}...")
        else:
            misses += 1
            print(f"    [--] PRID {prid}: no Hindi release", end="\r")

        # Respectful delay between requests
        time.sleep(1.5)

        # Periodic checkpoint so a long run isn't all-or-nothing
        if (hits + misses) % 50 == 0:
            with open("pib_hindi_releases.json", "w", encoding="utf-8") as f:
                json.dump(scraped_data, f, ensure_ascii=False, indent=2)
            print(f"\n[*] Checkpoint: {hits} hits / {misses} misses so far -- saved.")

    with open("pib_hindi_releases.json", "w", encoding="utf-8") as f:
        json.dump(scraped_data, f, ensure_ascii=False, indent=2)

    print(f"\n[*] Finished! {hits} Hindi releases scraped, {misses} PRIDs had none.")
    print(f"[*] Saved to pib_hindi_releases.json")