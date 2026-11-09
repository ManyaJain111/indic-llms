

import argparse
import bz2
import json
import re
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import mwparserfromhell
import requests


RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")
RE_CATEGORY_LINE = re.compile(r"^\s*(Category|श्रेणी)\s*:.*$", re.MULTILINE | re.IGNORECASE)


def clean_wikitext(text: str) -> str:
    """Strip wiki markup down to plain readable text using a real wikitext
    parser (handles nested templates/infoboxes correctly, unlike regex)."""
    if not text:
        return ""

    wikicode = mwparserfromhell.parse(text)

    plain = wikicode.strip_code(normalize=True, collapse=True)

    plain = RE_CATEGORY_LINE.sub("", plain)

    plain = RE_MULTI_SPACE.sub(" ", plain)
    plain = RE_MULTI_NEWLINE.sub("\n\n", plain)

    return plain.strip()


def is_redirect_or_disambig(wikitext: str) -> bool:
    lowered = wikitext[:200].lower()
    return lowered.startswith("#redirect") or lowered.startswith("#REDIRECT".lower())


def word_count(text: str) -> int:
    return len(text.split())


# Separator between articles in the plain-text output.
ARTICLE_SEPARATOR = "\n\n" + ("=" * 80) + "\n\n"


def format_article_txt(title: str, text: str) -> str:
    return f"TITLE: {title}\n\n{text}"


# Dump-based extraction.
DUMP_INDEX_URL = "https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles.xml.bz2"

# MediaWiki export XML namespace.
MW_NS = "{http://www.mediawiki.org/xml/export-0.11/}"


def download_dump(lang: str, dest: Path) -> Path:
    """Download the latest pages-articles XML dump for a target Wikipedia language."""
    url = DUMP_INDEX_URL.format(lang=lang)
    headers = {
        "User-Agent": "LMA-corpus-research-bot/1.0 (contact: your_email@example.com)"
    }
    print(f"Downloading dump: {url}")

    with requests.get(url, stream=True, timeout=60, headers=headers) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / 1e6:.0f}MB / {total / 1e6:.0f}MB ({pct:.1f}%)",
                          end="", flush=True)
    print("\nDownload complete:", dest)
    return dest


def iter_dump_articles(dump_path: Path):
    """Stream a bz2 XML dump and yield (title, wikitext) for each article."""
    with bz2.open(dump_path, "rb") as f:
        context = ET.iterparse(f, events=("end",))
        for _, elem in context:
            tag = elem.tag
            if tag == f"{MW_NS}page":
                ns_elem = elem.find(f"{MW_NS}ns")
                title_elem = elem.find(f"{MW_NS}title")
                revision = elem.find(f"{MW_NS}revision")

                # Only keep the main article namespace.
                if ns_elem is not None and ns_elem.text == "0" and revision is not None:
                    text_elem = revision.find(f"{MW_NS}text")
                    if title_elem is not None and text_elem is not None and text_elem.text:
                        yield title_elem.text, text_elem.text

                elem.clear()


def run_dump_extraction(lang: str, out_path: Path, keep_dump: bool = False,
                         min_words: int = 30, target_words: int = None,
                         out_format: str = "txt"):
    dump_path = Path(f"{lang}wiki-latest-pages-articles.xml.bz2")

    if not dump_path.exists():
        download_dump(lang, dump_path)
    else:
        print(f"Using existing dump file: {dump_path}")

    total_words = 0
    total_articles = 0
    skipped = 0
    first_write = True

    with open(out_path, "w", encoding="utf-8") as out_f:
        for title, wikitext in iter_dump_articles(dump_path):
            if is_redirect_or_disambig(wikitext):
                skipped += 1
                continue

            clean = clean_wikitext(wikitext)
            wc = word_count(clean)

            if wc < min_words:
                skipped += 1
                continue

            if out_format == "jsonl":
                record = {
                    "source": f"{lang}.wikipedia.org",
                    "title": title,
                    "text": clean,
                    "word_count": wc,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:  # plain, human-readable .txt
                if not first_write:
                    out_f.write(ARTICLE_SEPARATOR)
                out_f.write(format_article_txt(title, clean))
                first_write = False

            total_words += wc
            total_articles += 1

            if total_articles % 1000 == 0:
                print(f"\r  {total_articles} articles, {total_words:,} words "
                      f"(skipped {skipped})", end="", flush=True)

            if target_words and total_words >= target_words:
                print(f"\nReached target of {target_words:,} words, stopping.")
                break

    print(f"\nDone. {total_articles} articles, {total_words:,} words -> {out_path}")

    if not keep_dump:
        dump_path.unlink(missing_ok=True)
        print(f"Removed dump file {dump_path} (pass --keep-dump to retain it)")


# Live API-based extraction.

def api_get_all_titles(lang: str, limit: int):
    """Enumerate article titles via the MediaWiki allpages API."""
    session = requests.Session()
    session.headers.update({"User-Agent": "corpus-research-bot/1.0 (contact: you@email.com)"})
    api_url = f"https://{lang}.wikipedia.org/w/api.php"

    titles = []
    apcontinue = None

    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "allpages",
            "apnamespace": 0,
            "aplimit": "500",
            "format": "json",
        }
        if apcontinue:
            params["apcontinue"] = apcontinue

        resp = session.get(api_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        page_batch = [p["title"] for p in data.get("query", {}).get("allpages", [])]
        titles.extend(page_batch)

        apcontinue = data.get("continue", {}).get("apcontinue")
        if not apcontinue:
            break

        time.sleep(0.2)  # be polite to the live API

    return titles[:limit]


def api_get_article_text(lang: str, title: str, session: requests.Session) -> str:
    """Fetch a single article's plain extracted text via the API's
    'extracts' prop (already rendered to plain text server-side —
    no wikitext cleanup needed)."""
    api_url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "format": "json",
    }
    resp = session.get(api_url, params=params, timeout=30)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", "") or ""
    return ""


def run_api_extraction(lang: str, out_path: Path, limit: int, min_words: int = 30,
                        delay: float = 0.3, out_format: str = "txt"):
    print(f"Fetching article title list for {lang}.wikipedia.org (limit={limit})...")
    titles = api_get_all_titles(lang, limit)
    print(f"Got {len(titles)} titles. Fetching article text...")

    session = requests.Session()
    session.headers.update({"User-Agent": "corpus-research-bot/1.0 (contact: you@email.com)"})

    total_words = 0
    total_articles = 0
    first_write = True

    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, title in enumerate(titles, 1):
            try:
                text = api_get_article_text(lang, title, session)
            except requests.RequestException as e:
                print(f"\n  [warn] failed on '{title}': {e}")
                time.sleep(2)
                continue

            wc = word_count(text)
            if wc < min_words:
                continue

            if out_format == "jsonl":
                record = {
                    "source": f"{lang}.wikipedia.org",
                    "title": title,
                    "text": text,
                    "word_count": wc,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:  # plain, human-readable .txt
                if not first_write:
                    out_f.write(ARTICLE_SEPARATOR)
                out_f.write(format_article_txt(title, text))
                first_write = False

            total_words += wc
            total_articles += 1

            if i % 50 == 0:
                print(f"\r  {i}/{len(titles)} processed, {total_articles} kept, "
                      f"{total_words:,} words", end="", flush=True)

            time.sleep(delay)

    print(f"\nDone. {total_articles} articles, {total_words:,} words -> {out_path}")


LATIN_LETTER = re.compile(r"[A-Za-z]")


ALLOWED_CHARS = re.compile(r"[^\u0900-\u097F0-9।॥,.\-!?\s]")

RE_TITLE_LINE = re.compile(r"^TITLE:.*$", re.MULTILINE)


def is_clean_nepali_word(word: str) -> bool:
    """A word is kept only if it contains NO Latin letters at all. This
    drops English words, code-mixed tokens (e.g. 'COVID-19', 'iPhone'),
    URLs, and email fragments in one check, rather than trying to strip
    English letters out of a mixed token and leaving broken remnants."""
    return not LATIN_LETTER.search(word)


def clean_nepali_purity(text: str) -> str:
    """Strip non-Nepali words and special characters from already-extracted
    plain text. Digits are preserved; English/Latin-letter words and stray
    special characters are removed."""
    if not text:
        return ""

    cleaned_lines = []
    for line in text.split("\n"):
        # Preserve TITLE: headers and blank lines / separators as-is so the
        # output file structure (article boundaries) stays intact.
        if line.startswith("TITLE:") or not line.strip() or set(line.strip()) == {"="}:
            cleaned_lines.append(line)
            continue

        words = line.split()
        kept_words = [w for w in words if is_clean_nepali_word(w)]
        line = " ".join(kept_words)

        # Character-level pass: strip any remaining special characters from
        # what's left (stray punctuation/symbols not caught by the word filter).
        line = ALLOWED_CHARS.sub("", line)
        line = RE_MULTI_SPACE.sub(" ", line).strip()

        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)
    result = RE_MULTI_NEWLINE.sub("\n\n", result)
    return result.strip()


def run_purity_clean(in_path: Path, out_path: Path, min_words: int = 30):
    """Read an existing scraped .txt corpus file, apply purity cleaning to
    each article, drop any article that falls below min_words AFTER
    cleaning (since stripping English/special chars can shrink an article
    a lot if it was heavily code-mixed), and write the result."""
    print(f"Cleaning {in_path} -> {out_path}")

    raw = in_path.read_text(encoding="utf-8")
    articles = raw.split(ARTICLE_SEPARATOR)

    kept = 0
    dropped = 0
    total_words = 0
    out_chunks = []

    for article in articles:
        cleaned = clean_nepali_purity(article)

        # Word count excluding the "TITLE: xxx" line itself
        body = RE_TITLE_LINE.sub("", cleaned)
        wc = word_count(body)

        if wc < min_words:
            dropped += 1
            continue

        out_chunks.append(cleaned)
        kept += 1
        total_words += wc

        if kept % 1000 == 0:
            print(f"\r  {kept} articles cleaned, {total_words:,} words "
                  f"(dropped {dropped})", end="", flush=True)

    out_path.write_text(ARTICLE_SEPARATOR.join(out_chunks), encoding="utf-8")
    print(f"\nDone. {kept} articles kept, {dropped} dropped (fell below "
          f"{min_words} words after cleaning), {total_words:,} words -> {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Wikipedia corpus scraper")
    parser.add_argument("method", choices=["dump", "api", "clean"],
                         help="'dump' = download+parse full XML dump (recommended, fast, whole wiki). "
                              "'api' = live per-article pulls via MediaWiki API (slower, for smaller/targeted pulls). "
                              "'clean' = post-process an already-scraped .txt file: drop English/non-Nepali "
                              "words and strip special characters (digits are kept).")
    parser.add_argument("--lang", help="Wikipedia language code, e.g. ne (Nepali), as (Assamese), hi (Hindi). Required for dump/api.")
    parser.add_argument("--in", dest="in_path", help="[clean] Input .txt file to clean (output of a previous dump/api run)")
    parser.add_argument("--out", default="nepalese_scraper/wikipedia_nepalese_data.txt",
                         help="Output file path. For 'clean', defaults to '<input>_clean.txt' if not set.")
    parser.add_argument("--format", choices=["txt", "jsonl"], default="txt",
                         help="'txt' = plain human-readable text, one article after another, easy to open and inspect. "
                              "'jsonl' = one JSON object per line with metadata (title, word_count, source).")
    parser.add_argument("--min-words", type=int, default=30, help="Skip/drop articles shorter than this")
    parser.add_argument("--target-words", type=int, default=None, help="[dump] Stop once this many words collected")
    parser.add_argument("--keep-dump", action="store_true", help="[dump] Don't delete the downloaded .bz2 after extraction")
    parser.add_argument("--limit", type=int, default=5000, help="[api] Max number of articles to fetch")
    parser.add_argument("--delay", type=float, default=0.3, help="[api] Seconds between requests")

    args = parser.parse_args()

    if args.method == "clean":
        if not args.in_path:
            parser.error("--in is required for the 'clean' method")
        in_path = Path(args.in_path)
        out_path = Path(args.out) if args.out != "nepalese_scraper/wikipedia_nepalese_data.txt" \
            else in_path.with_name(in_path.stem + "_clean" + in_path.suffix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        run_purity_clean(in_path, out_path, min_words=args.min_words)
        return

    if not args.lang:
        parser.error("--lang is required for dump/api methods")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.method == "dump":
        run_dump_extraction(args.lang, out_path, keep_dump=args.keep_dump,
                             min_words=args.min_words, target_words=args.target_words,
                             out_format=args.format)
    else:
        run_api_extraction(args.lang, out_path, limit=args.limit,
                            min_words=args.min_words, delay=args.delay,
                            out_format=args.format)


if __name__ == "__main__":
    main()
