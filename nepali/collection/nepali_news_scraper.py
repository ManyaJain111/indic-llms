
import argparse
import hashlib
import json
import re
import time
import urllib.robotparser
import threading
import concurrent.futures
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from collections import deque
from bs4 import BeautifulSoup

import requests
import cloudscraper
import trafilatura

import warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ----------------------------------------------------------------------
# Site configuration
# ----------------------------------------------------------------------
#
# base_url: root of the site, used for robots.txt + resolving relative links
# sitemap_candidates: common sitemap paths tried in order if robots.txt
#   doesn't explicitly list one (many news CMSs don't declare it in robots.txt
#   even though the sitemap exists at a standard path)
#
# NOTE: verify these are still live before a big run — news sites
# occasionally restructure. Test with --sites <one> --target-words 50000
# first to confirm a site actually yields articles before trusting it for
# the full run.

SITES = {
    # Literature & Creative Writing
    "sahityapost": {"base_url": "https://sahityapost.com"},
    "nepalisahityaghar": {"base_url": "https://nepalisahityaghar.com"},
    "samakalinsahitya": {"base_url": "http://www.samakalinsahitya.com"},
    
    # Books & E-Libraries
    "thuprai": {"base_url": "https://thuprai.com"},
    "pustakalaya": {"base_url": "https://pustakalaya.org"},
    
    # Technology & Science
    "techlekh": {"base_url": "https://techlekh.com"},
    "techpana": {"base_url": "https://www.techpana.com"},
    "ictframe": {"base_url": "https://ictframe.com"},
    
    # Education & Student Portals
    "edusanjal": {"base_url": "https://edusanjal.com"},
    "tyrocity": {"base_url": "https://tyrocity.com"},
    "collegenp": {"base_url": "https://www.collegenp.com"},
    
    # Health & Lifestyle
    "swasthyakhabar": {"base_url": "https://swasthyakhabar.com"},
    "swasthyanepal": {"base_url": "https://swasthyanepal.com"},
    
    # Culture, General Blogs & Magazines
    "nepaliaustralian": {"base_url": "https://nepaliaustralian.com"},
    "gorkhapatra_sahitya": {"base_url": "https://gorkhapatraonline.com/categories/sahitya"},
    "shikshya": {"base_url": "https://shikshya.org"},
}

DEBUG = False


def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


USER_AGENT = "LMA-corpus-research-bot/1.0 (contact: your_email@example.com; research use)"

DEFAULT_DELAY = 0.01

SKIP_PATH_PATTERNS = re.compile(
    r"/(tag|tags|author|authors|search|video|videos|"
    r"gallery|photo|photos|live|epaper|advertise|about|contact|privacy|rss)(/|$)",
    re.IGNORECASE,
)


SKIP_SITEMAP_NAME_PATTERN = re.compile(
    r"/(author|authors|tag|tags|member|members)(-\d+)?\.xml$",
    re.IGNORECASE,
)



def load_robots(base_url: str, session: requests.Session) -> urllib.robotparser.RobotFileParser:
    """Fetch and parse robots.txt using OUR session (with proper UA), since
    RobotFileParser.read() by default uses urllib with no custom headers
    and can itself get blocked."""
    rp = urllib.robotparser.RobotFileParser()
    robots_url = urljoin(base_url, "/robots.txt")
    try:
        resp = session.get(robots_url, timeout=15)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
        else:

            rp.parse([])
    except requests.RequestException:
        rp.parse([])
    return rp


def extract_links(html: str, base_url: str, current_url: str) -> list:
    """Extract all valid internal links from the HTML page."""
    soup = BeautifulSoup(html, 'lxml')
    links = []
    base_domain = urlparse(base_url).netloc
    base_domain_stripped = base_domain[4:] if base_domain.startswith("www.") else base_domain
    
    for a in soup.find_all('a', href=True):
        href = a['href']
        # Convert relative to absolute
        abs_url = urljoin(current_url, href)
        parsed = urlparse(abs_url)
        
        # Must be same domain (ignoring www. mismatch)
        parsed_netloc_stripped = parsed.netloc[4:] if parsed.netloc.startswith("www.") else parsed.netloc
        if parsed_netloc_stripped != base_domain_stripped:
            continue
            
        # Keep query parameters but remove fragment to avoid same page hash linking
        clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ''))
        
        # Check against skip patterns
        if SKIP_PATH_PATTERNS.search(parsed.path):
            continue
            
        links.append(clean_url)
    return links



def word_count(text: str) -> int:
    return len(text.split())




RE_WS = re.compile(r"\s+")


def content_fingerprint(text: str) -> str:
    normalized = RE_WS.sub(" ", text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()



LATIN_LETTER = re.compile(r"[A-Za-z]")
ALLOWED_CHARS = re.compile(r"[^\u0900-\u097F0-9।॥,.\-!?\s]")
RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")
RE_MULTI_NEWLINE = re.compile(r"\n{3,}")


def is_clean_nepali_word(word: str) -> bool:
    return not LATIN_LETTER.search(word)


def clean_nepali_purity(text: str) -> str:
    """Drop English/Latin-letter words, strip stray special characters.
    Digits (Devanagari and Latin) are preserved."""
    if not text:
        return ""
    cleaned_lines = []
    for line in text.split("\n"):
        if not line.strip():
            cleaned_lines.append(line)
            continue
        words = [w for w in line.split() if is_clean_nepali_word(w)]
        line = " ".join(words)
        line = ALLOWED_CHARS.sub("", line)
        line = RE_MULTI_SPACE.sub(" ", line).strip()
        cleaned_lines.append(line)
    result = "\n".join(cleaned_lines)
    result = RE_MULTI_NEWLINE.sub("\n\n", result)
    return result.strip()


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if "queues" not in state:
            state["queues"] = {}
        if "visited_urls" not in state:
            state["visited_urls"] = {}
        return state
    return {"processed_urls": [], "seen_hashes": [], "total_words": 0, "queues": {}, "visited_urls": {}}


def save_state(state_path: Path, state: dict):
    state_path.write_text(json.dumps(state), encoding="utf-8")



ARTICLE_SEPARATOR = "\n\n" + ("=" * 80) + "\n\n"


def format_article(url: str, source: str, text: str) -> str:
    return f"SOURCE: {source}\nURL: {url}\n\n{text}"


file_lock = threading.Lock()
state_lock = threading.Lock()
global_processed_urls = set()
global_seen_hashes = set()

def scrape_site_worker(site_name, base_url, target_words, min_words, delay, apply_purity_clean, max_urls_per_site, state, state_path, out_f, shared_queues, shared_visited):
    global global_processed_urls, global_seen_hashes
    
    session = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True})

    
    robots = load_robots(base_url, session)
    crawl_delay = robots.crawl_delay(USER_AGENT) or delay
    if crawl_delay:
        crawl_delay = min(crawl_delay, 0.1)

    queue = shared_queues[site_name]
    site_visited = shared_visited[site_name]

    site_words_collected = 0
    last_request_time = 0.0
    urls_crawled = 0
    
    dprint(f"[{site_name}] INITIAL QUEUE: {len(queue)}, VISITED: {len(site_visited)}, PROCESSED: {len(global_processed_urls)}")

    empty_queue_retries = 0
    try:
        while urls_crawled < max_urls_per_site:
            url = None
            queue_was_empty = False
            
            with state_lock:
                if state.get("total_words", 0) >= target_words:
                    break
                if not queue:
                    queue_was_empty = True
                else:
                    while queue:
                        candidate = queue.popleft()
                        if candidate not in site_visited and candidate not in global_processed_urls:
                            url = candidate
                            site_visited.add(url)
                            break

            if queue_was_empty:
                empty_queue_retries += 1
                if empty_queue_retries > 60:
                    break
                time.sleep(0.5)
                continue
                
            if not url:
                continue
                
            empty_queue_retries = 0
            urls_crawled += 1

            if not robots.can_fetch(USER_AGENT, url):
                dprint(f"    [robots.txt disallows] {url}")
                continue

            elapsed = time.time() - last_request_time
            if elapsed < crawl_delay:
                time.sleep(crawl_delay - elapsed)
            last_request_time = time.time()

            try:
                resp = session.get(url, timeout=7)
                dprint(f"    GET {url} -> HTTP {resp.status_code}, {len(resp.content)} bytes")
                if resp.status_code == 403 and url == base_url:
                    print(f"\n  [IP BLOCKED] {site_name} is blocking this server (Cloudflare 403).")
                    break
                if resp.status_code != 200:
                    continue
                html = resp.text

                if "Just a moment..." in html or "<title>Just a moment...</title>" in html:
                    if url == base_url:
                        print(f"\n  [IP BLOCKED] {site_name} is blocking this server (Cloudflare Challenge).")
                        break
                    else:
                        dprint(f"    [Cloudflare blocked] {url}")
                        continue
            except requests.RequestException as e:
                dprint(f"    GET {url} -> request failed: {e}")
                continue

            new_links = extract_links(html, base_url, url)
            
            with state_lock:
                for link in new_links:
                    if link not in site_visited and link not in global_processed_urls and link not in queue:
                        queue.append(link)

            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=False,
            )
            if not text:
                continue

            fp = content_fingerprint(text)
            with state_lock:
                if fp in global_seen_hashes:
                    dprint(f"    [duplicate content] {url}")
                    continue
                global_seen_hashes.add(fp)
                global_processed_urls.add(url)

            if apply_purity_clean:
                text = clean_nepali_purity(text)

            wc = word_count(text)
            if wc < min_words:
                dprint(f"    [too short after cleaning: {wc} words < {min_words}] {url}")
                continue

            with file_lock:
                out_f.write(ARTICLE_SEPARATOR)
                out_f.write(format_article(url, site_name, text))
                out_f.flush()

            with state_lock:
                state["total_words"] = state.get("total_words", 0) + wc
                current_total = state["total_words"]
                site_words_collected += wc
                
                if urls_crawled % 10 == 0:
                    state_to_save = {
                        "processed_urls": list(global_processed_urls),
                        "seen_hashes": list(global_seen_hashes),
                        "total_words": current_total,
                        "queues": {k: list(v) for k, v in shared_queues.items()},
                        "visited_urls": {k: list(v) for k, v in shared_visited.items()},
                    }
                    save_state(state_path, state_to_save)
                    print(f"\r  [Thread: {site_name}] {site_words_collected:,} words | {current_total:,} / {target_words:,} total", end="", flush=True)

    except Exception as e:
        import traceback
        print(f"\n[CRASH IN {site_name}]: {e}\n{traceback.format_exc()}")

    with state_lock:
        state_to_save = {
            "processed_urls": list(global_processed_urls),
            "seen_hashes": list(global_seen_hashes),
            "total_words": state.get("total_words", 0),
            "queues": {k: list(v) for k, v in shared_queues.items()},
            "visited_urls": {k: list(v) for k, v in shared_visited.items()},
        }
        save_state(state_path, state_to_save)
    print(f"\n  [{site_name}] done: {site_words_collected:,} words collected")


def run_crawl(site_names: list, out_path: Path, state_path: Path,
              target_words: int, min_words: int = 80, delay: float = DEFAULT_DELAY,
              apply_purity_clean: bool = True, max_urls_per_site: int = 50000):
    global global_processed_urls, global_seen_hashes
    state = load_state(state_path)
    global_processed_urls = set(state.get("processed_urls", []))
    global_seen_hashes = set(state.get("seen_hashes", []))
    total_words = state.get("total_words", 0)

    shared_queues = {}
    shared_visited = {}
    for site_name in SITES:
        shared_queues[site_name] = deque(state.get("queues", {}).get(site_name, []))
        shared_visited[site_name] = set(state.get("visited_urls", {}).get(site_name, []))
        if not shared_queues[site_name] and site_name in site_names:
            base = SITES[site_name]["base_url"]
            shared_queues[site_name].append(base)
            shared_visited[site_name].discard(base)
            global_processed_urls.discard(base)

    mode = "a" if out_path.exists() and global_processed_urls else "w"
    out_f = open(out_path, mode, encoding="utf-8")

    print(f"Starting Multi-Threaded crawl. Already have {total_words:,} words from a previous run."
          if total_words > 0 else "Starting fresh Multi-Threaded crawl.")

    THREADS_PER_SITE = 5

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(site_names) * THREADS_PER_SITE) as executor:
            futures = []
            for site_name in site_names:
                if site_name not in SITES:
                    print(f"  [warn] unknown site '{site_name}', skipping")
                    continue
                config = SITES[site_name]
                base_url = config["base_url"]
                print(f"Spawning {THREADS_PER_SITE} threads for {site_name} ({base_url})")
                for _ in range(THREADS_PER_SITE):
                    futures.append(executor.submit(
                        scrape_site_worker, site_name, base_url, target_words, min_words,
                        delay, apply_purity_clean, max_urls_per_site, state, state_path, out_f,
                        shared_queues, shared_visited
                    ))
            
            concurrent.futures.wait(futures)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted — progress saved, re-run with --resume to continue.")
    finally:
        out_f.close()
        with state_lock:
            state_to_save = {
                "processed_urls": list(global_processed_urls),
                "seen_hashes": list(global_seen_hashes),
                "total_words": state.get("total_words", 0),
                "queues": {k: list(v) for k, v in shared_queues.items()},
                "visited_urls": {k: list(v) for k, v in shared_visited.items()},
            }
            save_state(state_path, state_to_save)

    total_words = state.get("total_words", 0)
    print(f"\nTotal: {total_words:,} words -> {out_path}")
    if total_words < target_words:
        print(f"Note: fell short of {target_words:,} word target.")



def main():
    parser = argparse.ArgumentParser(description="Nepali news corpus scraper")
    parser.add_argument("--sites", default=",".join(SITES.keys()),
                         help=f"Comma-separated site names to crawl. Available: {', '.join(SITES.keys())}")
    parser.add_argument("--out", default="nepalese_scraper/nepali_news_data.txt",
                         help="Output .txt file path")
    parser.add_argument("--state", default=None,
                         help="State file for resumability. Defaults to '<out>.state.json'")
    parser.add_argument("--target-words", type=int, default=50_000_000,
                         help="Stop once this many total words are collected (default: 50,000,000)")
    parser.add_argument("--min-words", type=int, default=80,
                         help="Skip articles shorter than this (after cleaning)")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                         help="Minimum seconds between requests to the same domain (robots.txt crawl-delay overrides if longer)")
    parser.add_argument("--max-urls-per-site", type=int, default=50000,
                         help="Cap on candidate article URLs discovered per site")
    parser.add_argument("--no-clean", action="store_true",
                         help="Skip the purity-cleaning pass (keep raw extracted text, English words included)")
    parser.add_argument("--resume", action="store_true",
                         help="Resume from state file instead of starting fresh (fresh start also auto-resumes "
                              "if a state file already exists — this flag is mainly documentation of intent)")
    parser.add_argument("--debug", action="store_true",
                         help="Print detailed diagnostics: every HTTP status code, why URLs are skipped, "
                              "why extraction returns nothing. Use this when a site yields 0 words.")

    args = parser.parse_args()

    global DEBUG
    DEBUG = args.debug

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state) if args.state else out_path.with_suffix(out_path.suffix + ".state.json")

    site_names = [s.strip() for s in args.sites.split(",") if s.strip()]

    run_crawl(
        site_names=site_names,
        out_path=out_path,
        state_path=state_path,
        target_words=args.target_words,
        min_words=args.min_words,
        delay=args.delay,
        apply_purity_clean=not args.no_clean,
        max_urls_per_site=args.max_urls_per_site,
    )


if __name__ == "__main__":
    main()
