#!/usr/bin/env python3
"""
Hindi News Scraper — CLI Entrypoint

Usage examples:
  # Scrape all sites, save to SQLite, export JSONL
  python main.py --all --output jsonl

  # Scrape specific sites with a limit
  python main.py --sites bhaskar amarujala --limit 200

  # Export already-collected data from DB to CSV
  python main.py --export-only --format csv

  # Scrape, then export to all formats
  python main.py --all --format jsonl json csv --limit 500

  # Show stats from existing DB
  python main.py --stats
"""

import argparse
import logging
import sys
from pathlib import Path

from scraper.utils import load_config, setup_logging, deduplicate
from scraper.sites.dainik_bhaskar import BhaskarScraper
from scraper.sites.amar_ujala import AmarUjalaScraper
from scraper.sites.navbharat_times import NavbharatTimesScraper
from scraper.sites.hindustan import HindustanScraper
from scraper.sites.patrika import PatrikaScraper
from storage.db import ArticleDB

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = logging.getLogger(__name__)

# ── Site registry ──────────────────────────────────────────────────────────────
SITE_MAP = {
    "bhaskar":   ("bhaskar",   BhaskarScraper),
    "amarujala": ("amarujala", AmarUjalaScraper),
    "navbharat": ("navbharat", NavbharatTimesScraper),
    "hindustan": ("hindustan", HindustanScraper),
    "patrika":   ("patrika",   PatrikaScraper),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hindi_scraper",
        description="Scrape Hindi news articles from major Indian news websites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Site selection
    site_group = parser.add_mutually_exclusive_group()
    site_group.add_argument(
        "--all",
        action="store_true",
        help="Scrape all enabled sites in config.",
    )
    site_group.add_argument(
        "--sites",
        nargs="+",
        choices=list(SITE_MAP.keys()),
        metavar="SITE",
        help=f"One or more sites to scrape. Choices: {', '.join(SITE_MAP.keys())}",
    )

    # Output & format
    parser.add_argument(
        "--format",
        nargs="+",
        choices=["jsonl", "json", "csv"],
        default=["jsonl"],
        help="Output format(s). Default: jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output files. Overrides config.yaml setting.",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to SQLite DB file. Overrides config.yaml setting.",
    )

    # Scrape controls
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max total articles to collect (across all sites).",
    )
    parser.add_argument(
        "--per-site-limit",
        type=int,
        default=None,
        help="Max articles per site.",
    )

    # Modes
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip scraping; only export existing DB to the requested format(s).",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics and exit.",
    )

    # Misc
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file. Default: config.yaml",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity. Default: INFO",
    )

    return parser


def get_sites_to_scrape(args, cfg: dict) -> list[str]:
    """Resolve which site keys to scrape based on CLI args and config."""
    if args.all:
        return [
            key for key, (cfg_key, _) in SITE_MAP.items()
            if cfg["sites"].get(cfg_key, {}).get("enabled", True)
        ]
    if args.sites:
        return args.sites
    # Default: scrape all enabled sites
    return [
        key for key, (cfg_key, _) in SITE_MAP.items()
        if cfg["sites"].get(cfg_key, {}).get("enabled", True)
    ]


def print_banner():
    print(
        "\n"
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load config
    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    global_cfg = cfg["global"]
    output_dir = Path(args.output_dir or global_cfg["output_dir"])
    db_path = args.db_path or global_cfg["db_path"]

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(global_cfg.get("log_file"), level=args.log_level)

    print_banner()

    with ArticleDB(db_path) as db:

        # ── Stats mode ─────────────────────────────────────────────────────────
        if args.stats:
            stats = db.stats()
            print(f"\n Database Statistics — {db_path}")
            print(f"   Total articles : {stats['total']:,}")
            print(f"\n   By source:")
            for src, cnt in sorted(stats["by_source"].items()):
                print(f"     {src:<25} {cnt:>6,}")
            print(f"\n   By category:")
            for cat, cnt in sorted(stats["by_category"].items()):
                print(f"     {cat:<25} {cnt:>6,}")
            return

        # ── Export-only mode ───────────────────────────────────────────────────
        if args.export_only:
            _export(db, output_dir, args.format)
            return

        # ── Scrape mode ────────────────────────────────────────────────────────
        sites_to_run = get_sites_to_scrape(args, cfg)
        if not sites_to_run:
            logger.error("No sites selected. Use --all or --sites SITE [SITE ...]")
            sys.exit(1)

        logger.info(f"Scraping sites: {', '.join(sites_to_run)}")

        # Fetch already-stored hashes for deduplication
        seen_url_hashes = db.get_all_url_hashes()
        seen_content_hashes = db.get_all_content_hashes()
        total_collected = 0

        for site_key in sites_to_run:
            cfg_key, ScraperClass = SITE_MAP[site_key]
            site_cfg = cfg["sites"].get(cfg_key)
            if not site_cfg:
                logger.warning(f"No config found for site key '{cfg_key}', skipping.")
                continue

            logger.info(f"\n{'─'*50}")
            logger.info(f"  Scraping: {site_cfg['name']}")

            scraper = ScraperClass(site_cfg, global_cfg)
            per_site_limit = args.per_site_limit
            if args.limit:
                remaining = args.limit - total_collected
                if remaining <= 0:
                    logger.info("Global article limit reached. Stopping.")
                    break
                if per_site_limit:
                    per_site_limit = min(per_site_limit, remaining)
                else:
                    per_site_limit = remaining

            try:
                articles = scraper.scrape(limit=per_site_limit, seen_url_hashes=seen_url_hashes)
            except Exception as e:
                logger.error(f"Fatal error scraping {site_cfg['name']}: {e}")
                continue

            # Deduplicate against DB
            unique, seen_url_hashes, seen_content_hashes = deduplicate(articles, seen_url_hashes, seen_content_hashes)
            inserted = db.save_articles(unique)
            total_collected += inserted

            logger.info(
                f" {site_cfg['name']}: scraped={len(articles)}, "
                f"unique={len(unique)}, saved={inserted}"
            )

        # ── Post-scrape export ─────────────────────────────────────────────────
        logger.info(f"\n{'─'*50}")
        logger.info(f" Total newly saved articles: {total_collected:,}")
        logger.info(f"  Total in DB: {db.total():,}")

        if args.format:
            _export(db, output_dir, args.format)

    print(f"\nDone! Output saved to: {output_dir.resolve()}")


def _export(db: ArticleDB, output_dir: Path, formats: list[str]):
    """Export DB content to all requested formats."""
    for fmt in formats:
        path = output_dir / f"articles.{fmt}"
        if fmt == "jsonl":
            db.export_jsonl(str(path))
        elif fmt == "json":
            db.export_json(str(path))
        elif fmt == "csv":
            db.export_csv(str(path))
        print(f"  Exported → {path.resolve()}")


if __name__ == "__main__":
    main()
