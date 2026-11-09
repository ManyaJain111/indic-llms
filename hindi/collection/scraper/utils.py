"""Scraper utility helpers."""
import logging
import os
from pathlib import Path

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load and return the YAML config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(log_file: str = None, level: str = "INFO") -> None:
    """Configure root logger with console and optional file handler."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    handlers = [logging.StreamHandler()]

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def deduplicate(articles: list, seen_url_hashes: set, seen_content_hashes: set) -> tuple[list, set, set]:
    """
    Filter out articles whose url_hash or content_hash is already seen.
    Returns (unique_articles, updated_url_hashes, updated_content_hashes).
    """
    unique = []
    for a in articles:
        if a.url_hash not in seen_url_hashes and (not a.content_hash or a.content_hash not in seen_content_hashes):
            seen_url_hashes.add(a.url_hash)
            if a.content_hash:
                seen_content_hashes.add(a.content_hash)
            unique.append(a)
    return unique, seen_url_hashes, seen_content_hashes
