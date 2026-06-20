"""
data_loader.py — Memory-efficient streaming parser for candidates.jsonl or .jsonl.gz.
Uses a generator to process one candidate at a time, keeping RAM usage low.
"""

import gzip
import json
import logging
from pathlib import Path
from typing import Generator, Iterator

logger = logging.getLogger(__name__)


def stream_candidates(path: str) -> Generator[dict, None, None]:
    """
    Stream candidates line-by-line from a .jsonl or .jsonl.gz file.
    Never loads the full file into memory at once.

    Handles both plain .jsonl and gzip-compressed .jsonl.gz files.

    Args:
        path: Absolute or relative path to the candidates file.

    Yields:
        dict: A single parsed candidate JSON record.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    # Support both .jsonl.gz and plain .jsonl
    # p.suffix gives the LAST suffix: .gz for compressed, .jsonl for plain
    is_gzip = p.suffix == ".gz"
    opener = gzip.open if is_gzip else open

    logger.info(f"Streaming candidates from: {path} (gzip={is_gzip})")

    line_num = 0
    ok_count = 0
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            line_num += 1
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                ok_count += 1
                yield record
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")

    logger.info(f"Streaming complete: {ok_count} valid records from {line_num} total lines.")


def count_candidates(path: str) -> int:
    """
    Count the total number of valid candidates without loading them into memory.
    Useful for initialising progress bars in precompute.py.

    Args:
        path: Absolute or relative path to the candidates file.

    Returns:
        int: Number of valid (parseable) candidate records.
    """
    # Exhaust the generator without storing anything
    return sum(1 for _ in stream_candidates(path))


def load_all_candidates(path: str) -> list:
    """
    Load all candidates into a list.

    WARNING: For 100K candidates (~487 MB), this uses ~2–3 GB of RAM.
    Only use this for small datasets (e.g., sample_candidates.json) or
    during debugging. Use stream_candidates() for production processing.

    Args:
        path: Absolute or relative path to the candidates file.

    Returns:
        list[dict]: All parsed candidate records.
    """
    logger.warning(
        "load_all_candidates(): loading entire file into memory. "
        "Use stream_candidates() for large files."
    )
    return list(stream_candidates(path))
