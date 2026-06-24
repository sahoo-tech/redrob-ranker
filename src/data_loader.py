import gzip
import json
import logging
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


def stream_candidates(path: str) -> Generator[dict, None, None]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

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
    return sum(1 for _ in stream_candidates(path))


def load_all_candidates(path: str) -> list:
    logger.warning(
        "load_all_candidates(): loading entire file into memory. "
        "Use stream_candidates() for large files."
    )
    return list(stream_candidates(path))
