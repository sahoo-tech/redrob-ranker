import datetime
import re
from typing import Optional

from . import config


_TODAY = datetime.date.today()


def parse_date(date_str: Optional[str]) -> Optional[datetime.date]:
    if not date_str:
        return None
    s = date_str.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def dates_overlap(
    start1: Optional[str],
    end1: Optional[str],
    start2: Optional[str],
    end2: Optional[str],
    ref_date: datetime.date = _TODAY,
) -> bool:
    s1 = parse_date(start1)
    e1 = parse_date(end1) or ref_date
    s2 = parse_date(start2)
    e2 = parse_date(end2) or ref_date

    if s1 is None or s2 is None:
        return False

    return s1 <= e2 and s2 <= e1


def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return s.strip().lower()


def is_tier1_institution(name: Optional[str]) -> bool:
    if not name:
        return False
    n = normalize_text(name)
    return any(kw in n for kw in config.TIER_1_INSTITUTIONS)


def get_experience_score(years: float) -> float:
    bands = config.EXP_BANDS
    for i, (lo, hi, score) in enumerate(bands):
        if i < len(bands) - 1:
            if lo <= years < hi:
                return score
        else:
            if years >= lo:
                return score
    return 0.10
