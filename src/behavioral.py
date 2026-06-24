import datetime
from typing import Optional

from . import config
from .utils import parse_date


_TODAY = datetime.date.today()
_BEHAVIORAL_MIN = 0.30
_BEHAVIORAL_MAX = 1.20


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _availability_score(signals: dict) -> float:
    score = 0.0

    if signals.get("open_to_work_flag", False):
        score += 0.5

    last_active_str = signals.get("last_active_date") or ""
    last_active = parse_date(last_active_str)
    if last_active:
        days_inactive = (_TODAY - last_active).days
        if days_inactive <= 30:
            score += 0.5
        elif days_inactive <= 90:
            score += 0.3
        elif days_inactive <= 180:
            score += 0.1

    return _clamp(score)


def _responsiveness_score(signals: dict) -> float:
    score = 0.0

    rrr = float(signals.get("recruiter_response_rate") or 0)
    score += rrr * 0.6

    avg_rt = float(signals.get("avg_response_time_hours") or 0)
    if avg_rt > 0:
        rt_score = _clamp(1.0 - (avg_rt / 72.0))
        score += rt_score * 0.4

    return _clamp(score)


def _commitment_score(signals: dict) -> float:
    score = 0.0

    icr = float(signals.get("interview_completion_rate") or 0)
    score += icr * 0.6

    oar = float(signals.get("offer_acceptance_rate") or -1)
    if oar >= 0:
        score += oar * 0.4
    else:
        score += 0.2

    return _clamp(score)


def _engagement_score(signals: dict) -> float:
    score = 0.0

    pcs = float(signals.get("profile_completeness_score") or 0)
    score += (pcs / 100.0) * 0.5

    github = float(signals.get("github_activity_score") or -1)
    if github >= 0:
        score += (github / 100.0) * 0.5
    else:
        score += 0.1

    return _clamp(score)


def _notice_score(signals: dict) -> float:
    days = int(signals.get("notice_period_days") or 0)
    if days <= 15:
        return 1.0
    elif days <= 30:
        return 0.85
    elif days <= 60:
        return 0.65
    elif days <= 90:
        return 0.45
    else:
        return 0.20


def compute_behavioral_multiplier(candidate: dict) -> float:
    signals = candidate.get("redrob_signals", {}) or {}

    bw = config.BEHAVIORAL_WEIGHTS
    composite = (
        bw["availability"]   * _availability_score(signals)
        + bw["responsiveness"] * _responsiveness_score(signals)
        + bw["commitment"]     * _commitment_score(signals)
        + bw["engagement"]     * _engagement_score(signals)
        + bw["notice"]         * _notice_score(signals)
    )

    multiplier = _BEHAVIORAL_MIN + composite * (_BEHAVIORAL_MAX - _BEHAVIORAL_MIN)
    return _clamp(multiplier, _BEHAVIORAL_MIN, _BEHAVIORAL_MAX)
