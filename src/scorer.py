from __future__ import annotations

from . import config
from .features import CandidateFeatures


def compute_career_quality(f: CandidateFeatures) -> float:
    raw = (
        0.20 * f.experience_score
        + 0.20 * f.career_title_match
        + 0.15 * f.company_type_score
        + 0.10 * f.education_tier
        + 0.10 * f.location_score
        + 0.15 * f.production_depth
        + 0.10 * f.retrieval_depth
    )

    if f.research_only_flag:
        raw *= 0.70
    if f.strategy_heavy_flag:
        raw *= 0.60
    if f.shallow_ai_flag:
        raw *= 0.85

    return min(max(raw, 0.0), 1.0)


def compute_skill_score(f: CandidateFeatures) -> float:
    raw = (
        0.70 * f.required_skill_score
        + 0.30 * f.skill_duration_score
    )
    return min(max(raw, 0.0), 1.0)


def _parse_id_number(candidate_id: str) -> int:
    stripped = candidate_id.replace("CAND_", "").lstrip("0") or "0"
    try:
        return int(stripped)
    except ValueError:
        return 0


def compute_final_score(
    f: CandidateFeatures,
    semantic_similarity: float,
    behavioral_multiplier: float,
    is_honeypot: bool,
) -> float:
    if is_honeypot:
        return 0.0

    career_quality = compute_career_quality(f)
    skill_score = compute_skill_score(f)

    w = config.WEIGHTS
    base = (
        w["semantic_similarity"] * semantic_similarity
        + w["skill_score"] * skill_score
        + w["career_quality"] * career_quality
    )

    if f.consulting_only_flag:
        base *= config.CONSULTING_ONLY_MULTIPLIER
    elif f.title_mismatch_flag:
        base *= config.TITLE_MISMATCH_MULTIPLIER

    score = base * behavioral_multiplier

    id_number = _parse_id_number(f.candidate_id)
    notice_epsilon = ((180 - min(f.notice_period_days, 180)) / 180.0) * 1e-6
    response_epsilon = f.recruiter_response_rate * 1e-7
    id_epsilon = ((9_999_999 - min(id_number, 9_999_999)) / 10_000_000.0) * 1e-9

    return score + notice_epsilon + response_epsilon + id_epsilon
