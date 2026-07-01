from __future__ import annotations

import dataclasses
import re
from typing import FrozenSet, List

from . import config
from .utils import (
    get_experience_score,
    is_tier1_institution,
    normalize_text,
)


def _build_fictional_word_sets(fictional_lower: List[str]) -> List[FrozenSet[str]]:
    result = []
    for fc in fictional_lower:
        words = frozenset(w for w in re.split(r'\W+', fc) if w)
        if words:
            result.append(words)
    return result


_FICTIONAL_WORD_SETS: List[FrozenSet[str]] = _build_fictional_word_sets(
    [c.lower() for c in config.FICTIONAL_COMPANIES]
)

_RELEVANT_FIELDS_LOWER: List[str] = [rf.lower() for rf in config.RELEVANT_FIELDS_OF_STUDY]
_REQUIRED_SKILLS_LOWER: List[str] = [s.lower() for s in config.REQUIRED_SKILLS]
_AI_TITLES_LOWER: List[str] = [t.lower() for t in config.AI_ENGINEER_TITLES]
_DISQ_TITLES_LOWER: List[str] = [t.lower() for t in config.DISQUALIFIER_TITLES]
_CONSULTING_LOWER: List[str] = [c.lower() for c in config.CONSULTING_FIRMS]
_PRODUCT_LOWER: List[str] = [c.lower() for c in config.PRODUCT_COMPANIES]
_PREFERRED_LOCS_LOWER: List[str] = [loc.lower() for loc in config.PREFERRED_LOCATIONS]
_PRODUCTION_SIGNALS_LOWER: List[str] = [s.lower() for s in config.PRODUCTION_SIGNALS]
_RETRIEVAL_SIGNALS_LOWER: List[str] = [s.lower() for s in config.RETRIEVAL_SIGNALS]
_ENGINEERING_SIGNALS_LOWER: List[str] = [s.lower() for s in config.ENGINEERING_SIGNALS]
_RESEARCH_ONLY_LOWER: List[str] = [s.lower() for s in config.RESEARCH_ONLY_SIGNALS]
_STRATEGY_ONLY_LOWER: List[str] = [s.lower() for s in config.STRATEGY_ONLY_SIGNALS]
_SHALLOW_AI_LOWER: List[str] = [s.lower() for s in config.SHALLOW_AI_SIGNALS]


def _is_fictional(company_normalized: str) -> bool:
    word_set = frozenset(w for w in re.split(r'\W+', company_normalized) if w)
    return any(fc_words.issubset(word_set) for fc_words in _FICTIONAL_WORD_SETS)


_PROFICIENCY_WEIGHT = {
    "expert":       1.0,
    "advanced":     1.0,
    "intermediate": 0.7,
    "beginner":     0.3,
}


@dataclasses.dataclass
class CandidateFeatures:
    candidate_id: str

    experience_score: float        = 0.0
    required_skill_score: float    = 0.0
    skill_duration_score: float    = 0.0
    career_title_match: float      = 0.0
    company_type_score: float      = 0.0
    education_tier: float          = 0.0
    location_score: float          = 0.0

    consulting_only_flag: bool     = False
    title_mismatch_flag: bool      = False

    years_of_experience: float     = 0.0
    open_to_work: bool             = False
    willing_to_relocate: bool      = False

    notice_period_days: int        = 90
    recruiter_response_rate: float = 0.0

    production_depth: float   = 0.0
    retrieval_depth: float    = 0.0
    research_only_flag: bool  = False
    strategy_heavy_flag: bool = False
    shallow_ai_flag: bool     = False


def extract_features(candidate: dict) -> CandidateFeatures:
    cid = candidate.get("candidate_id", "")
    profile = candidate.get("profile", {}) or {}
    signals = candidate.get("redrob_signals", {}) or {}

    f = CandidateFeatures(candidate_id=cid)

    years = float(profile.get("years_of_experience") or 0)
    f.years_of_experience = years
    f.experience_score = get_experience_score(years)

    f.open_to_work = bool(signals.get("open_to_work_flag", False))
    f.willing_to_relocate = bool(signals.get("willing_to_relocate", False))
    f.notice_period_days = int(signals.get("notice_period_days") or 90)
    f.recruiter_response_rate = float(signals.get("recruiter_response_rate") or 0.0)

    skills = candidate.get("skills", []) or []

    req_score = 0.0
    req_duration_months = 0
    max_req_score = float(len(_REQUIRED_SKILLS_LOWER))

    for skill in skills:
        name = normalize_text(skill.get("name") or "")
        prof = normalize_text(skill.get("proficiency") or "")
        months = int(skill.get("duration_months") or 0)
        weight = _PROFICIENCY_WEIGHT.get(prof, 0.3)

        if any(req in name for req in _REQUIRED_SKILLS_LOWER):
            req_score += weight
            req_duration_months += months

    if max_req_score > 0:
        f.required_skill_score = min(req_score / max_req_score * 3.0, 1.0)

    f.skill_duration_score = min(req_duration_months / 120.0, 1.0)

    career = candidate.get("career_history", []) or []
    ai_title_matches = 0
    consulting_count = 0
    product_count = 0
    prod_hits = 0
    retr_hits = 0
    research_hits = 0
    strategy_hits = 0
    shallow_hits = 0
    total_roles = len(career)

    for role in career:
        title = normalize_text(role.get("title") or "")
        company = normalize_text(role.get("company") or "")
        desc = normalize_text(role.get("description") or "")

        if any(at in title for at in _AI_TITLES_LOWER):
            ai_title_matches += 1

        if any(cf in company for cf in _CONSULTING_LOWER):
            consulting_count += 1
        elif any(pc in company for pc in _PRODUCT_LOWER):
            product_count += 1

        if any(s in desc for s in _PRODUCTION_SIGNALS_LOWER) or any(s in desc for s in _ENGINEERING_SIGNALS_LOWER):
            prod_hits += 1
        if any(s in desc for s in _RETRIEVAL_SIGNALS_LOWER):
            retr_hits += 1
        if any(s in desc for s in _RESEARCH_ONLY_LOWER):
            research_hits += 1
        if any(s in desc for s in _STRATEGY_ONLY_LOWER):
            strategy_hits += 1
        if any(s in desc for s in _SHALLOW_AI_LOWER):
            shallow_hits += 1

    if total_roles > 0:
        f.career_title_match = ai_title_matches / total_roles

        non_fictional = [
            r for r in career
            if not _is_fictional(normalize_text(r.get("company") or ""))
        ]
        non_fic_count = len(non_fictional)
        product_fraction = product_count / non_fic_count if non_fic_count > 0 else 0.0
        consulting_fraction = consulting_count / non_fic_count if non_fic_count > 0 else 0.0

        f.consulting_only_flag = (consulting_count == non_fic_count) if non_fic_count > 0 else False
        f.company_type_score = max(0.0, product_fraction - (consulting_fraction * 0.5))

        f.production_depth = prod_hits / total_roles
        f.retrieval_depth = retr_hits / total_roles
        f.research_only_flag = research_hits > 0 and prod_hits == 0
        f.strategy_heavy_flag = strategy_hits > 0 and prod_hits == 0
        f.shallow_ai_flag = shallow_hits > 0 and retr_hits == 0 and prod_hits == 0

    current_title = normalize_text(profile.get("current_title") or "")
    f.title_mismatch_flag = (
        any(dt in current_title for dt in _DISQ_TITLES_LOWER)
        and not any(at in current_title for at in _AI_TITLES_LOWER)
    )

    education = candidate.get("education", []) or []
    tier_score = 0.0
    for edu in education:
        tier = normalize_text(edu.get("tier") or "")
        institution = edu.get("institution") or ""
        field = normalize_text(edu.get("field_of_study") or "")

        if tier == "tier_1" or is_tier1_institution(institution):
            tier_score = max(tier_score, 1.0)
        elif tier == "tier_2":
            tier_score = max(tier_score, 0.7)
        elif tier == "tier_3":
            tier_score = max(tier_score, 0.4)

        if any(rf in field for rf in _RELEVANT_FIELDS_LOWER):
            tier_score = min(tier_score + 0.1, 1.0)

    f.education_tier = tier_score

    location = normalize_text(profile.get("location") or "")
    country = normalize_text(profile.get("country") or "")

    if any(pl in location for pl in _PREFERRED_LOCS_LOWER):
        f.location_score = 1.0
    elif country in ("india", "in"):
        f.location_score = 0.6
    elif f.willing_to_relocate:
        f.location_score = 0.4
    else:
        f.location_score = 0.1

    return f
