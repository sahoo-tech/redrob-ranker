from __future__ import annotations

import dataclasses
from typing import List, Optional

from . import config
from .utils import (
    get_experience_score,
    is_tier1_institution,
    normalize_text,
)


_PROFICIENCY_WEIGHT = {
    "expert":       1.0,
    "advanced":     1.0,
    "intermediate": 0.7,
    "beginner":     0.3,
}


@dataclasses.dataclass
class CandidateFeatures:
    candidate_id: str

    experience_score: float       = 0.0
    required_skill_score: float   = 0.0
    skill_duration_score: float   = 0.0
    career_title_match: float     = 0.0
    company_type_score: float     = 0.0
    education_tier: float         = 0.0
    location_score: float         = 0.0

    consulting_only_flag: bool    = False
    title_mismatch_flag: bool     = False

    years_of_experience: float    = 0.0
    open_to_work: bool            = False
    willing_to_relocate: bool     = False


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

    skills = candidate.get("skills", []) or []
    required_lower = [s.lower() for s in config.REQUIRED_SKILLS]

    req_score = 0.0
    req_duration_months = 0
    max_req_score = len(required_lower) * 1.0

    for skill in skills:
        name = normalize_text(skill.get("name") or "")
        prof = normalize_text(skill.get("proficiency") or "")
        months = int(skill.get("duration_months") or 0)
        weight = _PROFICIENCY_WEIGHT.get(prof, 0.3)

        if any(req in name for req in required_lower):
            req_score += weight
            req_duration_months += months

    if max_req_score > 0:
        f.required_skill_score = min(req_score / max_req_score * 3.0, 1.0)

    f.skill_duration_score = min(req_duration_months / 120.0, 1.0)

    career = candidate.get("career_history", []) or []
    ai_title_matches = 0
    disqualifier_matches = 0
    consulting_count = 0
    product_count = 0
    total_roles = len(career)

    ai_lower = [t.lower() for t in config.AI_ENGINEER_TITLES]
    disq_lower = [t.lower() for t in config.DISQUALIFIER_TITLES]
    consulting_lower = [c.lower() for c in config.CONSULTING_FIRMS]
    product_lower = [c.lower() for c in config.PRODUCT_COMPANIES]
    fictional_lower = [c.lower() for c in config.FICTIONAL_COMPANIES]

    for role in career:
        title = normalize_text(role.get("title") or "")
        company = normalize_text(role.get("company") or "")

        if any(at in title for at in ai_lower):
            ai_title_matches += 1
        if any(dt in title for dt in disq_lower):
            disqualifier_matches += 1

        if any(cf in company for cf in consulting_lower):
            consulting_count += 1
        elif any(pc in company for pc in product_lower):
            product_count += 1

    if total_roles > 0:
        f.career_title_match = ai_title_matches / total_roles
        f.consulting_only_flag = (
            consulting_count == total_roles
            and total_roles > 0
        )

        non_fictional = [
            r for r in career
            if not any(fc in normalize_text(r.get("company") or "") for fc in fictional_lower)
        ]
        non_fic_count = len(non_fictional)
        product_fraction = product_count / non_fic_count if non_fic_count > 0 else 0.0
        consulting_fraction = consulting_count / non_fic_count if non_fic_count > 0 else 0.0

        f.company_type_score = max(0.0, product_fraction - (consulting_fraction * 0.5))

    current_title = normalize_text(profile.get("current_title") or "")
    f.title_mismatch_flag = (
        any(dt in current_title for dt in disq_lower)
        and not any(at in current_title for at in ai_lower)
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

        relevant_fields_lower = [rf.lower() for rf in config.RELEVANT_FIELDS_OF_STUDY]
        if any(rf in field for rf in relevant_fields_lower):
            tier_score = min(tier_score + 0.1, 1.0)

    f.education_tier = tier_score

    location = normalize_text(profile.get("location") or "")
    country = normalize_text(profile.get("country") or "")
    preferred_lower = [l.lower() for l in config.PREFERRED_LOCATIONS]

    if any(pl in location for pl in preferred_lower):
        f.location_score = 1.0
    elif country in ("india", "in"):
        f.location_score = 0.6
    elif f.willing_to_relocate:
        f.location_score = 0.4
    else:
        f.location_score = 0.1

    return f
