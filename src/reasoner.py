from __future__ import annotations

from typing import List

from . import config
from .features import CandidateFeatures
from .utils import normalize_text

_REQUIRED_SKILLS_LOWER: List[str] = [s.lower() for s in config.REQUIRED_SKILLS]
_PRODUCT_COMPANIES_LOWER: List[str] = [c.lower() for c in config.PRODUCT_COMPANIES]


_EXP_SEGMENTS_CONFIDENT = [
    "A highly accomplished engineer with {years} years",
    "A distinguished {years}-year engineering veteran",
    "An elite practitioner with {years} years of industry depth",
]

_EXP_SEGMENTS_POSITIVE = [
    "A skilled engineer with {years} years",
    "A capable practitioner with {years} years of hands-on experience",
    "An experienced professional with {years} years",
]

_EXP_SEGMENTS_OBJECTIVE = [
    "A professional with {years} years of experience",
    "An engineer with {years} years in the field",
    "A candidate with {years} years of relevant experience",
]

_SKILLS_INTROS_CONFIDENT = ["excelling in", "commanding expertise in", "mastering"]
_SKILLS_INTROS_POSITIVE = ["specializing in", "proficient in", "focused on"]
_SKILLS_INTROS_OBJECTIVE = ["experienced with", "working with", "familiar with"]

_COMPANY_PHRASES_CONFIDENT = [
    "backed by a stellar track record at {companies}",
    "with elite product pedigree at {companies}",
    "distinguished by product-company depth at {companies}",
]

_COMPANY_PHRASES_POSITIVE = [
    "with solid product experience at {companies}",
    "demonstrating strong product background at {companies}",
    "with proven deployment history at {companies}",
]

_COMPANY_PHRASES_OBJECTIVE = [
    "with background including {companies}",
    "having worked at {companies}",
    "with experience at {companies}",
]

_BEHAVIORAL_PHRASES_CONFIDENT = [
    "immediately available and highly responsive to outreach",
    "ready to join immediately with a strong recruiter response profile",
    "available now with a track record of rapid engagement",
]

_BEHAVIORAL_PHRASES_POSITIVE = [
    "responsive with a short notice window",
    "actively engaging with recruiters and available soon",
    "demonstrating strong platform responsiveness",
]

_BEHAVIORAL_PHRASES_OBJECTIVE = [
    "with a reasonable notice period",
    "available within the standard hiring window",
    "engaging with recruiters on the platform",
]


def _stable_seed(candidate_id: str) -> int:
    normalized = candidate_id.upper().replace("CAND_", "").lstrip("0") or "0"
    return sum(ord(c) * (i + 1) for i, c in enumerate(normalized)) % 1000


def _pick(options: list, seed: int) -> str:
    return options[seed % len(options)]


def _get_matching_skills(candidate: dict) -> List[str]:
    skills = candidate.get("skills", []) or []
    matched = []
    for skill in skills:
        name = normalize_text(skill.get("name") or "")
        if any(req in name for req in _REQUIRED_SKILLS_LOWER):
            matched.append((skill.get("name") or "").strip())
    return matched[:3]


def _get_top_skills(candidate: dict) -> List[str]:
    skills = candidate.get("skills", []) or []
    sorted_skills = sorted(
        skills,
        key=lambda s: int(s.get("duration_months") or 0),
        reverse=True,
    )
    return [
        (s.get("name") or "").strip()
        for s in sorted_skills
        if (s.get("name") or "").strip()
    ][:3]


def _get_product_companies(candidate: dict) -> List[str]:
    career = candidate.get("career_history", []) or []
    seen = []
    for role in career:
        company = (role.get("company") or "").strip()
        if any(pc in normalize_text(company) for pc in _PRODUCT_COMPANIES_LOWER):
            if company not in seen:
                seen.append(company)
    return seen[:2]


def _tier(rank: int) -> str:
    if rank <= 10:
        return "confident"
    elif rank <= 50:
        return "positive"
    return "objective"


def _truncate_at_word(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip(",.") + "."


def compose_reason(candidate: dict, f: CandidateFeatures, rank: int) -> str:
    tier = _tier(rank)
    seed = _stable_seed(f.candidate_id)

    years = int(f.years_of_experience)

    if tier == "confident":
        exp_seg = _pick(_EXP_SEGMENTS_CONFIDENT, seed).format(years=years)
        skills_intro = _pick(_SKILLS_INTROS_CONFIDENT, seed + 1)
        company_phrases = _COMPANY_PHRASES_CONFIDENT
        behavioral_phrases = _BEHAVIORAL_PHRASES_CONFIDENT
    elif tier == "positive":
        exp_seg = _pick(_EXP_SEGMENTS_POSITIVE, seed).format(years=years)
        skills_intro = _pick(_SKILLS_INTROS_POSITIVE, seed + 1)
        company_phrases = _COMPANY_PHRASES_POSITIVE
        behavioral_phrases = _BEHAVIORAL_PHRASES_POSITIVE
    else:
        exp_seg = _pick(_EXP_SEGMENTS_OBJECTIVE, seed).format(years=years)
        skills_intro = _pick(_SKILLS_INTROS_OBJECTIVE, seed + 1)
        company_phrases = _COMPANY_PHRASES_OBJECTIVE
        behavioral_phrases = _BEHAVIORAL_PHRASES_OBJECTIVE

    matched_skills = _get_matching_skills(candidate)
    product_companies = _get_product_companies(candidate)

    if matched_skills:
        skills_seg = f"{skills_intro} {', '.join(matched_skills)}"
    else:
        top_skills = _get_top_skills(candidate)
        if top_skills:
            skills_seg = f"{skills_intro} {', '.join(top_skills)}"
        else:
            skills_seg = None

    if product_companies:
        companies_str = " and ".join(product_companies)
        company_seg = _pick(company_phrases, seed + 2).format(companies=companies_str)
    else:
        company_seg = None

    if f.open_to_work or f.notice_period_days <= 15 or f.recruiter_response_rate >= 0.8:
        behavioral_seg = _pick(behavioral_phrases, seed + 3)
    else:
        behavioral_seg = None

    parts = [seg for seg in [exp_seg, skills_seg, company_seg, behavioral_seg] if seg]

    result = ", ".join(parts) + "."
    return _truncate_at_word(result, config.REASONING_MAX_LEN)
