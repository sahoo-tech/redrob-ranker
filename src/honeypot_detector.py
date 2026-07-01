from typing import Tuple

from . import config
from .utils import dates_overlap, normalize_text


_AI_SKILL_TERMS = {normalize_text(s) for s in config.REQUIRED_SKILLS + config.NICE_TO_HAVE_SKILLS}
_RETRIEVAL_TERMS = {normalize_text(t) for t in config.RETRIEVAL_SIGNALS}

_SPECIFIC_RETRIEVAL_TOOLS = {
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "vector database", "dense retrieval",
    "approximate nearest neighbor", "ann",
}

_TECH_TITLE_FRAGMENTS = {
    "engineer", "developer", "scientist", "analyst", "architect",
    "researcher", "programmer", "devops", "sre", "mlops", "data",
}


def _is_nontechnical_title(title: str) -> bool:
    t = normalize_text(title)
    return not any(frag in t for frag in _TECH_TITLE_FRAGMENTS)


def _career_mentions_retrieval(career: list) -> bool:
    for role in career:
        desc = normalize_text(role.get("description") or "")
        if any(term in desc for term in _RETRIEVAL_TERMS):
            return True
    return False


def detect_honeypot(candidate: dict) -> Tuple[bool, str]:
    profile = candidate.get("profile", {}) or {}
    skills = candidate.get("skills", []) or []
    career = candidate.get("career_history", []) or []

    years = float(profile.get("years_of_experience") or 0)
    if years > 45:
        return True, f"years_of_experience={years} is impossible"

    expert_zero_count = 0
    for skill in skills:
        prof = normalize_text(skill.get("proficiency") or "")
        months = int(skill.get("duration_months") or 0)
        if prof in ("expert", "advanced") and months == 0:
            expert_zero_count += 1
    if expert_zero_count >= 3:
        return True, f"{expert_zero_count} skills claim expert/advanced with 0 months experience"

    current_title = profile.get("current_title") or ""
    ai_skill_count = sum(
        1 for s in skills
        if any(term in normalize_text(s.get("name") or "") for term in _AI_SKILL_TERMS)
    )
    if _is_nontechnical_title(current_title) and ai_skill_count >= 8:
        return (
            True,
            f"Non-technical title '{current_title}' with {ai_skill_count} AI skills declared",
        )

    non_current = [r for r in career if not r.get("is_current", False)]
    for i in range(len(non_current)):
        for j in range(i + 1, len(non_current)):
            r1 = non_current[i]
            r2 = non_current[j]
            if dates_overlap(
                r1.get("start_date"), r1.get("end_date"),
                r2.get("start_date"), r2.get("end_date"),
            ):
                return (
                    True,
                    f"Overlapping employment: '{r1.get('company')}' and '{r2.get('company')}'",
                )

    has_career_descriptions = any(
        (role.get("description") or "").strip() for role in career
    )
    specific_retrieval_claims = sum(
        1 for s in skills
        if any(rt in normalize_text(s.get("name") or "") for rt in _SPECIFIC_RETRIEVAL_TOOLS)
        and normalize_text(s.get("proficiency") or "") in ("expert", "advanced")
    )
    if specific_retrieval_claims >= 3 and has_career_descriptions and not _career_mentions_retrieval(career):
        return (
            True,
            f"Claims {specific_retrieval_claims} expert/advanced retrieval tools but no retrieval language in career descriptions",
        )

    return False, ""
