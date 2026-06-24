import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)


def load_model(model_name: str, cache_dir: Optional[str] = None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )

    logger.info(f"Loading model '{model_name}' from cache: {cache_dir}")
    model = SentenceTransformer(model_name, cache_folder=cache_dir)
    logger.info("Model loaded successfully.")
    return model


def encode_texts(
    model,
    texts: List[str],
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    if not texts:
        raise ValueError("encode_texts() called with an empty list.")

    logger.info(f"Encoding {len(texts):,} texts in batches of {batch_size}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    logger.info(f"Encoding complete. Shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    return embeddings.astype(np.float32)


def build_candidate_text(candidate: dict) -> str:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    parts: List[str] = []

    headline = (profile.get("headline") or "").strip()
    summary = (profile.get("summary") or "").strip()
    if headline:
        parts.append(headline)
    if summary:
        parts.append(summary)

    title = (profile.get("current_title") or "").strip()
    company = (profile.get("current_company") or "").strip()
    industry = (profile.get("current_industry") or "").strip()
    if title:
        role_ctx = f"Current role: {title}"
        if company:
            role_ctx += f" at {company}"
        if industry:
            role_ctx += f" ({industry})"
        parts.append(role_ctx)

    career = candidate.get("career_history", [])
    for role in career[:5]:
        role_title = (role.get("title") or "").strip()
        role_company = (role.get("company") or "").strip()
        role_desc = (role.get("description") or "").strip()
        if role_desc:
            prefix = f"{role_title} at {role_company}: " if role_title else ""
            parts.append(f"{prefix}{role_desc}")

    skills = candidate.get("skills", [])
    skills_sorted = sorted(
        skills,
        key=lambda s: s.get("duration_months") or 0,
        reverse=True,
    )
    skill_strings: List[str] = []
    for skill in skills_sorted:
        name = (skill.get("name") or "").strip()
        prof = (skill.get("proficiency") or "").strip()
        months = skill.get("duration_months") or 0
        if name:
            skill_strings.append(f"{name} ({prof}, {months}mo)")
    if skill_strings:
        parts.append("Skills: " + ", ".join(skill_strings))

    edu_strings: List[str] = []
    for edu in candidate.get("education", []):
        degree = (edu.get("degree") or "").strip()
        field = (edu.get("field_of_study") or "").strip()
        institution = (edu.get("institution") or "").strip()
        tier = (edu.get("tier") or "").strip()
        if field or degree:
            entry = f"{degree} in {field}" if (degree and field) else (degree or field)
            if institution:
                entry += f" from {institution}"
            if tier == "tier_1":
                entry += " [Tier-1]"
            edu_strings.append(entry)
    if edu_strings:
        parts.append("Education: " + "; ".join(edu_strings))

    certs = [
        (c.get("name") or "").strip()
        for c in candidate.get("certifications", [])
        if (c.get("name") or "").strip()
    ]
    if certs:
        parts.append("Certifications: " + ", ".join(certs))

    assessment_scores: dict = signals.get("skill_assessment_scores") or {}
    if assessment_scores:
        assessed = [
            f"{skill} ({score:.0f}/100)"
            for skill, score in sorted(
                assessment_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]
        parts.append("Assessed skills: " + ", ".join(assessed))

    return " | ".join(parts)
