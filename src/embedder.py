"""
embedder.py — Local Sentence-Transformer wrapper for offline semantic embedding.
Downloads and caches the model locally; never calls an external API during ranking.
"""

import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)


def load_model(model_name: str, cache_dir: Optional[str] = None):
    """
    Load a local sentence-transformer model.

    On first call, this downloads the model weights to cache_dir.
    On subsequent calls (offline), it loads directly from cache_dir.

    Args:
        model_name: HuggingFace model name (e.g., 'all-MiniLM-L6-v2')
        cache_dir: Directory to cache the downloaded model. Should be
                   set to config.MODEL_DIR to keep artifacts self-contained.

    Returns:
        SentenceTransformer model instance.

    Raises:
        ImportError: If sentence-transformers is not installed.
    """
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
    """
    Encode a list of text strings into L2-normalised dense embedding vectors.
    Because normalize_embeddings=True, the dot product between any two vectors
    equals their cosine similarity — no separate normalisation step needed.

    Args:
        model: Loaded SentenceTransformer model.
        texts: List of strings to encode.
        batch_size: Encoding batch size. 64 is a good default for CPU.
                    Lower to 32 if you hit memory pressure.
        show_progress: Display tqdm progress bar (useful for 100K candidates).

    Returns:
        np.ndarray: Shape (N, EMBEDDING_DIM), dtype float32, L2-normalised.
    """
    if not texts:
        raise ValueError("encode_texts() called with an empty list.")

    logger.info(f"Encoding {len(texts):,} texts in batches of {batch_size}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # dot product == cosine similarity
    )
    logger.info(f"Encoding complete. Shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    return embeddings.astype(np.float32)


def build_candidate_text(candidate: dict) -> str:
    """
    Synthesize a rich, semantically dense text block from a candidate profile.
    This is the document fed to the embedding model. Richer = better recall.

    Construction order (most important signal first):
      1. Headline + summary
      2. Current role
      3. Career history descriptions (up to 5 most recent, sorted newest-first)
      4. Skills sorted by duration_months descending (most experienced first)
      5. Education (field of study + degree + institution)
      6. Certifications
      7. Platform skill assessment scores (validated by Redrob — high trust signal)

    Args:
        candidate: Parsed candidate JSON dict matching candidate_schema.json.

    Returns:
        str: Pipe-delimited combined text representing the candidate.
             Returns an empty string if the candidate dict has no usable data.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    parts: List[str] = []

    # ------------------------------------------------------------------
    # 1. Headline and summary
    # ------------------------------------------------------------------
    headline = (profile.get("headline") or "").strip()
    summary = (profile.get("summary") or "").strip()
    if headline:
        parts.append(headline)
    if summary:
        parts.append(summary)

    # ------------------------------------------------------------------
    # 2. Current role context
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3. Career history descriptions — up to 5 most recent roles
    #    The data is already ordered newest-first per the schema.
    # ------------------------------------------------------------------
    career = candidate.get("career_history", [])
    for role in career[:5]:
        role_title = (role.get("title") or "").strip()
        role_company = (role.get("company") or "").strip()
        role_desc = (role.get("description") or "").strip()
        if role_desc:
            prefix = f"{role_title} at {role_company}: " if role_title else ""
            parts.append(f"{prefix}{role_desc}")

    # ------------------------------------------------------------------
    # 4. Skills — sorted by duration_months descending so the model
    #    "sees" the candidate's deepest expertise first.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 5. Education — field of study carries strong semantic signal for AI roles
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 6. Certifications
    # ------------------------------------------------------------------
    certs = [
        (c.get("name") or "").strip()
        for c in candidate.get("certifications", [])
        if (c.get("name") or "").strip()
    ]
    if certs:
        parts.append("Certifications: " + ", ".join(certs))

    # ------------------------------------------------------------------
    # 7. Platform skill assessment scores (validated, high-trust signal)
    #    Only include skills with a score — missing means unassessed, not zero.
    # ------------------------------------------------------------------
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
