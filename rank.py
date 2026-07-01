from __future__ import annotations

import argparse
import csv
import logging
import os
import pickle
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.data_loader import stream_candidates
from src.scorer import compute_final_score
from src.reasoner import compose_reason

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(candidates_path: str, out_path: str) -> None:
    logger.info("Loading precomputed artifacts...")

    embeddings = np.load(config.EMBEDDINGS_PATH)
    candidate_ids = np.load(config.CANDIDATE_IDS_PATH, allow_pickle=True).tolist()
    jd_embedding = np.load(config.JD_EMBEDDING_PATH)

    with open(config.FEATURES_PATH, "rb") as fh:
        stored = pickle.load(fh)

    features_list = stored["features"]
    behavioral_list = stored["behavioral"]
    honeypot_list = stored["honeypot"]

    logger.info(f"Loaded {len(candidate_ids):,} candidates.")

    similarities = (embeddings @ jd_embedding).tolist()

    id_to_idx = {cid: i for i, cid in enumerate(candidate_ids)}

    scored = []
    for i, cid in enumerate(candidate_ids):
        score = compute_final_score(
            f=features_list[i],
            semantic_similarity=similarities[i],
            behavioral_multiplier=behavioral_list[i],
            is_honeypot=honeypot_list[i],
        )
        scored.append((score, cid))

    scored.sort(key=lambda x: (-x[0], x[1]))
    top100 = scored[:config.TOP_N]
    top100_ids = {cid for _, cid in top100}

    logger.info("Streaming candidates to collect top-100 profiles for reasoning...")
    profile_map: dict = {}
    for candidate in stream_candidates(candidates_path):
        cid = candidate.get("candidate_id", "")
        if cid in top100_ids:
            profile_map[cid] = candidate
        if len(profile_map) == len(top100_ids):
            break

    logger.info(f"Writing {config.TOP_N} rows to {out_path}...")
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, cid) in enumerate(top100, start=1):
            candidate = profile_map.get(cid, {})
            feat = features_list[id_to_idx[cid]]
            reason = compose_reason(candidate, feat, rank)
            writer.writerow([cid, rank, round(score, 6), reason])

    logger.info(f"Submission written to {out_path}")

    validate_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "validate_submission.py"
    )
    if os.path.exists(validate_script):
        result = subprocess.run(
            [sys.executable, validate_script, out_path],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            sys.exit(result.returncode)
    else:
        logger.warning("validate_submission.py not found at project root — skipping validation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="candidates.jsonl")
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()
    main(args.candidates, args.out)
