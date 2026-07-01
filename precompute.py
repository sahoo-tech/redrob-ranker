from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src import config
from src.data_loader import stream_candidates
from src.embedder import load_model, encode_texts, build_candidate_text
from src.features import extract_features
from src.behavioral import compute_behavioral_multiplier
from src.honeypot_detector import detect_honeypot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_CHUNK_SIZE = 5_000


def main(candidates_path: str) -> None:
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)

    model = load_model(config.EMBEDDING_MODEL_NAME, cache_dir=config.MODEL_DIR)

    jd_embedding = model.encode(
        config.JD_TEXT,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    np.save(config.JD_EMBEDDING_PATH, jd_embedding)

    candidate_ids = []
    features_list = []
    behavioral_list = []
    honeypot_list = []
    embedding_chunks = []

    chunk_texts = []

    def flush_chunk() -> None:
        if not chunk_texts:
            return
        chunk_embeddings = encode_texts(
            model,
            chunk_texts,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            show_progress=True,
        )
        embedding_chunks.append(chunk_embeddings)
        chunk_texts.clear()

    for candidate in stream_candidates(candidates_path):
        cid = candidate.get("candidate_id", "")
        hp, _ = detect_honeypot(candidate)
        candidate_ids.append(cid)
        features_list.append(extract_features(candidate))
        behavioral_list.append(compute_behavioral_multiplier(candidate))
        honeypot_list.append(hp)
        chunk_texts.append(build_candidate_text(candidate))

        if len(chunk_texts) >= _CHUNK_SIZE:
            flush_chunk()

    flush_chunk()

    embeddings = np.concatenate(embedding_chunks, axis=0)

    np.save(config.EMBEDDINGS_PATH, embeddings)
    np.save(config.CANDIDATE_IDS_PATH, np.array(candidate_ids, dtype=object))

    with open(config.FEATURES_PATH, "wb") as fh:
        pickle.dump(
            {
                "features": features_list,
                "behavioral": behavioral_list,
                "honeypot": honeypot_list,
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="candidates.jsonl")
    args = parser.parse_args()
    main(args.candidates)
