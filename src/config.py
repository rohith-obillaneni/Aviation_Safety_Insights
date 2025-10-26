# src/config.py
from dataclasses import dataclass
import os
import random
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # load .env once at import

@dataclass
class Settings:
    pinecone_api_key: str
    pinecone_index: str
    pinecone_namespace: str | None

    openai_api_key: str
    embed_provider: str            # "openai" or "sentence-transformers"
    embed_model: str               # "text-embedding-3-small"
    summary_model: str             # "o4-mini" or "gpt-4o-mini"

    vector_dim: int                # 1536
    top_k: int
    score_min: float

    sweep_target_max: int
    sweep_probes: int
    sweep_probe_k: int

    seed: int

def get_settings() -> Settings:
    pine_ns = os.getenv("PINECONE_NAMESPACE", "").strip() or None

    s = Settings(
        pinecone_api_key = os.getenv("PINECONE_API_KEY", ""),
        pinecone_index   = os.getenv("PINECONE_INDEX", "asrs-incident-reports"),
        pinecone_namespace = pine_ns,

        openai_api_key   = os.getenv("OPENAI_API_KEY", ""),
        embed_provider   = os.getenv("EMBED_PROVIDER", "openai"),
        embed_model      = os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        summary_model    = os.getenv("OPENAI_SUMMARY_MODEL", "o4-mini"),

        vector_dim       = int(os.getenv("VECTOR_DIM", "1536")),
        top_k            = int(os.getenv("TOP_K", "40")),
        score_min        = float(os.getenv("SCORE_MIN", "0.32")),

        sweep_target_max = int(os.getenv("SWEEP_TARGET_MAX", "3000")),
        sweep_probes     = int(os.getenv("SWEEP_PROBES", "40")),
        sweep_probe_k    = int(os.getenv("SWEEP_PROBE_K", "200")),

        seed             = int(os.getenv("SEED", "42")),
    )

    random.seed(s.seed)
    np.random.seed(s.seed)
    return s
