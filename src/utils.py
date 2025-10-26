# src/utils.py
from __future__ import annotations
from typing import Any, List, Dict
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from .config import Settings

# cache for sentence-transformers if you ever switch away from OpenAI
_ST_MODEL_CACHE: dict[str, SentenceTransformer] = {}

# utils.py
import pandas as pd
import numpy as np
# --- ADD near the top of src/utils.py ---
import pandas as pd
import numpy as np

def normalise_datetime_and_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse 'datetime' robustly and create a reliable string month bucket.
    - Accepts naive/tz-aware/mixed; coerces to UTC then drops tz for period ops.
    - Only sets 'month' when we truly have a timestamp.
    """
    if "datetime" not in df.columns:
        df["month"] = np.nan
        return df

    # Parse -> UTC -> drop tz for period bucketing
    dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df["datetime"] = dt

    has_dt = dt.notna()
    if has_dt.any():
        # convert to naive (no tz) purely for period buckets
        dt_naive = dt.dt.tz_convert("UTC").dt.tz_localize(None)
        df.loc[has_dt, "month"] = dt_naive[has_dt].dt.to_period("M").astype(str)
    else:
        df["month"] = np.nan

    return df

def enrich_incident_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean + enrich Pinecone / CSV incident data so downstream analytics
    and visualisations never break.

    - enforce datetime dtype
    - add 'month' column (YYYY-MM)
    - coerce risk_level to numeric
    - normalise common text cols
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # 1. datetime parsing
    if "datetime" in df.columns:
        # coerce errors='coerce' -> invalid -> NaT
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)

        # month bucket (YYYY-MM). If datetime is NaT, month becomes NaN.
        # We explicitly fill missing with "Unknown".
        df["month"] = df["datetime"].dt.to_period("M").astype(str)
        df.loc[df["month"] == "NaT", "month"] = np.nan

    # 2. numeric risk_level
    if "risk_level" in df.columns:
        df["risk_level"] = pd.to_numeric(df["risk_level"], errors="coerce")

    # 3. clean text-ish columns (avoid None in UI)
    for col in ["flight_phase", "cause_category", "light", "atc_advisory"]:
        if col in df.columns:
            df[col] = df[col].fillna("—")

    # also miss_distance / altitude: make sure not strings like ''.
    for col in ["miss_distance", "altitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_st_model(name: str) -> SentenceTransformer:
    if name not in _ST_MODEL_CACHE:
        _ST_MODEL_CACHE[name] = SentenceTransformer(name)
    return _ST_MODEL_CACHE[name]

def connect_pinecone(settings: Settings):
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY not provided.")
    pc = Pinecone(api_key=settings.pinecone_api_key)
    return pc.Index(settings.pinecone_index)

def embed_texts(settings: Settings, texts: list[str]) -> list[list[float]]:
    """
    Embed text using the same model that built the Pinecone index.
    For you this is 'text-embedding-3-small' (1536-dim).
    """
    if settings.embed_provider.lower().startswith("openai"):
        if not settings.openai_api_key:
            raise RuntimeError("Missing OPENAI_API_KEY.")
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.embeddings.create(
            model=settings.embed_model,
            input=texts
        )
        return [d.embedding for d in resp.data]

    # fallback: local model
    model = get_st_model(settings.embed_model)
    embs = model.encode(texts, normalize_embeddings=True)
    return embs.tolist()

def _risk_map(val: str | None) -> float | None:
    if val is None:
        return None
    v = str(val).strip().lower()
    if v == "low":
        return 1.0
    if v == "medium":
        return 2.0
    if v == "high":
        return 3.0
    return None

def df_from_matches(matches: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten Pinecone matches (with metadata) into a clean DataFrame and
    derive helper columns we use everywhere else.
    """
    rows = []
    for m in matches:
        meta = m.get("metadata", {}) or {}
        row = {**meta}
        row["match_id"] = m.get("id")
        row["score"] = m.get("score")
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # normalise numeric columns
    if "miss_distance" in df.columns:
        df["miss_distance"] = pd.to_numeric(df["miss_distance"], errors="coerce")
    if "altitude" in df.columns:
        df["altitude"] = pd.to_numeric(df["altitude"], errors="coerce")

    # normalise datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        # We'll also keep naive copies for month bucketing to silence tz warnings
        df["datetime_naive"] = (
            df["datetime"]
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
        )
        df["date"] = df["datetime_naive"].dt.date
        df["week"] = df["datetime_naive"].dt.isocalendar().week
        df["month"] = df["datetime_naive"].dt.to_period("M").astype(str)
        df["hour_of_day"] = df["datetime_naive"].dt.hour
    else:
        df["hour_of_day"] = np.nan
        df["month"] = np.nan

    # risk numeric
    if "risk_level" in df.columns:
        df["risk_numeric"] = df["risk_level"].map(_risk_map)
    else:
        df["risk_numeric"] = np.nan
    df = normalise_datetime_and_month(df)
    return df

def build_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summaries for Ask mode (targeted question).
    """
    out: Dict[str, Any] = {}
    if df is None or df.empty:
        return out

    # frequency by cause, phase
    if "cause_category" in df.columns:
        out["by_cause"] = (
            df["cause_category"]
            .fillna("NA")
            .value_counts()
            .head(10)
            .to_dict()
        )
    else:
        out["by_cause"] = {}

    if "flight_phase" in df.columns:
        out["by_phase"] = (
            df["flight_phase"]
            .fillna("NA")
            .value_counts()
            .head(10)
            .to_dict()
        )
    else:
        out["by_phase"] = {}

    if "risk_level" in df.columns:
        out["risk"] = (
            df["risk_level"]
            .fillna("NA")
            .value_counts()
            .to_dict()
        )
    else:
        out["risk"] = {}

    # median miss distance (guard against all-NaN)
    if "miss_distance" in df.columns and df["miss_distance"].notna().any():
        out["median_miss_distance"] = float(
            np.nanmedian(df["miss_distance"])
        )
    else:
        out["median_miss_distance"] = None

    out["count"] = int(len(df))
    return out

def utc_today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()
