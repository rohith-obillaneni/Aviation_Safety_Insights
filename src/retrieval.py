# src/retrieval.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import Settings, get_settings
from .utils import connect_pinecone, embed_texts, df_from_matches, build_stats

LONDON_TZ = ZoneInfo("Europe/London")

def _post_filter(df: pd.DataFrame,
                 date_from: Optional[str],
                 date_to: Optional[str],
                 phase: Optional[str]) -> pd.DataFrame:
    """
    Apply optional user filters:
    - date range (inclusive)
    - flight phase

    This version is defensive:
    - we always coerce df['datetime'] to pandas datetime[ns, UTC] if possible
    - we always create df['date'] as pure datetime.date for safe comparison
    - we only filter if the user actually passed a value
    """

    out = df.copy()

    # 1. Normalise datetime -> date
    # We accept a few possible column situations:
    #   - 'datetime' exists (string or timestamp-like)
    #   - 'date' might already be there but as string
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(
            out["datetime"],
            errors="coerce",
            utc=True
        )
        out["date"] = out["datetime"].dt.date
    else:
        # Fallback: if no datetime column, but there *is* a 'date' column,
        # try to coerce that to datetime.date values.
        if "date" in out.columns:
            # coerce to datetime64 first, then take .dt.date
            tmp = pd.to_datetime(out["date"], errors="coerce", utc=True)
            out["date"] = tmp.dt.date
        else:
            # No temporal info at all, just return early.
            # (We can't do date filtering with no dates.)
            out["date"] = pd.NaT

    # 2. Filter by date_from / date_to if provided
    # The incoming values (date_from, date_to) are strings like "2025-08-04".
    # We'll turn them into datetime.date.
    if date_from:
        try:
            start_date = datetime.fromisoformat(date_from).date()
            # Keep only rows where out["date"] is not NaT and >= start_date
            out = out[
                out["date"].apply(lambda d: (pd.notna(d) and d >= start_date))
            ]
        except ValueError:
            # If parsing fails, we just skip filtering on that bound.
            pass

    if date_to:
        try:
            end_date = datetime.fromisoformat(date_to).date()
            out = out[
                out["date"].apply(lambda d: (pd.notna(d) and d <= end_date))
            ]
        except ValueError:
            pass

    # 3. Filter by phase (case-insensitive contains match on 'flight_phase')
    if phase and "flight_phase" in out.columns:
        want_phase = str(phase).strip().lower()
        out = out[
            out["flight_phase"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.contains(want_phase, na=False)
        ]

    return out


def retrieve_for_question(
    settings: Settings,
    question: str,
    date_from: Optional[str],
    date_to: Optional[str],
    phase: Optional[str],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    1. Embed the user's safety question.
    2. Query Pinecone for top_k most semantically similar incidents.
    3. Apply post-filters locally.
    4. Compute stats.
    """
    index = connect_pinecone(settings)

    q_vec = embed_texts(settings, [question])[0]

    res = index.query(
        vector=q_vec,
        top_k=settings.top_k,
        include_metadata=True,
        namespace=settings.pinecone_namespace,
    )

    matches = res.get("matches", [])
    df = df_from_matches(matches)

    df = _post_filter(df, date_from, date_to, phase)

    stats = build_stats(df)

    return df, stats

def compose_answer(df: pd.DataFrame, stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Turn stats + df into:
      - summary (plain English, deterministic),
      - claims with citations (list of report_numbers),
      - stats for charts in the UI.
    """
    answer: Dict[str, Any] = {}

    if df is None or df.empty:
        answer["summary"] = (
            "No matching incidents were found for the specified criteria."
        )
        answer["claims"] = []
        answer["stats"] = stats
        return answer

    total = len(df)

    # Top causes
    if stats.get("by_cause"):
        top_cause, top_cause_count = next(iter(stats["by_cause"].items()))
    else:
        top_cause, top_cause_count = ("N/A", 0)

    # Top phase
    if stats.get("by_phase"):
        top_phase, top_phase_count = next(iter(stats["by_phase"].items()))
    else:
        top_phase, top_phase_count = ("N/A", 0)

    # Risk distribution
    risk_parts = []
    for lvl, cnt in stats.get("risk", {}).items():
        risk_parts.append(f"{lvl}:{cnt}")
    risk_str = ", ".join(risk_parts) if risk_parts else "N/A"

    # Representative incidents
    ex_reports = (
        df.get("report_number", pd.Series(dtype=str))
        .astype(str)
        .head(5)
        .tolist()
    )

    answer["summary"] = (
        f"This subset includes {total} incident(s). The most frequent cause "
        f"category is '{top_cause}' ({top_cause_count} cases). The most "
        f"common flight phase is '{top_phase}' ({top_phase_count} cases). "
        f"Risk score distribution: {risk_str}. Representative reports: "
        f"{', '.join(ex_reports)}."
    )

    claims: List[Dict[str, Any]] = []

    # Claim: most common cause
    claims.append({
        "text": (
            f"Most common cause category is '{top_cause}' "
            f"({top_cause_count}/{total} incidents)."
        ),
        "citations": ex_reports,
    })

    # Claim: most common phase
    claims.append({
        "text": (
            f"Most common flight phase is '{top_phase}' "
            f"({top_phase_count}/{total} incidents)."
        ),
        "citations": ex_reports,
    })

    # Claim: risk score distribution
    claims.append({
        "text": f"Risk score distribution for this subset is: {risk_str}.",
        "citations": ex_reports,
    })


    # Claim: closest miss distances if available
    if "miss_distance" in df.columns:
        df_sorted = df.sort_values(
            "miss_distance", na_position="last"
        ).head(3)
        md_list = []
        for _, row in df_sorted.iterrows():
            md_list.append(
                f"Report {row.get('report_number')} "
                f"miss_distance={row.get('miss_distance')}"
            )
        if md_list:
            claims.append({
                "text": (
                    "Smallest recorded miss_distance values in this subset "
                    "indicate notable proximity events: "
                    + "; ".join(md_list)
                ),
                "citations": [
                    str(rn)
                    for rn in df_sorted.get(
                        "report_number", pd.Series(dtype=str)
                    ).astype(str).tolist()
                ],
            })

    answer["claims"] = claims
    answer["stats"] = stats
    return answer
