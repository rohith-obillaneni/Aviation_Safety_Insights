# src/agent_graph.py
from __future__ import annotations
from typing import TypedDict, Optional, Literal, Dict, Any
import re
import json
import pandas as pd
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

from .config import get_settings
from .utils import (
    connect_pinecone,
    utc_today_iso,
    normalise_datetime_and_month,
)
from .retrieval import retrieve_for_question, compose_answer
from .analytics import (
    sweep_collect,
    cluster_themes,
    mine_rules,               # returns structured rules_bundle
    detect_anomalies,
    extract_critical5,
    compute_cluster_insights,
)
from .risk_engine import (
    compute_temporal_confidence,
    build_watchlist_from_rules,   # deduped watchlist builder
    build_run_metadata,           # your existing run-metadata function
)

# ------------------------------------------------------------------------------------
# Shared state definition
# ------------------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    # The full query string (including filters we appended like dates / phase)
    query: str

    # The clean user question without our injected filters.
    # We’ll use this for semantic retrieval so Pinecone stays on-topic.
    user_question_raw: Optional[str]

    mode: Literal["ask", "discover"]

    # Filters (for ask mode)
    date_from: Optional[str]
    date_to: Optional[str]
    phase_filter: Optional[str]

    # Dataframes / analytics
    df: pd.DataFrame
    stats: Dict[str, Any]

    # richer discover info (discover mode)
    discover_structured: Dict[str, Any]
    discover_bundle: Dict[str, Any]

    # final packaged answer
    answer: Dict[str, Any]



# ------------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------------

def _parse_filters(query: str) -> Dict[str, Any]:
    """
    Interpret the user's combined query (natural language + our injected filters).

    We do four things:
    1. Decide mode ("ask" vs "discover").
    2. Extract date_from / date_to if present.
    3. Extract a flight phase filter, preferring the *last-mentioned* phase.
       (We append `during {phase}` at the end of the query from the UI,
        so the dropdown wins over whatever the user typed originally.)
    4. Return everything back into agent state.
    """

    ql = query.lower()

    # 1. Mode routing: if user is asking for trends / patterns / spikes,
    #    we go DISCOVER, otherwise ASK.
    discover_triggers = [
        "unique pattern",
        "unique patterns",
        "patterns",
        "insight",
        "insights",
        "whole dataset",
        "entire dataset",
        "identify unique",
        "trend",
        "trends",
        "unknown pattern",
        "clusters",
        "rules",
        "spike",
        "risk radar",
        "risk patterns",
        "early warning",
        "high risk",
        "critical events",
    ]
    mode = "discover" if any(t in ql for t in discover_triggers) else "ask"

    # 2. Extract date_from / date_to
    # We look for `YYYY-MM-DD to YYYY-MM-DD`
    m = re.search(
        r"(\d{4}-\d{2}-\d{2})\s*(?:to|–|\.{2,})\s*(\d{4}-\d{2}-\d{2})",
        ql,
    )
    date_from = date_to = None
    if m:
        date_from, date_to = m.group(1), m.group(2)
    else:
        # Single date -> treat as both from and to
        m2 = re.search(r"(\d{4}-\d{2}-\d{2})", ql)
        if m2:
            date_from = m2.group(1)
            date_to = m2.group(1)

    # 3. Extract flight phase.
    # Instead of crude "if 'descent' in ql", we:
    # - define known phases
    # - find all that appear
    # - pick the one that appears LAST in the query text
    #   (so the dropdown-supplied `during Parked` at the end wins).
    phase_map = {
        "final approach": "Final Approach",
        "initial approach": "Initial Approach",
        "initial climb": "Initial Climb",
        "take-off": "Takeoff",
        "takeoff": "Takeoff",
        "parked": "Parked",
        "taxi": "Taxi",
        "landing": "Landing",
        "descent": "Descent",
        "climb": "Climb",
        "cruise": "Cruise",
        "approach": "Approach",  # generic fallback
    }

    phase_matches = []
    for needle, normalised in phase_map.items():
        idx = ql.rfind(needle)
        if idx != -1:
            phase_matches.append((idx, normalised))

    # choose the phase that appears LAST in the query
    # (highest idx). If none found, leave as None.
    phase_filter = None
    if phase_matches:
        phase_matches.sort(key=lambda x: x[0])  # sort by index ascending
        phase_filter = phase_matches[-1][1]     # take last one mentioned

    return {
        "mode": mode,
        "date_from": date_from,
        "date_to": date_to,
        "phase_filter": phase_filter,
    }


def _extract_output_text(resp) -> Optional[str]:
    """
    Extract plain text from an OpenAI Responses API response.
    Handles multiple SDK shapes safely.
    """
    # Preferred path: resp.output[*].content[*].text
    if hasattr(resp, "output") and resp.output is not None:
        chunks = []
        for item in (resp.output or []):
            maybe_content_list = getattr(item, "content", None)
            if maybe_content_list:
                for content in maybe_content_list:
                    if hasattr(content, "text") and content.text is not None:
                        chunks.append(str(content.text))
            if hasattr(item, "text") and item.text:
                chunks.append(str(item.text))
        if chunks:
            return "".join(chunks).strip()

    # Convenience shortcut
    if hasattr(resp, "output_text") and resp.output_text:
        return str(resp.output_text).strip()

    # Legacy-ish fallback
    if hasattr(resp, "data") and resp.data:
        try:
            first = resp.data[0]
            first_content = getattr(first, "content", None)
            if first_content:
                maybe_txt = getattr(first_content[0], "text", None)
                if maybe_txt:
                    return str(maybe_txt).strip()
        except Exception:
            pass

    return None


def _make_executive_summary(structured: Dict[str, Any]) -> str:
    """
    Generate a 2-paragraph exec brief for leadership.
    - Only restates provided structured data.
    - Calls out watchlist priorities and temporal confidence.
    - Makes it clear this is early warning, not regulatory blame.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    wl = structured.get("watchlist", [])
    headline_bits = []
    for item in wl[:3]:
        headline_bits.append(
            f"- {item['context']}: {item['severity_statement']}"
        )
    headlines_text = "\n".join(headline_bits) if headline_bits else "None."

    temporal_conf = structured.get("temporal_confidence", "UNKNOWN")

    prompt = (
        "You are an internal aviation safety analyst. "
        "Write an 'Operational Risk Radar' update for senior ops & safety leadership.\n\n"
        "Rules you MUST follow:\n"
        "- ONLY use the structured data provided.\n"
        "- Do NOT invent causes, timelines, mitigations, trends, or counts that are not present.\n"
        "- Call out the watchlist items as priority focus areas.\n"
        "- Acknowledge temporal confidence.\n"
        "- Mention Critical 5 (by report_number) if present.\n"
        "- Emphasise this is early warning, not a regulatory finding.\n\n"
        f"Headlines to restate in your own words:\n{headlines_text}\n\n"
        f"Temporal confidence in spike/trend detection: {temporal_conf}.\n\n"
        "Structured data:\n"
        f"{json.dumps(structured, indent=2)}\n\n"
        "Write 2 short paragraphs. Be direct, operational, factual."
    )

    resp = client.responses.create(
        model=settings.summary_model,
        input=prompt,
    )

    txt = _extract_output_text(resp)
    if not txt:
        txt = "Executive summary not available."

    return txt


# ------------------------------------------------------------------------------------
# Graph nodes
# ------------------------------------------------------------------------------------

def plan_node(state: AgentState) -> Dict[str, Any]:
    """
    Decide mode ('ask' vs 'discover') and extract basic filters.
    """
    return _parse_filters(state["query"])


def ask_node(state: AgentState) -> Dict[str, Any]:
    """
    ASK mode:
    1. Build a clean semantic question for Pinecone retrieval
       (strip vague time phrases like 'last week', 'today', etc.).
    2. Retrieve incidents.
    3. Apply user filters (date_from/date_to/phase).
    4. If that filtered set is empty, fall back and retry with relaxed filters
       (no date bounds) so we can still show something relevant.
    5. Normalise datetime/month for plotting.
    """

    settings = get_settings()

    # --- helper to clean the semantic query before embedding
    def _clean_semantic_question(q: str) -> str:
        """
        Pinecone matching is semantic. Terms like 'last week', 'yesterday'
        often aren't in the actual narratives, so they just add noise and
        kill recall. We strip them so retrieval still finds relevant cases.
        """
        txt = q.strip()

        # remove vague temporal phrases
        txt = re.sub(
            r"\b(last week|yesterday|today|tonight|recent|recently|this (morning|afternoon|evening))\b",
            "",
            txt,
            flags=re.IGNORECASE,
        )

        # collapse multiple spaces
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    # 1. choose which question string we embed into Pinecone
    raw_q = state.get("user_question_raw") or state["query"]
    base_question_clean = _clean_semantic_question(raw_q)

    # 2. first retrieval attempt with full filters
    df, stats = retrieve_for_question(
        settings=settings,
        question=base_question_clean,
        date_from=state.get("date_from"),
        date_to=state.get("date_to"),
        phase=state.get("phase_filter"),
    )

    # 3. if nothing came back, retry in a more forgiving way:
    #    - same semantic scenario
    #    - allow ANY date range (drop date_from/date_to)
    #    - still keep phase if the user clearly asked about one
    if (df is None or df.empty):
        df2, stats2 = retrieve_for_question(
            settings=settings,
            question=base_question_clean,
            date_from=None,
            date_to=None,
            phase=state.get("phase_filter"),
        )
        if df2 is not None and not df2.empty:
            df, stats = df2, stats2

    # 4. final normalisation for charts (month buckets, etc.)
    df = normalise_datetime_and_month(df)

    return {
        "df": df,
        "stats": stats,
    }



def discover_node(state: AgentState) -> Dict[str, Any]:
    """
    DISCOVER mode:
    - Sweep Pinecone with random probe vectors to get a broad, deduped sample
    - Then normalise datetime/month for temporal analysis later
    """
    settings = get_settings()
    index = connect_pinecone(settings)
    df = sweep_collect(index, settings)

    df = normalise_datetime_and_month(df)

    return {
        "df": df,
    }


def enrich_node(state: AgentState) -> Dict[str, Any]:
    """
    After retrieval:
    - ASK mode: we already have df + stats, so nothing extra.
    - DISCOVER mode: run fleet-level analytics:
        * cluster_themes → cluster labels / risk view
        * detect_anomalies → mark outliers
        * mine_rules → structured uplift rules (phase/light/etc → elevated risk)
        * build_watchlist_from_rules → dedupe near-duplicates (eg "Parked" twice)
        * extract_critical5 → most severe incidents
        * compute_temporal_confidence → honesty about trend reliability
        * build_run_metadata → audit trail for leadership
    """
    if state["mode"] == "ask":
        return {}

    settings = get_settings()
    df = state.get("df", pd.DataFrame()).copy()

    if df.empty:
        structured = {
            "sample": {
                "total_incidents": 0,
                "analysis_date": utc_today_iso(),
                "methodology": "No data retrieved; check Pinecone config.",
            },
            "clusters_table": [],
            "rule_narratives": [],
            "rules_struct": [],
            "critical5": [],
            "temporal_confidence": "UNKNOWN",
            "watchlist": [],
            "run_metadata": {},
        }
        return {
            "discover_structured": structured,
            "df": df,
        }

    # 1. Cluster themes (groups similar narratives / factors)
    dfx, _label_map = cluster_themes(df, seed=settings.seed)

    # 2. Anomaly detection (e.g. IsolationForest on miss_distance / altitude etc.)
    dfa = detect_anomalies(dfx, seed=settings.seed)
    if "anomaly" in dfa.columns:
        dfx["anomaly"] = dfa["anomaly"]
    else:
        dfx["anomaly"] = False

    # 3. Association rules -> high-value structured risk relationships
    rules_bundle = mine_rules(dfx)
    all_rules_struct = rules_bundle.get("rules", [])

    # Take only the strongest ~6 rules for reporting / readability.
    top_rules_struct = all_rules_struct[:6]

    # These short narratives feed "Elevated-risk conditions" in the UI.
    rule_narratives = [r["narrative"] for r in top_rules_struct]

    # 4. Critical 5 (highest-risk / closest calls)
    critical5 = extract_critical5(dfx)

    # 5. Cluster-level risk table for UI
    clusters_info = compute_cluster_insights(dfx)

    # 6. Temporal confidence (how reliable is any apparent spike / trend)
    temporal_conf = compute_temporal_confidence(dfx)

    # 7. Watchlist from top rules (deduped + human readable)
    watchlist = build_watchlist_from_rules(top_rules_struct, top_k=5)


    # 8. Sample/methodology metadata
    sample_meta = {
        "total_incidents": int(len(dfx)),
        "analysis_date": utc_today_iso(),
        "methodology": (
            "Sampled via random {dim}-dim vector sweep across Pinecone index "
            "'{idx}' (cosine, {model}). Deduplicated by report_number. "
            "Directional, not full-fleet statistics."
        ).format(
            dim=settings.vector_dim,
            idx=settings.pinecone_index,
            model=settings.embed_model,
        ),
    }

    # 9. Run metadata for audit trail / transparency in UI
    run_meta = build_run_metadata(
        sample=sample_meta,
        temporal_confidence=temporal_conf,
        pinecone_index=settings.pinecone_index,
        model_name=settings.summary_model,
        seed=settings.seed,
    )

    # 10. Final structured object for summariser + UI
    structured = {
        "sample": sample_meta,
        "clusters_table": clusters_info,
        "rule_narratives": rule_narratives,       # shortened narratives (top ~6)
        "rules_struct": top_rules_struct,         # only high-value rules
        "critical5": critical5,
        "temporal_confidence": temporal_conf,
        "watchlist": watchlist,
        "run_metadata": run_meta,
    }



    return {
        "df": dfx,  # enriched dataframe (cluster labels, anomaly flag, etc.)
        "discover_structured": structured,
    }


def summarise_node(state: AgentState) -> Dict[str, Any]:
    """
    DISCOVER mode:
        - Turn the structured analytics into an exec-facing summary.
    ASK mode:
        - Leave summarising to compose_answer in format_node.
    """
    if state["mode"] == "ask":
        return {}

    structured = state.get("discover_structured", {})
    if not structured:
        # Fallback if something goes wrong upstream
        bundle = {
            "executive_summary": "No data to summarise.",
            "clusters_table": [],
            "rule_narratives": [],
            "critical5": [],
            "watchlist": [],
            "run_metadata": {},
            "sample": {
                "total_incidents": 0,
                "analysis_date": utc_today_iso(),
                "methodology": "No data.",
            },
        }
        return {"discover_bundle": bundle}

    exec_summary = _make_executive_summary(structured)

    bundle = {
        "executive_summary": exec_summary,
        "clusters_table": structured.get("clusters_table", []),
        "rule_narratives": structured.get("rule_narratives", []),
        "critical5": structured.get("critical5", []),
        "watchlist": structured.get("watchlist", []),
        "run_metadata": structured.get("run_metadata", {}),
        "sample": structured.get("sample", {}),
    }

    return {"discover_bundle": bundle}


def format_node(state: AgentState) -> Dict[str, Any]:
    """
    Final packaging for Streamlit.
    ASK mode: return slice summary, claims, charts, evidence.
    DISCOVER mode: return fleet-level bundle.
    """
    if state["mode"] == "ask":
        out = compose_answer(
            state.get("df", pd.DataFrame()),
            state.get("stats", {}),
        )
        return {"answer": out}

    # discover mode
    return {"answer": state.get("discover_bundle", {})}


# ------------------------------------------------------------------------------------
# Routing & graph wiring
# ------------------------------------------------------------------------------------

def route_after_plan(state: AgentState):
    """
    After planning, jump to either 'ask' or 'discover'.
    LangGraph will branch to the node with that exact name.
    """
    return state["mode"]  # guaranteed "ask" or "discover"


def build_graph():
    """
    Build the LangGraph pipeline:

    START
      → plan
      → (ask | discover)
      → enrich
      → summarise
      → format
      → END
    """
    builder = StateGraph(AgentState)

    builder.add_node("plan", plan_node)
    builder.add_node("ask", ask_node)
    builder.add_node("discover", discover_node)
    builder.add_node("enrich", enrich_node)
    builder.add_node("summarise", summarise_node)
    builder.add_node("format", format_node)

    builder.add_edge(START, "plan")

    # Conditional branch after planning
    builder.add_conditional_edges("plan", route_after_plan)

    # Rejoin flow
    builder.add_edge("ask", "enrich")
    builder.add_edge("discover", "enrich")
    builder.add_edge("enrich", "summarise")
    builder.add_edge("summarise", "format")
    builder.add_edge("format", END)

    return builder.compile()


_compiled_graph = None

def run_agentic(
    full_query: str,
    raw_question: Optional[str] = None,
) -> Dict[str, Any]:
    """
    full_query:
        The augmented query we build in the UI (includes dates / 'during Phase').
        This is what _parse_filters() looks at to infer mode, date_from/date_to, etc.

    raw_question:
        The user's original free-text scenario. We use this (after cleaning) for
        semantic retrieval in ASK so Pinecone isn't polluted by ranges, dates, etc.
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()

    init_state: AgentState = {
        "query": full_query.strip(),
    }

    if raw_question:
        init_state["user_question_raw"] = raw_question.strip()

    final_state: AgentState = _compiled_graph.invoke(init_state)

    return {
        "mode": final_state.get("mode"),
        "answer": final_state.get("answer", {}),
        "df": final_state.get("df"),
        "stats": final_state.get("stats"),
    }
