# src/agent_graph.py
from __future__ import annotations
from typing import TypedDict, Optional, Literal, Dict, Any
import re
import json
import pandas as pd
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

from .config import get_settings
from .utils import connect_pinecone, utc_today_iso
from .retrieval import retrieve_for_question, compose_answer
from .analytics import (
    sweep_collect,
    cluster_themes,
    mine_rules,
    generate_rule_narratives,
    detect_anomalies,
    extract_critical5,
    compute_cluster_insights,
    build_structured_bundle,
)
from .risk_engine import (
    compute_temporal_confidence,
    build_watchlist,
    build_run_metadata,
)


# -------------------------
# Shared state
# -------------------------

class AgentState(TypedDict, total=False):
    query: str
    mode: Literal["ask", "discover"]

    # ASK filters
    date_from: Optional[str]
    date_to: Optional[str]
    phase_filter: Optional[str]

    # dataframes / analytics
    df: pd.DataFrame
    stats: Dict[str, Any]

    # richer discover info
    discover_structured: Dict[str, Any]
    discover_bundle: Dict[str, Any]

    # final answer
    answer: Dict[str, Any]


# -------------------------
# Helpers
# -------------------------

def _parse_filters(query: str) -> Dict[str, Any]:
    ql = query.lower()

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
    ]
    mode = "discover" if any(t in ql for t in discover_triggers) else "ask"

    # detect date range yyyy-mm-dd .. yyyy-mm-dd
    m = re.search(
        r"(\d{4}-\d{2}-\d{2})\s*(?:to|–|\.{2,})\s*(\d{4}-\d{2}-\d{2})",
        ql,
    )
    date_from = date_to = None
    if m:
        date_from, date_to = m.group(1), m.group(2)
    else:
        m2 = re.search(r"(\d{4}-\d{2}-\d{2})", ql)
        if m2:
            date_from = m2.group(1)
            date_to = m2.group(1)

    phase_filter = None
    if "approach" in ql:
        phase_filter = "Approach"
    elif "takeoff" in ql or "take-off" in ql:
        phase_filter = "Takeoff"

    return {
        "mode": mode,
        "date_from": date_from,
        "date_to": date_to,
        "phase_filter": phase_filter,
    }


def _extract_output_text(resp) -> Optional[str]:
    """
    Extract plain text from an OpenAI Responses API response safely.

    We try:
    1. resp.output[*].content[*].text (if present and iterable)
    2. resp.output[*].text (some SDK variants)
    3. resp.output_text (SDK convenience field)
    4. legacy-style resp.data[0].content[0].text

    We join all collected chunks.
    """
    # 1. resp.output[*].content[*].text
    if hasattr(resp, "output") and resp.output is not None:
        chunks = []
        for item in resp.output or []:
            maybe_content_list = getattr(item, "content", None)
            if maybe_content_list:
                for content in maybe_content_list:
                    if hasattr(content, "text") and content.text is not None:
                        chunks.append(str(content.text))
            # Some SDK variants put direct text on the item
            if hasattr(item, "text") and item.text:
                chunks.append(str(item.text))
        if chunks:
            return "".join(chunks).strip()

    # 2. resp.output_text convenience
    if hasattr(resp, "output_text") and resp.output_text:
        return str(resp.output_text).strip()

    # 3. Very defensive legacy fallback
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
    Call OpenAI Responses API to produce a 1–2 paragraph exec brief:

    - What is spiking
    - Which clusters are high risk
    - Which conditions are driving high risk
    - Which incidents are critical

    We ONLY pass structured data we've already computed.
    We forbid invention in the prompt.

    We deliberately keep the request simple (no reasoning.effort etc.)
    to avoid model-parameter errors across different OpenAI models.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    headlines_hint = []
    wl = structured.get("watchlist", [])
    for item in wl[:3]:
        headlines_hint.append(
            f"- {item['context']}: {item['severity_statement']}"
        )
    headlines_text = "\n".join(headlines_hint) if headlines_hint else "None."

    temporal_conf = structured.get("temporal_confidence", "UNKNOWN")

    prompt = (
        "You are an internal aviation safety analyst. "
        "Your job is to write an 'Operational Risk Radar' update.\n\n"
        "You MUST follow these rules:\n"
        "- ONLY use and restate the structured data provided.\n"
        "- DO NOT invent causes, timelines, mitigations, time trends or counts.\n"
        "- Call out the watchlist items as priority focus areas.\n"
        "- Mention any data confidence limits (temporal confidence).\n"
        "- Cite report_numbers from Critical 5.\n"
        "- Make it crystal clear this is early warning, not a regulatory finding.\n\n"
        f"Headlines to restate in your own words:\n{headlines_text}\n\n"
        f"Temporal confidence in spike/trend detection: {temporal_conf}.\n\n"
        "Structured data:\n"
        f"{json.dumps(structured, indent=2)}\n\n"
        "Write 2 short paragraphs for senior operations and safety leadership."
    )


    # Keep the call minimal: just model + input.
    # This avoids the 'reasoning.effort' error you hit.
    resp = client.responses.create(
        model=settings.summary_model,
        input=prompt,
    )

    txt = _extract_output_text(resp)
    if not txt:
        txt = "Executive summary not available."

    return txt



# -------------------------
# Nodes
# -------------------------

def plan_node(state: AgentState) -> Dict[str, Any]:
    return _parse_filters(state["query"])


def ask_node(state: AgentState) -> Dict[str, Any]:
    """
    ASK path: retrieve semantically similar incidents for the question and compute stats.
    """
    settings = get_settings()
    df, stats = retrieve_for_question(
        settings=settings,
        question=state["query"],
        date_from=state.get("date_from"),
        date_to=state.get("date_to"),
        phase=state.get("phase_filter"),
    )
    return {
        "df": df,
        "stats": stats,
    }


def discover_node(state: AgentState) -> Dict[str, Any]:
    """
    DISCOVER path: sweep Pinecone using random probe vectors to collect a large,
    de-duplicated sample across the dataset.
    """
    settings = get_settings()
    index = connect_pinecone(settings)
    df = sweep_collect(index, settings)
    return {
        "df": df,
    }


def enrich_node(state: AgentState) -> Dict[str, Any]:
    """
    After ask/discover:
    - For ask: nothing special.
    - For discover:
        * cluster themes and rank by risk_priority
        * anomaly flags
        * rule narratives (conditions predicting High risk)
        * Critical 5 serious incidents
        * structured bundle for summariser + UI
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
            "critical5": [],
        }
        return {
            "discover_structured": structured,
            "df": df,
        }

    # 1. cluster + label
    dfx, _label_map = cluster_themes(df, seed=settings.seed)

    # 2. anomaly flags (IsolationForest)
    dfa = detect_anomalies(dfx, seed=settings.seed)
    dfx["anomaly"] = dfa["anomaly"]

    # 3. rules -> narratives
    rules_df = mine_rules(dfx)
    rule_lines = generate_rule_narratives(dfx, rules_df)

    # 4. Critical 5 (high-risk, closest calls)
    critical5 = extract_critical5(dfx)

    # 5. cluster-level risk insights
    clusters_info = compute_cluster_insights(dfx)

    # 6. structured bundle for summary + UI
    structured = build_structured_bundle(
        dfx,
        clusters_info,
        rule_lines,
        critical5,
    )
    temporal_conf = compute_temporal_confidence(dfx)
    watchlist = build_watchlist(rule_lines, critical5)

    run_meta = build_run_metadata(
        sample=structured["sample"],
        temporal_confidence=temporal_conf,
        pinecone_index=settings.pinecone_index,
        model_name=settings.summary_model,
        seed=settings.seed,
    )

    # add extras to structured so summary + UI can see them
    structured["temporal_confidence"] = temporal_conf
    structured["watchlist"] = watchlist
    structured["run_metadata"] = run_meta

    return {
        "df": dfx,  # enriched dataframe w/ cluster, anomaly, etc.
        "discover_structured": structured,
    }


def summarise_node(state: AgentState) -> Dict[str, Any]:
    """
    For discover:
        - Call OpenAI to get exec summary (risk radar brief),
          merge that back into a bundle we expose to the UI.
    For ask:
        - Nothing, we summarise later in format_node.
    """
    if state["mode"] == "ask":
        return {}

    structured = state.get("discover_structured", {})
    if not structured:
        bundle = {
            "executive_summary": "No data to summarise.",
            "clusters_table": [],
            "rule_narratives": [],
            "critical5": [],
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
    Final packaging into 'answer' for the UI.
    """
    if state["mode"] == "ask":
        out = compose_answer(
            state.get("df", pd.DataFrame()),
            state.get("stats", {}),
        )
        return {"answer": out}

    # discover
    return {"answer": state.get("discover_bundle", {})}


# -------------------------
# Routing
# -------------------------

def route_after_plan(state: AgentState):
    return state["mode"]  # "ask" or "discover"


# -------------------------
# Build graph
# -------------------------

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("plan", plan_node)
    builder.add_node("ask", ask_node)
    builder.add_node("discover", discover_node)
    builder.add_node("enrich", enrich_node)
    builder.add_node("summarise", summarise_node)
    builder.add_node("format", format_node)

    builder.add_edge(START, "plan")

    # Branch after plan
    builder.add_conditional_edges("plan", route_after_plan)

    # ask & discover both feed enrich -> summarise -> format -> END
    builder.add_edge("ask", "enrich")
    builder.add_edge("discover", "enrich")
    builder.add_edge("enrich", "summarise")
    builder.add_edge("summarise", "format")
    builder.add_edge("format", END)

    return builder.compile()


_compiled_graph = None

def run_agentic(query: str) -> Dict[str, Any]:
    """
    Entry point Streamlit calls:
    - Builds/uses compiled LangGraph agent pipeline
    - Returns dict for UI
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()

    init_state: AgentState = {
        "query": query,
    }

    final_state: AgentState = _compiled_graph.invoke(init_state)

    return {
        "mode": final_state.get("mode"),
        "answer": final_state.get("answer", {}),
        "df": final_state.get("df"),
        "stats": final_state.get("stats"),
    }
