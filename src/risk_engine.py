from __future__ import annotations
from typing import List, Dict, Any
import math
import pandas as pd


# risk_engine.py
from collections import defaultdict

def dedupe_watchlist(items: list[dict]) -> list[dict]:
    """
    Merge watchlist entries that describe the same context
    (e.g. "flight phase = Parked") so stakeholders don't see repeats.

    Rules:
    - group by .context (case-insensitive trimmed)
    - keep the item with highest risk_index
    - union all evidence IDs (deduped)
    """
    buckets = {}

    for it in items:
        ctx_raw = it.get("context", "").strip()
        ctx_key = ctx_raw.lower()

        # normalise risk_index to numeric for comparison
        risk_val = it.get("risk_index", None)
        try:
            risk_val = float(risk_val)
        except (TypeError, ValueError):
            risk_val = None

        if ctx_key not in buckets:
            # first time we see this context
            buckets[ctx_key] = {
                "context": ctx_raw,
                "severity_statement": it.get("severity_statement", ""),
                "risk_index": risk_val,
                "action_hint": it.get("action_hint", ""),
                "evidence": list(it.get("evidence", [])),
            }
        else:
            # already have something for this context
            existing = buckets[ctx_key]

            # merge evidence
            existing["evidence"].extend(it.get("evidence", []))
            existing["evidence"] = sorted(list(set(existing["evidence"])))

            # choose the higher risk_index (if any)
            if risk_val is not None:
                if existing["risk_index"] is None or risk_val > existing["risk_index"]:
                    existing["risk_index"] = risk_val
                    existing["severity_statement"] = it.get("severity_statement", existing["severity_statement"])
                    existing["action_hint"] = it.get("action_hint", existing["action_hint"])

    # turn buckets -> list and sort by risk_index desc
    merged = list(buckets.values())
    merged.sort(key=lambda x: (x["risk_index"] is not None, x["risk_index"]), reverse=True)

    # final pass: round risk_index nicely and ensure strings
    for row in merged:
        if row["risk_index"] is not None:
            row["risk_index"] = round(float(row["risk_index"]), 2)
        else:
            row["risk_index"] = "—"

    return merged


def compute_temporal_confidence(df: pd.DataFrame) -> str:
    """
    Rate how trustworthy our spike / trend statements are.
    HIGH    = we have month data for >70% rows
    MEDIUM  = we have month for 30-70%
    LOW     = <30% month coverage (likely your case now)

    This is surfaced to stakeholders so we're honest about limits.
    """
    if "month" not in df.columns:
        return "LOW"
    coverage = df["month"].notna().mean()
    if coverage > 0.7:
        return "HIGH"
    if coverage > 0.3:
        return "MEDIUM"
    return "LOW"

# --- ADD to src/risk_engine.py ---
from typing import List, Dict, Any
import math

def _canon_key(ctx: Dict[str, str]) -> str:
    """
    Canonical grouping key for near-duplicate contexts.
    We group by (phase, light, cause) when present. Adjust to taste.
    """
    phase = ctx.get("flight_phase", "").lower()
    light = ctx.get("light", "").lower()
    cause = ctx.get("cause_category", "").lower()
    return f"phase:{phase}|light:{light}|cause:{cause}"

def build_watchlist_from_rules(
    rules_struct: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Take the high-value rules (already filtered by mine_rules)
    and convert them into a prioritised watchlist for ops/safety.

    We expose:
    - context (plain-English conditions)
    - uplift_x  (times baseline risk)
    - exposure_incidents (how many incidents)
    - priority  (HIGH / MEDIUM)
    - action_hint (what Ops / Safety should actually review)
    """

    def _action_hint(ctx_text: str) -> str:
        low = ctx_text.lower()
        if "final approach" in low or "initial approach" in low or "approach" in low:
            return "Crew SOP / approach stability review."
        if "lighting" in low or "not_available" in low or "night" in low:
            return "Ramp / ground handling in low/no lighting."
        if "equipment failure" in low or "failure" in low:
            return "Review abnormal / emergency procedures and redundancy."
        if "initial climb" in low or "climb" in low:
            return "Review climb-out workload and callout discipline."
        if "descent" in low or "landing" in low:
            return "Stabilised descent / landing discipline review."
        return "Audit procedures & briefing in this context."

    enriched = []
    for r in rules_struct:
        ctx_text = r.get("context_text", "context")
        uplift_val = float(r.get("uplift") or 0.0)
        n_val = int(r.get("n", 0))

        priority = "HIGH" if uplift_val >= 2.0 else "MEDIUM"

        enriched.append({
            "context": ctx_text,
            "severity_statement": r.get("narrative"),
            "uplift_x": round(uplift_val, 2),
            "exposure_incidents": n_val,
            "priority": priority,
            "action_hint": _action_hint(ctx_text),
            "evidence": [],  # (optionally attach Critical-5 IDs here later)
        })

    # sort strongest first: uplift × volume
    enriched.sort(
        key=lambda it: (it["uplift_x"] * max(it["exposure_incidents"], 1)),
        reverse=True,
    )

    return enriched[:top_k]



def build_watchlist(rule_narratives: List[str],
                    critical5: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Turn rule_narratives + critical incidents into an "Operational Watchlist".
    Each watch item has:
    - context (short description of the scenario)
    - severity_statement (lift/confidence phrased)
    - action_hint (suggested focus area)
    - evidence (top report_numbers)
    - risk_index (severity_uplift × log10(volume+1))

    We parse our own rule_narratives, which look like:
    "When flight phase = Parked and lighting = not available, 80% of incidents
     were classified High risk, ~2.2× the baseline rate (91 incidents)."

    We'll extract:
      context = text after "When "
      pct, lift, volume = 80, 2.2, 91

    Then we estimate a risk_index = lift * log10(volume+1).
    """
    items: List[Dict[str, Any]] = []

    for sent in rule_narratives:
        # naive pattern grab
        # context
        ctx = ""
        pct = None
        lift = None
        volume = None

        # "When X, NN% of incidents were classified ... ~L× ... (VOL incidents)."
        if sent.lower().startswith("when "):
            # split on ", " after "When "
            try:
                ctx_part = sent[5:].split(",")[0].strip()
                ctx = ctx_part
            except Exception:
                ctx = sent

        # percentage
        import re
        m_pct = re.search(r"(\d+)% of incidents", sent)
        if m_pct:
            pct = int(m_pct.group(1))

        # lift
        m_lift = re.search(r"~([0-9.]+)×", sent)
        if m_lift:
            lift = float(m_lift.group(1))

        # volume
        m_vol = re.search(r"\((\d+)\s+incidents\)", sent)
        if m_vol:
            volume = int(m_vol.group(1))

        # compute risk_index
        if lift is not None and volume is not None:
            risk_index = round(lift * math.log10(volume + 1), 2)
        else:
            risk_index = None

        # crude action hint from context keywords
        lower_ctx = ctx.lower()
        if "parked" in lower_ctx or "lighting" in lower_ctx:
            action_hint = "Audit ramp / ground handling in low or no lighting."
        elif "cruise" in lower_ctx and "pilot" in lower_ctx:
            action_hint = "Review workload management and checklist discipline in cruise."
        elif "taxi" in lower_ctx:
            action_hint = "Review surface movement / taxi guidance and clearances."
        else:
            action_hint = "Requires operational review."

        items.append({
            "context": ctx,
            "severity_statement": sent,
            "action_hint": action_hint,
            "risk_index": risk_index,
            "evidence": [],  # we will fill from critical5 below
            "source": "rule",
        })

    # attach evidence from Critical 5
    # the idea: for each watch item, if its context mentions e.g. Parked, attach the IDs
    for item in items:
        ctx_l = item["context"].lower()
        ev_ids = []
        for crit in critical5:
            phase = str(crit.get("flight_phase", "")).lower()
            if phase and phase in ctx_l:
                ev_ids.append(str(crit.get("report_number")))
        # limit
        item["evidence"] = ev_ids[:5]

    # sort by risk_index desc, fallback pct desc
    def _score(it):
        if it["risk_index"] is not None:
            return it["risk_index"]
        # fallback: try extract pct again
        import re
        m_pct2 = re.search(r"(\d+)% of incidents", it["severity_statement"])
        if m_pct2:
            return int(m_pct2.group(1)) / 10.0
        return 0.0

    items.sort(key=_score, reverse=True)

    # take top 3 for UI
    return items[:3]


def build_run_metadata(sample: Dict[str, Any],
                       temporal_confidence: str,
                       pinecone_index: str,
                       model_name: str,
                       seed: int) -> Dict[str, Any]:
    """
    Prepare a footer block for UI / audit trail.
    """
    return {
        "generated_at": sample.get("analysis_date"),
        "sample_size": sample.get("total_incidents"),
        "temporal_confidence": temporal_confidence,
        "pinecone_index": pinecone_index,
        "summary_model": model_name,
        "seed": seed,
        "method": sample.get("methodology"),
    }
