# src/analytics.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from mlxtend.frequent_patterns import apriori, association_rules
from datetime import datetime, timezone

from .config import Settings
from .utils import df_from_matches


# ----------------------------
# Pinecone sweep / sampling
# ----------------------------

def _random_unit_vector(dim: int, rng: np.random.Generator) -> list[float]:
    v = rng.normal(size=(dim,))
    norm = np.linalg.norm(v) + 1e-12
    return (v / norm).astype("float32").tolist()


def sweep_collect(index, settings: Settings) -> pd.DataFrame:
    """
    Broad sample of incidents from Pinecone:
    - Generate random N-dim unit vectors
    - Query Pinecone for each
    - Deduplicate by report_number
    - Stop at sweep_target_max
    """
    rng = np.random.default_rng(settings.seed)
    all_matches: list[Dict[str, Any]] = []
    seen_reports: set[str] = set()

    for _ in range(settings.sweep_probes):
        probe_vec = _random_unit_vector(settings.vector_dim, rng)
        res = index.query(
            vector=probe_vec,
            top_k=settings.sweep_probe_k,
            include_metadata=True,
            namespace=settings.pinecone_namespace,
        )

        for m in res.get("matches", []):
            meta = m.get("metadata", {}) or {}
            rep_no = str(meta.get("report_number", m.get("id", "")))

            if rep_no in seen_reports:
                continue

            seen_reports.add(rep_no)
            all_matches.append(m)

            if len(all_matches) >= settings.sweep_target_max:
                break

        if len(all_matches) >= settings.sweep_target_max:
            break

    df = df_from_matches(all_matches)
    return df


# ----------------------------
# Clustering / themes
# ----------------------------

def cluster_themes(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, Dict[int,str]]:
    """
    Cluster incidents by text similarity and attach indicative labels
    (top TF-IDF terms near each centroid).
    """
    if df.empty:
        return df.copy(), {}

    text_cols = [
        c for c in [
            "summary_ai",
            "synopsis",
            "primary_problem",
            "full_narrative",
        ]
        if c in df.columns
    ]
    if not text_cols:
        df["_cluster_text"] = ""
    else:
        df["_cluster_text"] = df[text_cols].fillna("").agg(" ".join, axis=1)

    vec = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
    )
    X = vec.fit_transform(df["_cluster_text"])

    # cluster count heuristic
    n = max(4, min(12, max(1, X.shape[0] // 200)))

    km = KMeans(
        n_clusters=n,
        random_state=seed,
        n_init=10,
    )
    clust = km.fit_predict(X)

    df_out = df.copy()
    df_out["cluster"] = clust

    # label clusters by their centroid's strongest terms
    terms = np.array(vec.get_feature_names_out())
    label_map: Dict[int, str] = {}
    for i in range(n):
        centroid = km.cluster_centers_[i]
        top_idx = np.argsort(centroid)[::-1][:5]
        label = ", ".join(terms[top_idx])
        label_map[i] = label
    df_out["cluster_label"] = df_out["cluster"].map(label_map)

    df_out = df_out.drop(columns=["_cluster_text"], errors="ignore")
    return df_out, label_map


# ----------------------------
# Association rules
# ----------------------------

def _pretty_token(tok: str) -> str:
    """
    Convert dummy column names like:
        'flight_phase_Final Approach'
        'risk_level_High'
        'light_not_available'
        'atc_advisory_Traffic alert'

    into readable phrases:
        'flight phase = Final Approach'
        'High risk'
        'lighting = not available'
        'ATC advisory = Traffic alert'
    """
    mappings = {
        "flight_phase": "flight phase",
        "cause_category": "cause category",
        "atc_advisory": "ATC advisory",
        "risk_level": "risk level",   # handled specially for 'High risk'
        "light": "lighting",
    }

    for prefix, nice in mappings.items():
        prefix_ = prefix + "_"
        if tok.startswith(prefix_):
            # take the bit after the first "prefix_"
            value = tok[len(prefix_):]
            # clean it up
            value = value.replace("_", " ").strip().strip("'").strip('"')
            if prefix == "risk_level":
                # e.g. risk_level_High -> 'High risk'
                if value:
                    return f"{value} risk".strip()
                else:
                    return "High risk"
            return f"{nice} = {value}".strip()

    # didn't match known patterns
    return tok.replace("_", " ").strip().strip("'").strip('"')


def _risk_phrase_from_tokens(tokens: list[str]) -> str:
    """
    Turn ['risk_level_High', 'risk_level_3.0'] into a readable umbrella like
    'High risk' / 'Medium risk' / 'Low risk' / 'elevated risk'.
    """
    if not tokens:
        return "elevated risk"

    joined = " ".join(tokens).lower()
    if "high" in joined:
        return "High risk"
    if "med" in joined:
        return "Medium risk"
    if "low" in joined:
        return "Low risk"

    # if tokens look numeric e.g. risk_level_4.0, risk_level_5.0
    # we'll just call it "elevated risk"
    return "elevated risk"


def generate_rule_narratives(df: pd.DataFrame, rules: pd.DataFrame) -> List[str]:
    """
    Produce leadership-facing bullet sentences that explain when risk spikes.

    We keep only strong rules:
    - confidence >= 0.6
    - lift >= 1.5
    - support_n >= 10
    - consequents talk about risk_level (so it's operationally meaningful)

    Output example:
    "When flight phase = Approach and lighting = Night, 62% of incidents were
     classified High risk, ~1.9× the baseline rate (37 incidents)."
    """
    out: List[str] = []
    if df is None or df.empty or rules is None or rules.empty:
        return out

    total = len(df)
    seen_sentences = set()

    for _, r in rules.iterrows():
        conf = float(r["confidence"])
        lift_val = float(r["lift"])
        support = float(r["support"])
        support_n = int(round(support * total))

        if conf < 0.6:
            continue
        if lift_val < 1.5:
            continue
        if support_n < 10:
            continue

        ants_raw = list(r["antecedents"])
        cons_raw = list(r["consequents"])

        # keep rules where consequents are about risk_level_*
        cons_risk = [t for t in cons_raw if t.startswith("risk_level")]
        if not cons_risk:
            continue

        # human summary of the risk consequence
        risk_phrase = _risk_phrase_from_tokens(cons_risk)

        # antecedents without risk_level_* noise
        ants_clean = [t for t in ants_raw if not t.startswith("risk_level")]
        antecedent_phrase = " and ".join(_pretty_token(t) for t in ants_clean) \
            if ants_clean else "these conditions"

        sent = (
            f"When {antecedent_phrase}, "
            f"{round(conf*100)}% of incidents were classified {risk_phrase}, "
            f"~{round(lift_val,1)}× the baseline rate "
            f"({support_n} incidents)."
        )

        if sent not in seen_sentences:
            seen_sentences.add(sent)
            out.append(sent)

    return out[:5]


def extract_critical5(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Surface the 'Critical 5' – the incidents you want Safety to look at now.

    Priority 1:
      - any row where risk_level contains 'high' (case-insensitive),
      - sort by smallest miss_distance (closest calls first),
      - take top 5.

    Fallback:
      - sort by highest risk_numeric, then smallest miss_distance.
      - take top 5.

    Final fallback:
      - just take 5 most recent/non-null rows.

    This guarantees we nearly always show *something* serious.
    """
    if df.empty:
        return []

    work = df.copy()
    work["miss_distance_num"] = pd.to_numeric(
        work.get("miss_distance"), errors="coerce"
    )
    work["risk_numeric"] = pd.to_numeric(
        work.get("risk_numeric"), errors="coerce"
    )

    # Priority 1: explicit 'High' in risk_level text
    if "risk_level" in work.columns:
        hi_mask = work["risk_level"].astype(str).str.contains(
            "high", case=False, na=False
        )
    else:
        hi_mask = pd.Series(False, index=work.index)

    hi = work[hi_mask].copy()

    if not hi.empty:
        src = hi.sort_values(
            ["miss_distance_num", "risk_numeric"],
            ascending=[True, False],
            na_position="last",
        ).head(5)
    else:
        # fallback by numeric risk
        if work["risk_numeric"].notna().any():
            src = work.sort_values(
                ["risk_numeric", "miss_distance_num"],
                ascending=[False, True],
                na_position="last",
            ).head(5)
        else:
            # last resort: first 5 rows
            src = work.head(5)

    picks: List[Dict[str, Any]] = []
    for _, row in src.iterrows():
        picks.append({
            "report_number": row.get("report_number"),
            "flight_phase": row.get("flight_phase"),
            "miss_distance": row.get("miss_distance"),
            "cause_category": row.get("cause_category"),
            "risk_level": row.get("risk_level"),
        })
    return picks


def mine_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mine association rules between categorical factors (phase, light,
    ATC advisory, cause category, risk level).
    Returns a raw rules dataframe.
    """
    cats = [
        "flight_phase",
        "light",
        "atc_advisory",
        "cause_category",
        "risk_level",
    ]
    present = [c for c in cats if c in df.columns]

    if not present or df.empty:
        return pd.DataFrame()

    # one-hot encode categorical vars into boolean indicators
    basket = pd.get_dummies(df[present].fillna("NA"))
    basket = basket.astype(bool)

    if basket.empty:
        return pd.DataFrame()

    freq = apriori(basket, min_support=0.02, use_colnames=True)
    if freq.empty:
        return pd.DataFrame()

    rules = association_rules(freq, metric="lift", min_threshold=1.5)
    rules = rules.sort_values(["lift", "confidence"], ascending=False)
    return rules.reset_index(drop=True)




# ----------------------------
# Anomaly / outlier detection
# ----------------------------

def detect_anomalies(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    IsolationForest on numeric features.
    """
    if df.empty:
        return pd.DataFrame({"anomaly": []})

    work = df.copy()

    work["miss_distance_num"] = pd.to_numeric(work.get("miss_distance"), errors="coerce")
    work["altitude_num"] = pd.to_numeric(work.get("altitude"), errors="coerce")
    work["hour_num"] = pd.to_numeric(work.get("hour_of_day"), errors="coerce")

    feat_cols = ["miss_distance_num", "altitude_num", "hour_num"]
    feat_df = work[feat_cols].copy()

    # handle all-NaN columns
    for c in feat_cols:
        if feat_df[c].isna().all():
            feat_df[c] = 0.0

    if len(feat_df) < 5:
        return pd.DataFrame({"anomaly": [False]*len(df)})

    clf = IsolationForest(
        contamination=0.02,
        random_state=seed,
    )
    pred = clf.fit_predict(feat_df)  # -1 = anomaly, 1 = normal
    outliers = (pred == -1)

    return pd.DataFrame({"anomaly": outliers})






# ----------------------------
# Cluster insights / spike detection
# ----------------------------

def compute_cluster_insights(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    For each cluster:
    - size
    - % High risk
    - avg_risk (Low=1, Med=2, High=3)
    - risk_priority score = avg_risk * (1 + size/total)
    - spike_factor: last_month / (avg of previous months)
    - top phases / causes
    - 5 example report_numbers
    """
    if df.empty or "cluster" not in df.columns:
        return []

    total = len(df)
    out: List[Dict[str, Any]] = []

    for c, g in df.groupby("cluster"):
        size = len(g)

        # % High risk
        if "risk_level" in g.columns and size > 0:
            pct_high = float(
                g["risk_level"].astype(str).str.contains("high", case=False, na=False).mean()
            ) * 100.0
        else:
            pct_high = None

        # avg_risk_numeric
        if "risk_numeric" in g.columns and size > 0:
            avg_risk = float(g["risk_numeric"].mean())
        else:
            avg_risk = None

        # risk_priority (bigger cluster with higher avg risk floats up)
        score_component = avg_risk if avg_risk is not None else 0.0
        risk_priority = score_component * (1.0 + size / max(total, 1))

        # spike_factor by month
        spike_factor = None
        if "month" in g.columns and g["month"].notna().any():
            monthly_counts = (
                g["month"]
                .fillna("NA")
                .value_counts()
                .sort_index()
            )
            if len(monthly_counts) >= 4:
                last_month_val = monthly_counts.iloc[-1]
                prev_avg = monthly_counts.iloc[:-1].mean()
                if prev_avg and prev_avg > 0:
                    spike_factor = float(last_month_val / prev_avg)

        # top phases
        top_phases = ", ".join(
            f"{k}:{v}"
            for k, v in g.get("flight_phase", pd.Series(dtype=str))
            .fillna("NA")
            .value_counts()
            .head(3)
            .to_dict()
            .items()
        )

        # top causes
        top_causes = ", ".join(
            f"{k}:{v}"
            for k, v in g.get("cause_category", pd.Series(dtype=str))
            .fillna("NA")
            .value_counts()
            .head(3)
            .to_dict()
            .items()
        )

        examples = ", ".join(
            g.get("report_number", pd.Series(dtype=str))
            .astype(str)
            .head(5)
            .tolist()
        )

        out.append({
            "cluster": int(c),
            "label": g["cluster_label"].iloc[0] if "cluster_label" in g else "n/a",
            "size": size,
            "pct_high_risk": round(pct_high, 1) if pct_high is not None else None,
            "avg_risk": round(avg_risk, 2) if avg_risk is not None else None,
            "risk_priority": round(risk_priority, 3),
            "spike_factor": round(spike_factor, 2) if spike_factor else None,
            "top_phases": top_phases,
            "top_causes": top_causes,
            "examples": examples,
        })

    out.sort(key=lambda d: d["risk_priority"], reverse=True)
    return out


# ----------------------------
# Bundle for summariser / UI
# ----------------------------

def build_structured_bundle(
    df: pd.DataFrame,
    clusters_info: List[Dict[str, Any]],
    rule_narratives: List[str],
    critical5: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Structured summary we pass to:
    - the OpenAI summariser (executive brief)
    - the Streamlit UI for Discover
    """
    analysis_date = datetime.now(timezone.utc).date().isoformat()

    methodology = (
        "Sampled via random 1536-dim vector sweep across Pinecone index "
        "'asrs-incident-reports' (cosine, text-embedding-3-small, 1536 dim). "
        "Deduplicated by report_number. Directional, not full-fleet statistics."
    )

    bundle = {
        "sample": {
            "total_incidents": int(len(df)),
            "analysis_date": analysis_date,
            "methodology": methodology,
        },
        "clusters_table": clusters_info,
        "rule_narratives": rule_narratives,
        "critical5": critical5,
    }
    return bundle
