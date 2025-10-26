# app.py
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.config import get_settings
from src.agent_graph import run_agentic

# ---------- STREAMLIT BASE CONFIG ----------
st.set_page_config(
    page_title="✈️ Aviation Risk Radar",
    layout="wide",
)

settings = get_settings()

# ---------- GLOBAL STYLE / THEME OVERRIDES ----------
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-size: 17px;
        color: #1e293b;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    }

    /* Center page content so it feels like a designed product */
    .block-container {
        max-width: 1100px !important;
        padding-top: 0rem;
        padding-bottom: 4rem;
    }

    /* TOP NAV BAR */
    .top-nav {
        margin-top: 2.5rem; /* <-- pushes header down so it's never clipped */
        background: linear-gradient(180deg,#ffffff 0%,#f8fafc 100%);
        border: 1px solid rgba(0,0,0,0.05);
        border-radius: 0.75rem;
        box-shadow: 0 18px 40px rgba(0,0,0,0.07);
        padding: 1rem 1.25rem;
        margin-bottom: 2rem;

        display: flex;
        flex-direction: row;
        align-items: flex-start;
        flex-wrap: wrap;
        row-gap: 0.5rem;
        column-gap: 0.75rem;
    }
    .top-nav-left {
        display: flex;
        flex-direction: row;
        align-items: center;
        flex-wrap: wrap;
        column-gap: 0.75rem;
    }
    .app-title {
        font-size: 2.5rem;
        line-height: 1.25;
        font-weight: 600;
        background: linear-gradient(90deg,#10b981 0%,#6366f1 80%);
        -webkit-background-clip: text;
        color: transparent;
        display: flex;
        align-items: center;
        column-gap: .5rem;
    }
    .title-emoji {
        font-size: 1.5rem;
        line-height: 1.2;
    }
    .app-badge {
        font-size: 0.8rem;
        font-weight: 500;
        line-height: 1.2;
        color: #065f46;
        background: #d1fae5;
        border: 1px solid #6ee7b7;
        padding: 0.3rem 0.5rem;
        border-radius: 0.5rem;
    }
    .top-nav-desc {
        font-size: 0.95rem;
        line-height: 1.45;
        color: #475569;
        max-width: 800px;
        font-weight: 400;
    }

    /* SECTION HEADERS */
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        line-height: 1.3;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
        color: #0f172a;
        display: flex;
        flex-direction: row;
        align-items: baseline;
        column-gap: .5rem;
    }
    .section-hint {
        font-size: 1rem;
        color: #475569;
        line-height: 1.5;
        margin-bottom: 1rem;
        font-weight: 400;
    }

    /* SUMMARY CARD (Ask Summary + Executive Summary) */
    .summary-card {
        border-left: 5px solid #10b981;
        background: linear-gradient(180deg,#ecfdf5 0%, #ffffff 60%);
        border-radius: 0.9rem;
        border: 1px solid rgba(16,185,129,0.25);
        box-shadow: 0 30px 60px rgba(16,185,129,0.09);
        padding: 1.1rem 1rem 1rem 1rem;
        margin-bottom: 2rem;
    }
    .summary-head {
        font-size: 1.1rem;
        font-weight: 600;
        color: #065f46;
        margin-bottom: 0.6rem;
        line-height: 1.4;
        letter-spacing: -0.02em;
    }
    .summary-body {
        font-size: 1.05rem;
        color: #111827;
        line-height: 1.5;
        font-weight: 500;
        letter-spacing: -0.015em;
    }

    /* SUBHEAD LABELS for plots / tables */
    .table-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #6b7280;
        letter-spacing: .05em;
        margin-top: 1rem;
        margin-bottom: 0.4rem;
    }

    /* GENERAL BODY TEXT */
    .body-text {
        font-size: 1rem;
        line-height: 1.5;
        font-weight: 400;
        color: #1e293b;
    }

    /* STAT CARDS (Discover -> Run context) */
    .stat-row {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .stat-card {
        flex: 1 1 0px;
        min-width: 240px;

        background: radial-gradient(circle at 0% 0%, #1f2937 0%, #111827 70%);
        border: 1px solid #374151;
        border-radius: 0.9rem;
        padding: 1rem 1rem 0.9rem 1rem;
        color: #fff;
        box-shadow: 0 20px 40px rgba(0,0,0,0.6);

        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .stat-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        color: #9ca3af;
        letter-spacing: .05em;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #fff;
        line-height: 1.2;
    }
    .stat-small {
        font-size: 0.8rem;
        color: #9ca3af;
        line-height: 1.4;
        margin-top: 0.6rem;
        font-weight: 400;
    }

    /* WATCHLIST CARDS */
    .watch-item {
        border-left: 5px solid transparent;
        border-image: linear-gradient(#fb923c, #dc2626) 1;
        background: linear-gradient(135deg,#fff7ed 0%,#fef2f2 100%);
        border-radius: 0.8rem;
        padding: 1rem 1rem 0.9rem;
        margin-bottom: 1rem;
        box-shadow: 0 18px 36px rgba(0,0,0,0.07);
        border-top: 1px solid rgba(0,0,0,0.03);
        border-right: 1px solid rgba(0,0,0,0.03);
        border-bottom: 1px solid rgba(0,0,0,0.03);
    }
    .watch-context {
        font-weight: 600;
        font-size: 1rem;
        color: #7f1d1d;
        margin-bottom: 0.5rem;
        line-height: 1.4;
    }
    .watch-body {
        font-size: 1rem;
        color: #4a4a4a;
        line-height: 1.5;
        margin-bottom: 0.5rem;
        font-weight: 400;
    }
    .watch-meta {
        font-size: 0.9rem;
        color: #6b6b6b;
        line-height: 1.4;
    }

    /* FOOTNOTE / DISCLAIMER */
    .small-note {
        color: #6b7280;
        font-size: 0.9rem;
        line-height: 1.4;
        margin-top: 1.5rem;
        font-weight: 400;
    }

    /* Make dataframe headers a touch smaller */
    .dataframe thead tr th {
        font-size: 0.8rem !important;
    }

    /* Green primary buttons */
    .stButton > button {
        background: linear-gradient(90deg,#10b981 0%,#22c55e 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(0,0,0,0.15) !important;
        border-radius: 0.6rem !important;
        padding: 0.6rem 1rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 12px 24px rgba(16,185,129,0.4) !important;
        cursor: pointer;
    }
    .stButton > button:hover {
        box-shadow: 0 16px 32px rgba(16,185,129,0.55) !important;
        filter: brightness(1.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- TOP NAV / HEADER ----------
st.markdown(
    """
    <div class="top-nav">
        <div class="top-nav-left">
            <div class="app-title">
                <span class="title-emoji">✈️</span>
                <span>Aviation Risk Radar</span>
            </div>
            <div class="app-badge">Internal Safety Radar</div>
        </div>
        <div class="top-nav-desc">
            Live operational safety scan. Surfaces concentrated risk, recurring high-severity
            conditions, and the most serious current events with traceable incident IDs.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs
tabs = st.tabs([
    "🔎 Ask (Targeted query)",
    "🧭 Discover (Risk patterns & spikes)",
    "📥 CSV Offline",
])

# =====================================================================================
# TAB 1: ASK MODE (INVESTIGATION)
# =====================================================================================
with tabs[0]:
    st.markdown('<div class="section-title">Focused safety question</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-hint">'
        "Use this to investigate a specific scenario (eg ATC confusion on Descent last week). "
        "You’ll get main drivers, exposure distribution, and the report_numbers to pull."
        '</div>',
        unsafe_allow_html=True,
    )

    question = st.text_input("Question to investigate")
    col1, col2, col3 = st.columns(3)
    with col1:
        date_from = st.text_input("Date from (YYYY-MM-DD, optional)", value="")
    with col2:
        date_to = st.text_input("Date to (YYYY-MM-DD, optional)", value="")
    with col3:
        phase_filter = st.text_input("Flight phase (e.g. Approach, optional)", value="")

    if st.button("Run Ask"):
        # Build agent query text
        q_full = (question or "").strip()
        if date_from and date_to:
            q_full += f" between {date_from} and {date_to}"
        elif date_from:
            q_full += f" after {date_from}"
        elif date_to:
            q_full += f" before {date_to}"
        if phase_filter:
            q_full += f" during {phase_filter}"

        with st.spinner("Retrieving similar incidents and summarising drivers…"):
            result = run_agentic(q_full)

        mode = result.get("mode", "?")
        ans = result.get("answer") or {}
        df_ev = result.get("df")

        st.markdown(f"**Agent mode:** `{mode}`")

        if mode != "ask":
            st.warning(
                "This looks like a fleet/trend request. "
                "Use **🧭 Discover** to run the full risk sweep."
            )
        else:
            # ===== Summary card (bigger font, highlighted) =====
            st.markdown(
                "<div class='summary-card'>"
                "<div class='summary-head'>Summary</div>"
                f"<div class='summary-body'>{ans.get('summary','')}</div>"
                "</div>",
                unsafe_allow_html=True,
            )

            # ===== Key signals with evidence =====
            claims = ans.get("claims", [])
            if claims:
                st.markdown('<div class="section-title">Key signals in this subset</div>', unsafe_allow_html=True)
                for c in claims:
                    st.markdown(
                        f"<div class='body-text'>• {c['text']} — "
                        f"reports: <code>{', '.join(c['citations'])}</code></div>",
                        unsafe_allow_html=True,
                    )

            # ===== Exposure breakdown charts =====
            stats = ans.get("stats", {})
            if stats:
                st.markdown('<div class="section-title">Exposure breakdown</div>', unsafe_allow_html=True)
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    if stats.get("by_cause"):
                        st.markdown('<div class="table-title">Top causes</div>', unsafe_allow_html=True)
                        dfc = pd.DataFrame(list(stats["by_cause"].items()),
                                           columns=["Cause", "Count"])
                        if not dfc.empty:
                            fig, ax = plt.subplots()
                            ax.bar(dfc["Cause"], dfc["Count"])
                            ax.set_xticks(range(len(dfc["Cause"])))
                            ax.set_xticklabels(dfc["Cause"], rotation=45, ha="right")
                            ax.set_ylabel("Count")
                            st.pyplot(fig)

                with col_b:
                    if stats.get("by_phase"):
                        st.markdown('<div class="table-title">Top phases</div>', unsafe_allow_html=True)
                        dfp = pd.DataFrame(list(stats["by_phase"].items()),
                                           columns=["Phase", "Count"])
                        if not dfp.empty:
                            fig, ax = plt.subplots()
                            ax.bar(dfp["Phase"], dfp["Count"])
                            ax.set_xticks(range(len(dfp["Phase"])))
                            ax.set_xticklabels(dfp["Phase"], rotation=45, ha="right")
                            ax.set_ylabel("Count")
                            st.pyplot(fig)

                with col_c:
                    if stats.get("risk"):
                        st.markdown(
                            '<div class="table-title">Risk score distribution (higher = more severe)</div>',
                            unsafe_allow_html=True,
                        )
                        dfrisk = pd.DataFrame(list(stats["risk"].items()),
                                              columns=["RiskScore","Count"])
                        if not dfrisk.empty:
                            fig, ax = plt.subplots()
                            ax.bar(dfrisk["RiskScore"], dfrisk["Count"])
                            ax.set_xticks(range(len(dfrisk["RiskScore"])))
                            ax.set_xticklabels(dfrisk["RiskScore"],
                                               rotation=0, ha="center")
                            ax.set_ylabel("Count")
                            st.pyplot(fig)

            # ===== Evidence table =====
            if isinstance(df_ev, pd.DataFrame) and not df_ev.empty:
                st.markdown('<div class="section-title">Evidence incidents (top matches)</div>', unsafe_allow_html=True)
                cols_to_show = [
                    "report_number",
                    "datetime",
                    "flight_phase",
                    "cause_category",
                    "risk_level",
                    "miss_distance",
                    "score",
                ]
                cols_to_show = [c for c in cols_to_show if c in df_ev.columns]
                display_df = df_ev[cols_to_show].head(50).copy()
                display_df = display_df.fillna("—")
                st.dataframe(display_df, use_container_width=True)

    st.markdown(
        '<div class="small-note">'
        "Ask mode = forensic. You define the slice. We return exposure drivers and the exact IDs."
        "</div>",
        unsafe_allow_html=True,
    )


# =====================================================================================
# TAB 2: DISCOVER MODE (FLEET / RISK RADAR VIEW)
# =====================================================================================
with tabs[1]:
    st.markdown('<div class="section-title">Fleet risk sweep</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-hint">'
        "System-wide scan. Ranks concentrated operational risk, recurring elevated conditions, "
        "and surfaces the most serious current events (Critical 5)."
        '</div>',
        unsafe_allow_html=True,
    )

    if st.button("Run Discover (Pinecone sweep)"):
        with st.spinner("Scanning dataset, ranking elevated conditions, preparing safety brief…"):
            result = run_agentic(
                "identify unique risk patterns, trends, spikes, high risk and critical events across the entire dataset"
            )

        mode = result.get("mode", "?")
        ans = result.get("answer") or {}
        dfx = result.get("df")

        st.markdown(f"**Agent mode:** `{mode}`")

        # ===== Executive summary block (large, highlighted) =====
        st.markdown(
            "<div class='summary-card'>"
            "<div class='summary-head'>Executive Summary</div>"
            f"<div class='summary-body'>{ans.get('executive_summary','Executive summary not available.')}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        run_meta = ans.get("run_metadata", {})
        sample_meta = ans.get("sample", {})

        # ===== Run context / stat cards =====
        st.markdown('<div class="section-title">Run context</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-hint">'
            "Scope, confidence and data source for this sweep."
            '</div>',
            unsafe_allow_html=True,
        )

        # Row 1
        st.markdown('<div class="stat-row">', unsafe_allow_html=True)
        colA, colB, colC = st.columns(3)
        with colA:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Sample size</div>
                    <div class="stat-value">{sample_meta.get('total_incidents','?')}</div>
                    <div class="stat-small">incidents analysed</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with colB:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Temporal confidence</div>
                    <div class="stat-value">{run_meta.get('temporal_confidence','?')}</div>
                    <div class="stat-small">trend / spike reliability</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with colC:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Generated (UTC)</div>
                    <div class="stat-value">{run_meta.get('generated_at','?')}</div>
                    <div class="stat-small">{sample_meta.get('analysis_date','')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Row 2
        st.markdown('<div class="stat-row" style="margin-top:1rem;">', unsafe_allow_html=True)
        colD, colE = st.columns(2)
        with colD:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Data source</div>
                    <div class="stat-value" style="font-size:1.1rem;">
                        {run_meta.get('pinecone_index','?')}
                    </div>
                    <div class="stat-small">Pinecone semantic index</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with colE:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Model / seed</div>
                    <div class="stat-value" style="font-size:1.1rem;">
                        {run_meta.get('summary_model','?')}
                    </div>
                    <div class="stat-small">
                        seed {run_meta.get('seed','?')} · internal early warning
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # ===== Method snapshot =====
        st.markdown('<div class="section-title">Method</div>', unsafe_allow_html=True)
        st.markdown(
            f"<div class='body-text'>"
            f"Analysed {sample_meta.get('total_incidents','?')} incidents on "
            f"{sample_meta.get('analysis_date','?')}. "
            f"{sample_meta.get('methodology','')}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ===== Highest-priority clusters =====
        st.markdown('<div class="section-title">Highest-priority clusters</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-hint">'
            "Risk-weighted groups of similar incidents. Higher = concentrated exposure that deserves attention."
            '</div>',
            unsafe_allow_html=True,
        )

        clusters_table = pd.DataFrame(ans.get("clusters_table", []))
        if not clusters_table.empty:
            preferred_cols = [
                "cluster",
                "label",
                "size",
                "pct_high_risk",
                "avg_risk",
                "risk_priority",
                "spike_factor",
                "top_phases",
                "top_causes",
                "examples",
            ]
            show_cols = [c for c in preferred_cols if c in clusters_table.columns]
            clusters_display = clusters_table[show_cols].copy()
            clusters_display = clusters_display.fillna("—")  # remove None/NaN ugliness
            st.dataframe(clusters_display, use_container_width=True)
        else:
            st.info("No clusters available.")

        # ===== Elevated-risk conditions =====
        st.markdown('<div class="section-title">Elevated-risk conditions</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-hint">'
            "Conditions where the chance of elevated risk is much higher than baseline."
            '</div>',
            unsafe_allow_html=True,
        )
        rule_lines = ans.get("rule_narratives", [])
        if rule_lines:
            for rl in rule_lines:
                st.markdown(
                    f"<div class='body-text'>• {rl}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("None surfaced at current threshold.")

        # ===== Critical 5 =====
        st.markdown('<div class="section-title">Critical 5 (escalate now)</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-hint">'
            "Most severe / closest-call events. These IDs should be reviewed immediately."
            '</div>',
            unsafe_allow_html=True,
        )
        crit5 = pd.DataFrame(ans.get("critical5", []))
        if not crit5.empty:
            crit5_display = crit5.copy().fillna("—")
            st.dataframe(crit5_display, use_container_width=True)
        else:
            st.info("No Critical 5 surfaced in this sample.")

        # ===== Patterns & exposure =====
        if isinstance(dfx, pd.DataFrame) and not dfx.empty:
            st.markdown('<div class="section-title">Patterns & exposure</div>', unsafe_allow_html=True)

            # Outliers
            st.markdown('<div class="table-title">Outliers</div>', unsafe_allow_html=True)
            if "anomaly" in dfx.columns:
                anomal = dfx.loc[
                    dfx["anomaly"] == True,
                    [
                        c for c in [
                            "report_number",
                            "miss_distance",
                            "altitude",
                            "flight_phase",
                            "cause_category",
                            "risk_level",
                        ]
                        if c in dfx.columns
                    ],
                ].head(20)
                if not anomal.empty:
                    st.dataframe(anomal.fillna("—"), use_container_width=True)
                else:
                    st.info("No statistical outliers flagged at 2% contamination.")
            else:
                st.info("No anomaly signal computed.")

            col_a, col_b = st.columns(2)

            # Incidents by month — guarded
            with col_a:
                try:
                    if "month" in dfx.columns:
                        m = (
                            dfx["month"]
                            .dropna()
                            .value_counts()
                            .sort_index()
                        )
                        # Only plot if there's signal. If everything is one month, it's not helpful.
                        if len(m.index) > 1 and int(m.sum()) > 0:
                            st.markdown('<div class="table-title">Incidents by month</div>', unsafe_allow_html=True)
                            fig, ax = plt.subplots()
                            ax.bar(m.index, m.values, color="#10b981")
                            ax.set_xticks(range(len(m.index)))
                            ax.set_xticklabels(m.index, rotation=45, ha="right")
                            ax.set_ylabel("Count")
                            st.pyplot(fig)
                except Exception:
                    # Totally suppress if month data is messy
                    pass

            # Phase × Risk heatmap
            with col_b:
                if "flight_phase" in dfx.columns and "risk_level" in dfx.columns:
                    phase_counts = dfx["flight_phase"].fillna("NA").value_counts()
                    top_phases = phase_counts.head(10).index.tolist()
                    filt = dfx[dfx["flight_phase"].fillna("NA").isin(top_phases)].copy()

                    if not filt.empty:
                        st.markdown('<div class="table-title">Phase × Risk heatmap</div>', unsafe_allow_html=True)

                        pivot = pd.crosstab(
                            filt["flight_phase"].fillna("NA"),
                            filt["risk_level"].fillna("NA"),
                        )
                        # sort rows by total count descending
                        pivot = pivot.loc[
                            pivot.sum(axis=1).sort_values(ascending=False).index
                        ]

                        if not pivot.empty:
                            fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
                            im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
                            ax.set_xticks(range(len(pivot.columns)))
                            ax.set_xticklabels(
                                pivot.columns,
                                rotation=45,
                                ha="right",
                                fontsize=8,
                            )
                            ax.set_yticks(range(len(pivot.index)))
                            ax.set_yticklabels(
                                pivot.index,
                                fontsize=8,
                            )
                            ax.set_xlabel("Risk level / score")
                            ax.set_ylabel("Flight phase")

                            # overlay counts
                            for i in range(pivot.shape[0]):
                                for j in range(pivot.shape[1]):
                                    ax.text(
                                        j, i,
                                        str(pivot.values[i, j]),
                                        ha="center",
                                        va="center",
                                        fontsize=7,
                                        color="white" if pivot.values[i, j] else "black",
                                    )

                            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            cbar.set_label("Count", fontsize=8)
                            st.pyplot(fig)

        # ===== Operational Watchlist =====
        st.markdown('<div class="section-title">Operational Watchlist</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-hint">'
            "Highest priority focus areas. These combine volume and severity uplift. "
            "This is what Ops / Safety should action first."
            '</div>',
            unsafe_allow_html=True,
        )

        wl = ans.get("watchlist", [])
        if wl:
            for item in wl:
                st.markdown(
                    f"""
                    <div class="watch-item">
                        <div class="watch-context">{item['context']}</div>
                        <div class="watch-body">{item['severity_statement']}</div>
                        <div class="watch-meta">
                            <strong>Risk Index:</strong> {item.get('risk_index','—')}<br/>
                            <strong>Action:</strong> {item.get('action_hint','Requires operational review.')}<br/>
                            <strong>Evidence IDs:</strong> {", ".join(item.get("evidence", []))}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No watchlist items at current thresholds.")

        st.markdown(
            '<div class="small-note">'
            "Every claim above is backed by incident counts and report_numbers. "
            "This dashboard shows operational exposure, not regulatory fault."
            '</div>',
            unsafe_allow_html=True,
        )


# =====================================================================================
# TAB 3: OFFLINE CSV MODE
# =====================================================================================
with tabs[2]:
    st.markdown('<div class="section-title">Offline analysis from CSV</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-hint">'
        "Run the same safety analytics pipeline offline on a CSV dump "
        "(report_number, datetime, flight_phase, risk_level, miss_distance, etc.)."
        '</div>',
        unsafe_allow_html=True,
    )

    st.write("Coming soon: upload, run, export a briefing pack.")
