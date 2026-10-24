"""Model Drift Watchdog — Streamlit PoC.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import html
import re
import time
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agents.common import AGENT_META
from data_gen import SCENARIOS, evaluate
from graph import LANGGRAPH_AVAILABLE
from knowledge import CATEGORIES, clear_memory, get_kb, load_memory
from llm import OLLAMA_MODEL, ollama_status
from policy import (ALERT, CRITICAL_FEATURES, FIX_PIPELINE, MAX_ACCURACY_DROP, MAX_MISSING_RATE,
                    MAX_RETRIES, MIN_AUC_GAIN, MIN_SAMPLE_SIZE, NO_ACTION, P_VALUE_ALPHA,
                    PSI_MAJOR, PSI_MODERATE, RETRAIN)
from runner import load_context, run_scenario
from validation import robustness_runs, scenario_runs

st.set_page_config(page_title="Model Drift Watchdog", page_icon="🛰️", layout="wide")

# ---------------------------------------------------------------------------
# Look & feel
# ---------------------------------------------------------------------------
TEAL, BLUE, MUTED = "#14B8A6", "#3B82F6", "#9CA3AF"
DECISION_COLORS = {NO_ACTION: "#22C55E", ALERT: "#F59E0B", RETRAIN: "#EF4444", FIX_PIPELINE: "#A855F7"}
DECISION_ICONS = {NO_ACTION: "✅", ALERT: "⚠️", RETRAIN: "🔁", FIX_PIPELINE: "🛠️"}
SEVERITY_COLORS = {"stable": "#22C55E", "moderate": "#F59E0B", "major": "#EF4444", "n/a": MUTED}
STATUS_COLORS = {"ok": "#22C55E", "warn": "#F59E0B", "retry": "#3B82F6", "fail": "#EF4444", "info": MUTED}

st.markdown(
    """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
h1, h2, h3 {letter-spacing: -0.01em;}
.mdw-sub {color:#9CA3AF; margin-top:-0.6rem; margin-bottom:1.2rem;}
.mdw-card {background:#111A2E; border:1px solid rgba(20,184,166,0.18); border-radius:12px; padding:14px 16px; height:100%;}
.mdw-kpi-label {color:#9CA3AF; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.06em;}
.mdw-kpi-value {font-size:1.45rem; font-weight:700; margin-top:2px;}
.mdw-kpi-sub {color:#9CA3AF; font-size:0.8rem;}
.mdw-pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.75rem; font-weight:600;
           border:1px solid currentColor; margin-right:6px;}
.mdw-step {background:#111A2E; border-left:4px solid #14B8A6; border-radius:8px; padding:10px 14px; margin-bottom:8px;}
.mdw-step-head {display:flex; justify-content:space-between; align-items:center; gap:8px;}
.mdw-step-agent {font-weight:700;}
.mdw-step-meta {color:#9CA3AF; font-size:0.78rem;}
.mdw-step-body {margin-top:4px; font-size:0.92rem; color:#E5E7EB;}
.mdw-decision {border-radius:14px; padding:18px 20px; border:2px solid; background:#111A2E;}
.mdw-decision-title {font-size:1.9rem; font-weight:800;}
.mdw-alert {font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background:#0F172A; border:1px solid #1F2A44;
            border-radius:10px; padding:12px 14px; font-size:0.86rem; white-space:pre-wrap;}
.mdw-muted {color:#9CA3AF;}
</style>
""",
    unsafe_allow_html=True,
)


def pill(text: str, color: str) -> str:
    return f'<span class="mdw-pill" style="color:{color}">{text}</span>'


def md_inline(text: str) -> str:
    """Tiny markdown → HTML for **bold** and `code` inside custom HTML blocks."""
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.+?)`", r'<code style="color:#5EEAD4; background:#0F172A; padding:0 4px; border-radius:4px">\1</code>', text)


def kpi(col, label: str, value: str, sub: str = "", color: str = "#E5E7EB") -> None:
    col.markdown(
        f'<div class="mdw-card"><div class="mdw-kpi-label">{label}</div>'
        f'<div class="mdw-kpi-value" style="color:{color}">{value}</div>'
        f'<div class="mdw-kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB"), title=dict(x=0, xanchor="left", font=dict(size=14)),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=60))
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
    return fig


# ---------------------------------------------------------------------------
# Cached context & session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training the champion model on reference data…")
def context():
    return load_context()


@st.cache_data(ttl=30, show_spinner=False)
def llm_status():
    return ollama_status()


reference_df, original_champion, reference_test_df, original_baseline = context()

ss = st.session_state
ss.setdefault("result", None)
ss.setdefault("champion", original_champion)
ss.setdefault("baseline", original_baseline)
ss.setdefault("registry", [{"version": "v1", "deployed": "initial deployment", "source": "Reference data (5,000 rows)",
                            "auc": original_baseline["auc"], "approved_by": "—"}])
ss.setdefault("validation", None)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛰️ Watchdog controls")
    scenario = st.selectbox("Production batch scenario", list(SCENARIOS), index=0)
    spec = SCENARIOS[scenario]
    st.caption(f"**Injected change:** {spec.description}")
    st.caption(f"**Ground truth (for validation):** {DECISION_ICONS[spec.expected]} {spec.expected}")

    labels = st.toggle("True loan outcomes available", value=True,
                       help="In lending, whether a loan defaults is known only months later. "
                            "Turn off to monitor with input and prediction drift only.")
    status = llm_status()
    use_llm = st.toggle("Use local LLM (Ollama) for explanations", value=True)
    if status["available"]:
        st.caption(f"🟢 Ollama connected · `{OLLAMA_MODEL}`")
    elif status["server"]:
        st.caption(f"🟡 Ollama running but `{OLLAMA_MODEL}` not pulled — using mock LLM")
    else:
        st.caption("⚪ Ollama not running — mock LLM will write explanations")

    with st.expander("Demo controls (Critic loop)"):
        batch_size = st.slider("Batch size (rows)", 100, 1000, 1000, 50,
                               help=f"Below {MIN_SAMPLE_SIZE} rows the Critic asks the Profiler for a wider window. "
                                    "Very small batches escalate to a human after 2 retries.")
        faulty = st.checkbox("Inject a faulty first decision",
                             help="The Decision agent's first call uses a naive rule. Watch the Critic catch it.")
        use_memory = st.toggle("Agent memory (remember past runs)", value=True,
                               help="The Reporter recalls similar past runs from the run history.")
        engine_label = st.radio("Orchestrator", ["LangGraph", "Fallback runner"], horizontal=True,
                                index=0 if LANGGRAPH_AVAILABLE else 1, disabled=not LANGGRAPH_AVAILABLE)

    run = st.button("▶ Run Watchdog", type="primary", width="stretch")

    st.divider()
    current = ss.registry[-1]
    st.caption(f"**Deployed model:** {current['version']} · AUC {current['auc']:.3f}")
    if len(ss.registry) > 1 and st.button("Reset to original champion (v1)", width="stretch"):
        ss.champion, ss.baseline = original_champion, original_baseline
        ss.registry = ss.registry[:1]
        ss.result = None
        st.rerun()
    st.caption("Harsh Sharma (24070126067) · Divyansh Agarwal (24070126056)")

if run:
    with st.spinner("Agents at work…"):
        t0 = time.perf_counter()
        res = run_scenario(
            scenario, labels_available=labels, batch_size=batch_size, use_llm=use_llm,
            inject_faulty_decision=faulty,
            use_memory=use_memory,
            engine="langgraph" if engine_label == "LangGraph" else "fallback",
            champion=ss.champion, baseline=ss.baseline,
        )
        res["runtime_s"] = time.perf_counter() - t0
        res["human_action"] = None
        res["settings"] = {"labels": labels, "batch_size": batch_size, "faulty": faulty, "use_llm": use_llm}
        ss.result = res

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# 🛰️ Model Drift Watchdog")
st.markdown('<div class="mdw-sub">Agentic AI that <b>detects</b>, <b>decides</b>, <b>validates</b> and <b>acts</b> '
            "when a deployed credit-risk model goes stale · MLOps for digital lending</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Graph diagram
# ---------------------------------------------------------------------------
NODE_LABELS = {
    "ui": "🖥️ Streamlit UI", "planner": "🧭 Planner", "profiler": "🔎 Data Profiler",
    "replanner": "🧭 Planner\\n(re-plan)", "detector": "📡 Drift Detector", "decision": "🔀 Decision Agent",
    "critic": "🛡️ Critic Agent", "reporter": "📝 Reporter / Action", "human": "👤 Human approval",
    "kb": "📚 Knowledge Base\\n(concepts · policy · runbooks)",
}
STATIC_EDGES = [("ui", "planner", ""), ("planner", "profiler", ""), ("profiler", "detector", "data OK"),
                ("profiler", "replanner", "bad data"), ("replanner", "decision", "Fix pipeline"),
                ("detector", "decision", ""), ("decision", "critic", ""), ("critic", "reporter", "approve / escalate"),
                ("critic", "profiler", "retry: wider window"), ("critic", "detector", "retry: bootstrap"),
                ("critic", "decision", "retry: revise"), ("reporter", "human", "")]


def graph_dot(path: list[str] | None = None) -> str:
    visited = set(path or [])
    transitions = Counter(zip(path, path[1:])) if path else Counter()
    if path:
        visited |= {"ui"}
        transitions[("ui", "planner")] += 1
        if "reporter" in visited:
            transitions[("reporter", "human")] += 1
            visited.add("human")
        visited.add("kb")
    lines = ['digraph G {', 'bgcolor="transparent"; rankdir=TB; nodesep=0.35; ranksep=0.35;',
             'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 fontcolor="#E5E7EB" '
             'color="#334155" fillcolor="#111A2E" penwidth=1.4];',
             'edge [fontname="Helvetica" fontsize=9 color="#475569" fontcolor="#9CA3AF" arrowsize=0.7];']
    for n, label in NODE_LABELS.items():
        if path and n in visited:
            fill = {"critic": "#2E1F4F", "kb": "#2A2350"}.get(n, "#0F3D3A")
            lines.append(f'{n} [label="{label}" fillcolor="{fill}" color="{TEAL}" penwidth=2.2];')
        elif path:
            lines.append(f'{n} [label="{label}" fontcolor="#64748B" color="#1E293B"];')
        else:
            lines.append(f'{n} [label="{label}"];')
    for a, b, label in STATIC_EDGES:
        retry = a == "critic" and b in ("profiler", "detector", "decision")
        count = transitions.get((a, b), 0)
        attrs = [f'label="{label}{" ×" + str(count) if count > 1 else ""}"'] if label or count > 1 else []
        if retry:
            attrs.append('style=dashed constraint=false')
        if path and count:
            attrs.append(f'color="{"#F59E0B" if retry else TEAL}" penwidth=2.4 fontcolor="#E5E7EB"')
        lines.append(f'{a} -> {b} [{" ".join(attrs)}];')
    kb_color = "#A78BFA" if path else "#475569"
    for target in ("planner", "decision", "critic", "reporter"):
        lines.append(f'kb -> {target} [style=dotted arrowhead=none constraint=false color="{kb_color}"];')
    lines.append('{rank=same; kb; profiler}')
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Landing view
# ---------------------------------------------------------------------------
res = ss.result
if res is None:
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("### How it works")
        st.markdown(
            "A credit-risk model approves loans. When applicant data changes, it silently gets worse — "
            "and loan defaults are only known months later. The Watchdog checks every new batch:\n\n"
            "1. **Planner** builds a plan from the data (and re-plans if the data is broken)\n"
            "2. **Data Profiler** checks schema, missing values and sample size\n"
            "3. **Drift Detector** runs KS / Chi-square / PSI tests and prediction drift\n"
            "4. **Decision Agent** applies a fixed policy: No action · Alert · Retrain · Fix pipeline\n"
            "5. **Critic** validates the evidence and decision — retries up to 2× or escalates\n"
            "6. **Reporter / Action** writes the report, alerts the right team and trains a challenger\n"
            "7. **Human** approves before any new model is deployed\n\n"
            f"📚 Agents retrieve and cite a **knowledge base** of {len(get_kb())} documents "
            "(drift concepts, decision policy, runbooks, lending domain) and remember past runs."
        )
        st.info("👈 Pick a scenario in the sidebar and press **Run Watchdog**.")
        st.markdown("#### Scenarios")
        cols = st.columns(2)
        for i, (name, s) in enumerate(SCENARIOS.items()):
            c = DECISION_COLORS[s.expected]
            cols[i % 2].markdown(
                f'<div class="mdw-card" style="border-top:3px solid {c}; margin-bottom:10px">'
                f'<b>{name}</b><br><span class="mdw-muted">{s.description}</span><br>'
                f'<span style="color:{c}; font-weight:600">{DECISION_ICONS[s.expected]} Expected: {s.expected}</span></div>',
                unsafe_allow_html=True)
    with right:
        st.markdown("### Architecture")
        st.graphviz_chart(graph_dot(), width="stretch")
    st.stop()


# ---------------------------------------------------------------------------
# Results header
# ---------------------------------------------------------------------------
decision = res["decision"]
dcolor = DECISION_COLORS[decision]
q = res["quality_report"]
table = res.get("drift_table") if q["data_ok"] else None
expected = SCENARIOS[res["scenario"]].expected

k1, k2, k3, k4, k5 = st.columns(5)
kpi(k1, "Decision", f"{DECISION_ICONS[decision]} {decision}",
    ("Escalated to human" if res["escalated"] else f"Ground truth: {expected} " + ("✓" if decision == expected else "✗")), dcolor)
verdict_txt = {"approve": "Approved", "escalate": "Escalated"}.get(res["critic_verdict"], res["critic_verdict"])
kpi(k2, "Critic verdict", f"🛡️ {verdict_txt}", f"{res['retry_count']} retr{'y' if res['retry_count'] == 1 else 'ies'} (max {MAX_RETRIES})",
    "#EF4444" if res["escalated"] else TEAL)
if table:
    n_drift = sum(r["severity"] != "stable" for r in table)
    kpi(k3, "Drifted features", f"{n_drift} / {len(table)}",
        f"Prediction PSI {res['prediction_drift']['psi']:.3f}", SEVERITY_COLORS["major" if any(r['severity'] == 'major' for r in table) else ("moderate" if n_drift else "stable")])
else:
    kpi(k3, "Data quality", "❌ Failed", "; ".join(q["issues"]).replace("`", "")[:60], "#A855F7")
kpi(k4, "Rows checked", f"{q['rows']:,}", f"vs {len(reference_df):,} reference rows")
kpi(k5, "Run time", f"{res['runtime_s']:.1f} s", f"{res['engine']} · {len(res['trace'])} agent steps")

st.write("")
alert = res["alert"]
banner = st.error if alert["level"] == "high" else (st.warning if alert["level"] == "warning" else st.success)
banner(f"**{alert['title']}** → routed to {alert['route']}")

tab_trace, tab_drift, tab_dec, tab_rep, tab_kb, tab_val = st.tabs(
    ["🧭 Agent Trace", "📡 Drift Analysis", "🛡️ Decision & Critic", "📝 Report & Action", "📚 Knowledge Base",
     "✅ Validation"])

# ---------------------------------------------------------------------------
# Tab 1 — Agent trace
# ---------------------------------------------------------------------------
with tab_trace:
    left, right = st.columns([1, 1.25])
    with left:
        st.markdown("#### Path taken through the graph")
        st.graphviz_chart(graph_dot(res["path"]), width="stretch")
        st.caption("Highlighted nodes ran in this cycle. Orange dashed edges are Critic retries.")
        st.markdown("#### Task plan")
        icons = {"planned": "⬜", "skipped": "⏭️", "done": "✅"}
        ran = {e["agent"] for e in res["trace"]}
        for t in res["plan"]:
            status = "done" if t["status"] == "planned" and t["agent"] in ran else t["status"]
            st.markdown(f"{icons.get(status, '•')} **{t['agent']}** — {t['task']}")
    with right:
        st.markdown("#### Step-by-step agent trace")
        for e in res["trace"]:
            color = STATUS_COLORS.get(e["status"], TEAL)
            st.markdown(
                f'<div class="mdw-step" style="border-left-color:{color}">'
                f'<div class="mdw-step-head"><span class="mdw-step-agent">{e["step"]}. {e["icon"]} {e["agent"]} '
                f'<span class="mdw-step-meta">· {e["role"]}</span></span>'
                f'<span>{pill(e["action"], color)}<span class="mdw-step-meta">+{e["t_ms"]} ms</span></span></div>'
                f'<div class="mdw-step-body">{md_inline(e["summary"])}</div>'
                + (f'<div style="margin-top:6px">{"".join(pill("📚 " + k, "#A78BFA") for k in e.get("kb", []))}</div>'
                   if e.get("kb") else "") + '</div>',
                unsafe_allow_html=True)
        if res.get("human_action"):
            h = res["human_action"]
            st.markdown(
                f'<div class="mdw-step" style="border-left-color:{TEAL}"><div class="mdw-step-head">'
                f'<span class="mdw-step-agent">{len(res["trace"]) + 1}. 👤 Human <span class="mdw-step-meta">· Approver</span></span>'
                f'{pill(h["action"], TEAL)}</div><div class="mdw-step-body">{h["detail"]}</div></div>',
                unsafe_allow_html=True)
        with st.expander("Raw trace (JSON)"):
            st.json([{k: v for k, v in e.items()} for e in res["trace"]], expanded=False)

# ---------------------------------------------------------------------------
# Tab 2 — Drift analysis
# ---------------------------------------------------------------------------
with tab_drift:
    prod = res["production_df"]
    if not table:
        st.warning("**Drift tests were skipped by the Planner** — the data itself is broken, so any drift result would be "
                   "misleading. Showing the data-quality evidence instead.")
        mr = pd.Series(q["missing_rates"]).sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=mr.index, y=mr.values * 100,
                               marker_color=["#A855F7" if v > MAX_MISSING_RATE else TEAL for v in mr.values],
                               text=[f"{v:.1%}" for v in mr.values], textposition="outside"))
        fig.add_hline(y=MAX_MISSING_RATE * 100, line_dash="dash", line_color="#F59E0B",
                      annotation_text=f"limit {MAX_MISSING_RATE:.0%}", annotation_position="top right")
        fig.update_layout(title="Missing values per feature (%)", yaxis_title="% missing")
        st.plotly_chart(style_fig(fig, 380), width="stretch")
        st.dataframe(pd.DataFrame([{"Feature": c, "Missing %": f"{r:.1%}", "Status": "❌ above limit" if r > MAX_MISSING_RATE else "✅ ok"}
                                   for c, r in q["missing_rates"].items()]), hide_index=True, width="stretch")
    else:
        df = pd.DataFrame(table)
        c1, c2 = st.columns([1.1, 1])
        with c1:
            d = df.sort_values("psi")
            fig = go.Figure(go.Bar(
                x=d["psi"], y=[f"{f} ★" if c else f for f, c in zip(d["feature"], d["critical"])], orientation="h",
                marker_color=[SEVERITY_COLORS[s] for s in d["severity"]],
                text=[f"{p:.3f}" for p in d["psi"]], textposition="outside"))
            fig.add_vline(x=PSI_MODERATE, line_dash="dash", line_color="#F59E0B", annotation_text="moderate ",
                          annotation_position="bottom left", annotation_font_color="#F59E0B")
            fig.add_vline(x=PSI_MAJOR, line_dash="dash", line_color="#EF4444", annotation_text=" major",
                          annotation_position="bottom right", annotation_font_color="#EF4444")
            fig.update_layout(title="Population Stability Index per feature (★ = critical)", xaxis_title="PSI",
                              xaxis_range=[0, max(0.35, float(d["psi"].max()) * 1.22)])
            st.plotly_chart(style_fig(fig, 340), width="stretch")
        with c2:
            pdrift = res["prediction_drift"]
            fig = go.Figure()
            fig.add_histogram(x=pdrift["ref_scores"], name="Reference", histnorm="probability", nbinsx=30,
                              marker_color=BLUE, opacity=0.65)
            fig.add_histogram(x=pdrift["prod_scores"], name="Production", histnorm="probability", nbinsx=30,
                              marker_color="#F59E0B", opacity=0.65)
            fig.update_layout(barmode="overlay", title=f"Prediction drift — champion risk scores (PSI {pdrift['psi']:.3f})",
                              xaxis_title="Predicted default probability")
            st.plotly_chart(style_fig(fig, 340), width="stretch")

        show = pd.DataFrame({
            "Feature": df["feature"],
            "Critical": df["critical"].map({True: "★", False: ""}),
            "Test": df["test"],
            "Statistic": df["statistic"],
            "p-value": df["p_value"].map(lambda p: f"{p:.2e}" if p < 0.001 else f"{p:.3f}"),
            f"Significant (p<{P_VALUE_ALPHA})": df["drifted"].map({True: "yes", False: "no"}),
            "PSI": df["psi"],
            "Severity": df["severity"],
            "Bootstrap PSI 95% CI": df["bootstrap"].map(lambda b: f"[{b['ci_low']:.3f}, {b['ci_high']:.3f}]" if b else ""),
        })
        st.dataframe(show, hide_index=True, width="stretch",
                     column_config={"PSI": st.column_config.ProgressColumn("PSI", min_value=0, max_value=max(0.3, float(df["psi"].max())), format="%.3f")})

        st.markdown("#### Reference vs production distribution")
        feat = st.selectbox("Feature", list(df["feature"]),
                            index=int(df["psi"].values.argmax()), key="feat_pick")
        ref_col, prod_col = reference_df[feat], prod[feat]
        fig = go.Figure()
        if df.set_index("feature").loc[feat, "type"] == "numeric":
            fig.add_histogram(x=ref_col, name="Reference (training)", histnorm="probability", nbinsx=40, marker_color=BLUE, opacity=0.6)
            fig.add_histogram(x=prod_col, name="Production batch", histnorm="probability", nbinsx=40, marker_color="#F59E0B", opacity=0.6)
            fig.update_layout(barmode="overlay")
        else:
            r_share = ref_col.astype(str).value_counts(normalize=True).sort_index()
            p_share = prod_col.astype(str).value_counts(normalize=True).reindex(r_share.index, fill_value=0)
            fig.add_bar(x=r_share.index, y=r_share.values, name="Reference (training)", marker_color=BLUE)
            fig.add_bar(x=p_share.index, y=p_share.values, name="Production batch", marker_color="#F59E0B")
            fig.update_layout(barmode="group", yaxis_tickformat=".0%")
        row = df.set_index("feature").loc[feat]
        fig.update_layout(title=f"{feat} — {row['test']} p={row['p_value']:.3g} · PSI {row['psi']:.3f} ({row['severity']})")
        st.plotly_chart(style_fig(fig, 340), width="stretch")

        if res.get("performance"):
            p = res["performance"]
            st.markdown("#### Performance vs deployment baseline (labels available)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{p['current_accuracy']:.1%}", f"{-p['accuracy_drop'] * 100:+.1f} pts", delta_color="normal")
            m2.metric("AUC", f"{p['current_auc']:.3f}", f"{-p['auc_drop']:+.3f}")
            m3.metric("Default rate", f"{p['current_default_rate']:.1%}",
                      f"{(p['current_default_rate'] - p['baseline_default_rate']) * 100:+.1f} pts", delta_color="inverse")
            st.caption(f"Retrain trigger: accuracy drop > {MAX_ACCURACY_DROP * 100:.0f} points.")
        else:
            st.caption("Performance check skipped — true loan outcomes are not available yet (label delay).")

# ---------------------------------------------------------------------------
# Tab 3 — Decision & Critic
# ---------------------------------------------------------------------------
with tab_dec:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        ev = "".join(f"<li>{md_inline(e)}</li>" for e in res["decision_evidence"])
        st.markdown(
            f'<div class="mdw-decision" style="border-color:{dcolor}">'
            f'<div class="mdw-kpi-label">Recommended action</div>'
            f'<div class="mdw-decision-title" style="color:{dcolor}">{DECISION_ICONS[decision]} {decision}</div>'
            f'<div style="margin-top:6px"><b>Rule fired:</b> {md_inline(res["decision_rule"])}</div>'
            f'<div style="margin-top:6px"><b>Evidence:</b><ul style="margin-bottom:0">{ev}</ul></div>'
            + ('<div style="margin-top:8px; color:#EF4444"><b>🚨 Escalated to a human</b> — the Critic could not validate this '
               'after 2 retries, so no automatic action is taken.</div>' if res["escalated"] else "")
            + "</div>", unsafe_allow_html=True)
        st.write("")
        st.markdown("#### Decision policy")
        policy_rows = [
            (FIX_PIPELINE, f"Missing > {MAX_MISSING_RATE:.0%} on any feature, or schema mismatch"),
            (RETRAIN, f"Major drift (PSI > {PSI_MAJOR}) on a critical feature, prediction PSI > {PSI_MAJOR}, or accuracy drop > {MAX_ACCURACY_DROP * 100:.0f} pts"),
            (ALERT, f"Moderate drift ({PSI_MODERATE}–{PSI_MAJOR}) anywhere, or major drift on a non-critical feature"),
            (NO_ACTION, "Everything stable"),
        ]
        for name, cond in policy_rows:
            active = name == decision
            st.markdown(
                f'<div class="mdw-card" style="margin-bottom:6px; padding:8px 12px; '
                f'border:{"2px solid " + DECISION_COLORS[name] if active else "1px solid #1E293B"}; opacity:{1 if active else 0.55}">'
                f'<span style="color:{DECISION_COLORS[name]}; font-weight:700">{DECISION_ICONS[name]} {name}</span>'
                f'{" ← fired" if active else ""}<br><span class="mdw-muted" style="font-size:0.85rem">{cond}</span></div>',
                unsafe_allow_html=True)
        st.caption("Critical features: " + ", ".join(CRITICAL_FEATURES))
    with c2:
        st.markdown("#### Critic validation rounds")
        for h in res["critic_history"]:
            vcol = {"approve": "#22C55E", "retry": "#3B82F6", "escalate": "#EF4444"}[h["verdict"]]
            label = {"approve": "✅ Approved", "retry": f"🔁 Retry → {h['target']}", "escalate": "🚨 Escalate"}[h["verdict"]]
            st.markdown(f"**Round {h['round']}** — decision under review: **{h['decision']}** "
                        + pill(label, vcol), unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([{
                "Check": c["check"],
                "Result": {True: "✅ pass", False: "❌ fail", None: "➖ n/a"}[c["passed"]],
                "Detail": c["detail"].replace("`", ""),
                "If it fails": {"profiler": "Profiler: wider window", "detector": "Detector: bootstrap re-check",
                                "decision": "Decision: revise"}[c["on_fail"]],
            } for c in h["checks"]]), hide_index=True, width="stretch")
        st.caption(f"Max {MAX_RETRIES} retries, then escalate to a human. Decisions come from statistics and rules — "
                   "the LLM only explains them.")

# ---------------------------------------------------------------------------
# Tab 4 — Report & action
# ---------------------------------------------------------------------------
with tab_rep:
    c1, c2 = st.columns([1.15, 1])
    with c1:
        src = res["explanation_source"]
        st.markdown("#### Plain-English summary " + pill("Ollama · " + src.split(":", 1)[1] if src.startswith("ollama") else "Mock LLM (template)",
                                                         TEAL if src.startswith("ollama") else MUTED), unsafe_allow_html=True)
        st.markdown(res["explanation"])
        st.markdown("#### Drift report")
        with st.container(border=True):
            st.markdown(res["report"])
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        md = (f"# Model Drift Watchdog report\n\nGenerated {stamp}\n\n{res['report']}\n\n## Summary\n\n{res['explanation']}\n\n"
              f"## Agent trace\n\n" + "\n".join(f"{e['step']}. **{e['agent']}** — {e['action']}: {e['summary']}" for e in res["trace"]))
        st.download_button("⬇️ Download report (Markdown)", md, file_name=f"drift_report_{res['scenario'].replace(' ', '_').lower()}.md")
    with c2:
        st.markdown("#### Alert sent")
        st.markdown(f'<div class="mdw-alert">To: {alert["route"]}\n{alert["title"]}\n\n{alert["message"].replace("`", "")}</div>',
                    unsafe_allow_html=True)
        if res.get("incident"):
            inc = res["incident"]
            st.markdown("#### Incident ticket")
            st.markdown(f'<div class="mdw-card" style="border-left:4px solid #A855F7"><b>{inc["id"]}</b> · {inc["owner"]}<br>'
                        f'{inc["title"]}<br><span class="mdw-muted">Affected: {inc["affected"]}</span><br>{inc["action"]}</div>',
                        unsafe_allow_html=True)

        comp = res.get("model_comparison")
        if comp:
            st.markdown("#### Champion vs challenger (fresh, unseen data)")
            cmp_df = pd.DataFrame({
                "Metric": ["AUC", "Accuracy"],
                "Champion (current)": [comp["champion"]["auc"], comp["champion"]["accuracy"]],
                "Challenger (retrained)": [comp["challenger"]["auc"], comp["challenger"]["accuracy"]],
            })
            fig = go.Figure()
            fig.add_bar(x=cmp_df["Metric"], y=cmp_df["Champion (current)"], name="Champion", marker_color=BLUE,
                        text=[f"{v:.3f}" for v in cmp_df["Champion (current)"]], textposition="outside")
            fig.add_bar(x=cmp_df["Metric"], y=cmp_df["Challenger (retrained)"], name="Challenger", marker_color=TEAL,
                        text=[f"{v:.3f}" for v in cmp_df["Challenger (retrained)"]], textposition="outside")
            fig.update_layout(barmode="group", yaxis_range=[0.5, 1.0], title=f"AUC gain {comp['auc_gain']:+.3f} (needs ≥ {MIN_AUC_GAIN})")
            st.plotly_chart(style_fig(fig, 300), width="stretch")
            st.caption(f"Challenger trained on {comp['training_rows']:,} rows (reference + {comp['recent_rows']:,} recent labelled rows, "
                       f"recent weighted 3×) in {comp['train_seconds']} s · evaluated on {comp['holdout_rows']:,} fresh rows.")

        st.markdown("#### 👤 Human approval")
        h = res.get("human_action")
        if h:
            (st.success if h["action"] == "Approved" else st.info)(f"**{h['action']}** — {h['detail']}")
        elif res["escalated"]:
            st.error("The Critic escalated this run. Review the evidence and assign it.")
            if st.button("Acknowledge & assign to ML team", width="stretch"):
                res["human_action"] = {"action": "Acknowledged", "detail": "Escalation assigned to the ML team for manual review."}
                st.rerun()
        elif comp:
            if comp["eligible"]:
                st.write("The challenger beats the champion on fresh data. Deploy it?")
                a, b = st.columns(2)
                if a.button("✅ Approve & deploy", type="primary", width="stretch"):
                    version = f"v{len(ss.registry) + 1}"
                    ss.champion = res["challenger"]
                    ss.baseline = evaluate(res["challenger"], reference_test_df)
                    ss.registry.append({"version": version, "deployed": datetime.now().strftime("%H:%M:%S"),
                                        "source": f"Retrained after '{res['scenario']}'", "auc": comp["challenger"]["auc"],
                                        "approved_by": "Human reviewer"})
                    res["human_action"] = {"action": "Approved", "detail": f"Challenger deployed as {version} (AUC {comp['challenger']['auc']:.3f})."}
                    st.rerun()
                if b.button("❌ Reject", width="stretch"):
                    res["human_action"] = {"action": "Rejected", "detail": "Champion kept; challenger archived."}
                    st.rerun()
            else:
                st.warning(f"Challenger gain {comp['auc_gain']:+.3f} is below the {MIN_AUC_GAIN} bar — not eligible. Champion kept.")
        else:
            st.caption("No model change proposed in this cycle — nothing to approve.")

        st.markdown("#### Model registry")
        st.dataframe(pd.DataFrame(ss.registry).rename(columns={"version": "Version", "deployed": "Deployed", "source": "Trained on",
                                                                "auc": "AUC", "approved_by": "Approved by"}),
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Tab — Knowledge base
# ---------------------------------------------------------------------------
with tab_kb:
    kb = get_kb()
    refs = res.get("kb_refs", [])
    st.markdown("#### Knowledge used in this run")
    st.caption(f"The agents retrieved {len({r['id'] for r in refs})} of {len(kb)} documents with TF-IDF search "
               "and cited them in their steps. Retrieval explains and guides actions; it never changes a decision.")
    rb = res.get("runbook_id")
    if rb:
        doc = kb.get(rb)
        steps = "".join(f"<li>{md_inline(x)}</li>" for x in doc.steps)
        st.markdown(f'<div class="mdw-card" style="border-left:4px solid #A78BFA; margin-bottom:12px">'
                    f'<b>📘 Runbook followed: {doc.id} — {html.escape(doc.title)}</b>'
                    f'<ol style="margin:6px 0 0 0">{steps}</ol></div>', unsafe_allow_html=True)
    if refs:
        st.dataframe(pd.DataFrame([{"ID": r["id"], "Title": r["title"], "Category": r["category"],
                                    "Cited by": r["agent"], "Why": r["reason"], "Match score": r["score"]} for r in refs]),
                     hide_index=True, width="stretch")
        for doc_id in sorted({r["id"] for r in refs}):
            d = kb.get(doc_id)
            with st.expander(f"{d.id} — {d.title}"):
                st.markdown(re.sub(r"^# .*\n", "", d.body))

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("#### Search the knowledge base")
        q = st.text_input("Ask about drift, policy or what to do", value="what to do when data is missing",
                          key="kb_query")
        cat = st.selectbox("Category", ["all"] + CATEGORIES, key="kb_cat")
        hits = kb.search(q, k=5, category=None if cat == "all" else cat) if q.strip() else []
        if not hits:
            st.caption("No matching documents.")
        for d, score in hits:
            with st.expander(f"{d.id} — {d.title}  ·  score {score:.2f}"):
                st.caption(f"{d.category} · tags: {', '.join(d.tags)} · `{d.path}`")
                st.markdown(re.sub(r"^# .*\n", "", d.body))
    with c2:
        st.markdown("#### Agent memory")
        mem = load_memory()
        if res.get("memory_hits"):
            st.success(f"The Reporter recalled {len(res['memory_hits'])} similar past run(s) for this result.")
        if mem:
            st.dataframe(pd.DataFrame(mem[::-1][:15]).rename(columns={
                "time": "Time", "scenario": "Scenario", "decision": "Decision", "drifted": "Drifted features",
                "retries": "Retries", "escalated": "Escalated", "rows": "Rows"}), hide_index=True, width="stretch")
            if st.button("Clear memory", key="clear_mem"):
                clear_memory()
                st.rerun()
        else:
            st.caption("No runs remembered yet. Turn on *Agent memory* in Demo controls and run a few scenarios.")
        st.markdown("#### Browse")
        for c in CATEGORIES:
            docs = [d for d in kb.docs if d.category == c]
            with st.expander(f"{c.title()} ({len(docs)})"):
                for d in docs:
                    st.markdown(f"**{d.id}** — {d.title}")

# ---------------------------------------------------------------------------
# Tab 5 — Validation
# ---------------------------------------------------------------------------
with tab_val:
    st.markdown("#### Validation against injected ground truth")
    st.write("Every scenario has a known, injected change, so we can check the agents' output. "
             "The suite runs all 4 scenarios with and without labels, plus robustness tests for the Critic loop.")
    if st.button("▶ Run validation suite (15 workflow runs)", type="primary"):
        with st.spinner("Running all scenarios and robustness tests…"):
            ss.validation = (scenario_runs(use_llm=False), robustness_runs(use_llm=False))
    if ss.validation is not None:
        s_df, r_df = ss.validation
        v1, v2, v3, v4 = st.columns(4)
        kpi(v1, "Decision accuracy", f"{int(s_df['Match'].sum())}/{len(s_df)}", "scenario × labels on/off", "#22C55E")
        kpi(v2, "Detection recall", f"{s_df['Recall'].mean():.0%}", "injected features flagged", TEAL)
        kpi(v3, "Detection precision", f"{s_df['Precision'].mean():.0%}", "flagged features truly injected", TEAL)
        gains = s_df["AUC gain (retrain)"].dropna()
        kpi(v4, "Retrain AUC gain", f"{gains.mean():+.3f}" if len(gains) else "—", "challenger vs champion, fresh data", BLUE)
        st.write("")
        st.dataframe(s_df.assign(Match=s_df["Match"].map({True: "✅", False: "❌"}),
                                 Precision=s_df["Precision"].map("{:.0%}".format),
                                 Recall=s_df["Recall"].map("{:.0%}".format)),
                     hide_index=True, width="stretch")
        st.markdown(f"#### Robustness tests — {int(r_df['Passed'].sum())}/{len(r_df)} passed")
        st.dataframe(r_df.assign(Passed=r_df["Passed"].map({True: "✅", False: "❌"})), hide_index=True, width="stretch")
        st.caption("Detection = a feature is flagged if its KS/Chi² test is significant (p < 0.05) or PSI ≥ 0.10. "
                   "Limitations: synthetic data, per-feature tests (no multivariate drift), fixed thresholds.")
