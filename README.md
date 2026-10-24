# 🛰️ Model Drift Watchdog

**An Agentic AI system that detects, decides, validates and acts when ML models go stale.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-14B8A6)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-optional%20local%20LLM-000000)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Working%20PoC-22C55E)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://model-drift-watchdog.streamlit.app/)
![Tests](https://img.shields.io/badge/tests-25%20passing-22C55E)

### 🚀 Live demo: **[model-drift-watchdog.streamlit.app](https://model-drift-watchdog.streamlit.app/)**

> **CA-3: Domain-Specific Agentic AI Proof of Concept**
> Symbiosis Institute of Technology, Pune

| Team member | PRN |
|---|---|
| **Harsh Sharma** | 24070126067 |
| **Divyansh Agarwal** | 24070126056 |

**Domain:** MLOps (Machine Learning Operations), applied to a credit risk model at a digital lending company.

---

## 📌 Table of contents

1. [Problem statement](#1-problem-statement)
2. [Why this domain](#2-why-this-domain)
3. [Existing process vs proposed solution](#3-existing-process-vs-proposed-solution)
4. [Agents and responsibilities](#4-agents-and-responsibilities)
5. [Architecture](#5-architecture)
6. [Decision policy and Critic rules](#6-decision-policy-and-critic-rules)
7. [Demo scenarios (synthetic data)](#7-demo-scenarios-synthetic-data)
8. [Tech stack](#8-tech-stack)
9. [Project structure](#9-project-structure)
10. [Getting started](#10-getting-started)
11. [Validation and results](#11-validation-and-results)
12. [Limitations and future scope](#12-limitations-and-future-scope)
13. [Roadmap](#13-roadmap)
14. [Demo guide (5-minute presentation)](#14-demo-guide-5-minute-presentation)

---

## 1. Problem statement

Deployed machine learning models silently get worse when real-world data changes — a problem called **data drift**. Incomes rise with inflation, new kinds of customers (such as gig workers) start applying, or a broken data pipeline starts sending missing values.

The model keeps running, but it makes worse decisions. In lending, that means **risky loans get approved or good customers get rejected**, and nobody notices until losses appear months later.

Monitoring today is **manual, infrequent, and unable to tell real drift apart from a broken data pipeline**.

## 2. Why this domain

- 💸 **High financial stakes** — every wrong prediction is a bad loan or a lost customer.
- ⏳ **Label delay** — whether a loan defaults is known only months later, so accuracy can't be watched in real time. Monitoring the **input data** is the only early warning.
- ⚖️ **Regulatory risk** — lenders must be able to explain and audit automated credit decisions.
- 🏭 **Real industry pain** — monitoring and incident investigation take up a large share of ML teams' time.

## 3. Existing process vs proposed solution

| Aspect | ❌ Today (manual) | ✅ With Model Drift Watchdog |
|---|---|---|
| Frequency | Monthly or ad-hoc notebook checks | Every new data batch, automatically |
| Detection | After losses appear | Early, from input-data drift |
| Root cause | Engineers investigate by hand | Agents separate **real drift** from a **broken pipeline** |
| Decision | Gut feeling, inconsistent | Policy-based, **validated by a Critic agent** |
| Retraining | Manual and risky | Champion vs challenger comparison + **human approval** |
| Reporting | None or informal | Auto-generated report and alert |

**Impact:** faster detection · consistent decisions · fewer false alarms · full audit trail.

## 4. Agents and responsibilities

| Agent | Role type | Responsibility | Output |
|---|---|---|---|
| 🧭 **Planner** | Planner | Builds a task plan based on the data: picks tests per feature type, adds a performance check if labels exist, and **re-plans** if the Profiler finds bad data | Task plan |
| 🔎 **Data Profiler** | Research | Loads reference (training) and production data; checks schema, missing values, sample size and statistics | Data quality report |
| 📡 **Drift Detector** | Executor | Runs KS (numeric), Chi-square (categorical), PSI (all features), prediction drift, and accuracy if labels exist | Drift table |
| 🔀 **Decision Agent** | Business Automation | Applies the policy: **No action / Alert / Retrain / Fix pipeline** | Recommended action |
| 🛡️ **Critic** | Critic | Validates evidence and decision; triggers targeted retries (max 2) or escalates to a human | Approve / retry / escalate |
| 📝 **Reporter / Action** | Executor + Automation | Writes the report and alert; on Retrain, trains a challenger model and compares it with the champion | Report, alert, model comparison |
| 👤 **Human Approver** | Human-in-the-loop | Approves or rejects deploying the new model | Final sign-off |

All agents read and write **one shared state** (data, plan, results, decision, critic verdict, retry count). Orchestration is a **LangGraph state graph** whose conditional edges handle re-planning, Critic retries and escalation.

> **Design principle:** decisions come from statistics and rules, so they are **auditable and repeatable**. The LLM only *explains* them in plain English — it never makes the retrain decision itself.

## 5. Architecture

```mermaid
flowchart TD
    UI["🖥️ Streamlit UI<br/>pick scenario · run · view trace · approve"]
    P["🧭 Planner Agent<br/>labels present? feature types? → task plan"]
    DP["🔎 Data Profiler<br/>schema · missing values · sample size"]
    RP["🧭 Planner re-plans<br/>skip drift tests"]
    DD["📡 Drift Detector<br/>KS · Chi² · PSI · prediction drift · accuracy"]
    DA["🔀 Decision Agent<br/>No action / Alert / Retrain / Fix pipeline"]
    C{"🛡️ Critic Agent<br/>evidence + policy checks"}
    ESC["⚠️ Escalate to human"]
    R["📝 Reporter / Action Agent<br/>report · alert · champion vs challenger"]
    H["👤 Human approval"]
    END["✅ Deploy new model / close incident"]

    UI -->|"request + new data batch"| P
    P --> DP
    DP -->|"missing > 10% or schema error"| RP
    RP -->|"Fix pipeline"| DA
    DP -->|"data OK"| DD
    DD --> DA
    DA --> C
    C -.->|"retry ≤ 2"| DP
    C -.->|"retry ≤ 2"| DD
    C -.->|"retry ≤ 2"| DA
    C -->|"still failing"| ESC
    C -->|"approved"| R
    R --> H
    H --> END
```

**Data:** `reference` training data · production batch (per scenario) · champion model
**Tools:** pandas · NumPy · SciPy (KS, Chi²) · custom PSI · scikit-learn · Plotly · LangGraph · Ollama / mock LLM

More detail: [`docs/architecture.md`](docs/architecture.md)

## 6. Decision policy and Critic rules

**Drift thresholds**

| Check | Threshold |
|---|---|
| PSI per feature | `< 0.10` stable · `0.10 – 0.25` moderate · `> 0.25` major |
| KS / Chi-square | `p < 0.05` → drifted |
| Data-quality issue | missing rate `> 10%` on any feature, or schema mismatch |
| Critical features | `credit_score`, `monthly_income`, `loan_amount`, `employment_type` |

**Decision rules**

| Condition | Decision |
|---|---|
| Data-quality issue | 🛠️ **Fix pipeline** (never retrain on bad data) |
| Major drift on a critical feature, **or** prediction PSI `> 0.25`, **or** accuracy drop `> 5` points | 🔁 **Retrain** |
| Moderate drift anywhere, **or** major drift on a non-critical feature | ⚠️ **Alert** |
| Otherwise | ✅ **No action** |

**Critic retry rules**

| Problem found | What happens |
|---|---|
| Sample size `< 500` | Profiler uses a larger data window |
| KS and PSI disagree on a critical feature | Detector re-checks with a bootstrap test |
| Decision breaks the policy | Decision agent revises it |
| Still failing after 2 retries | Escalate to a human |

**Deployment rule:** the challenger model is accepted only if its AUC is at least `0.01` higher than the champion's on fresh data **and** a human approves.

## 7. Demo scenarios (synthetic data)

Synthetic loan-applicant data — **5,000** training rows plus a **1,000**-row production batch per scenario.
Features: `age`, `monthly_income`, `loan_amount`, `credit_score`, `employment_type` (salaried / self-employed / gig), `city_tier` (1/2/3). Target: `default` (0/1).

| # | Scenario | What's injected | Expected outcome |
|---|---|---|---|
| 1 | Normal week | Nothing | ✅ No action |
| 2 | Gradual drift | Incomes up ~20% (loan amounts follow), city tier shifts slightly | ⚠️ Alert |
| 3 | Sudden drift | Large influx of gig workers, lower credit scores — and gig workers now default more (concept change) | 🔁 Retrain → challenger wins → human approves |
| 4 | Pipeline bug | 40% of `credit_score` values missing | 🛠️ Fix pipeline (**not** retrain) |

Scenario 4 is the key test: a naive system would retrain on broken data — our agents don't.

## 8. Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| UI | Streamlit |
| Orchestration | LangGraph (with a built-in fallback runner) |
| Statistics | SciPy (KS, Chi-square), custom PSI |
| ML | scikit-learn (Random Forest champion / challenger) |
| Data | pandas, NumPy (synthetic generator) |
| Charts | Plotly |
| LLM (optional) | Ollama local model for explanations, template-based mock fallback |

No paid APIs are used.

## 9. Project structure

```
model-drift-watchdog/
├── app.py                  # Streamlit UI (entry point)
├── data_gen.py             # Synthetic loan data, 4 drift scenarios, champion model
├── policy.py               # Locked thresholds and decision rules
├── state.py                # Shared state passed between agents
├── graph.py                # LangGraph orchestration (+ identical fallback runner)
├── runner.py               # Builds the inputs for one run and executes it
├── llm.py                  # Ollama client + mock LLM fallback
├── validation.py           # Validation suite (also runs from the UI)
├── agents/
│   ├── planner.py          # Planner Agent — plan + re-plan
│   ├── profiler.py         # Data Profiler — quality checks, wider window on request
│   ├── detector.py         # Drift Detector — KS, Chi², PSI, prediction drift, bootstrap
│   ├── decision.py         # Decision Agent — applies the policy
│   ├── critic.py           # Critic Agent — 5 checks, retries, escalation
│   ├── reporter.py         # Reporter / Action — report, alert, incident, retrain
│   └── common.py           # Trace logging helpers
├── tests/                  # 25 pytest tests (all scenarios + Critic loop)
├── docs/                   # Problem statement, architecture, validation, demo guide
├── presentation/           # PPT for the CA-3 presentation
├── .streamlit/config.toml  # Dark theme
└── requirements.txt
```

## 10. Getting started

```bash
# 1. Clone
git clone https://github.com/HarshSharma-28/model-drift-watchdog.git
cd model-drift-watchdog

# 2. Create a virtual environment (Python 3.10+)
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

**Windows shortcut:** double-click `run_app.bat`. It finds Python, creates `.venv`, installs everything on the first run, and opens the app.

The app opens at http://localhost:8501. The first load takes a few seconds while the champion model is trained.

**Run the tests and the validation suite**

```bash
python -m pytest -q        # 25 tests
python validation.py       # prints the results tables below
```

**Optional — local LLM explanations with Ollama (free)**

```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3.2:3b
```

If Ollama isn't running, the app automatically uses the mock LLM, so the demo works on any laptop.
Use `OLLAMA_MODEL` / `OLLAMA_URL` environment variables to pick a different model or server.

### Using the app

| Sidebar control | What it does |
|---|---|
| Production batch scenario | Picks one of the 4 synthetic scenarios |
| True loan outcomes available | Adds the accuracy/AUC check (off = label delay, input drift only) |
| Use local LLM (Ollama) | Plain-English explanation from Ollama; falls back to the mock |
| Demo controls → Batch size | Below 500 rows the Critic asks for a wider window; 100 rows escalates |
| Demo controls → Inject a faulty first decision | The Decision agent's first call is naive — watch the Critic catch it |
| Demo controls → Orchestrator | LangGraph or the plain-Python fallback (identical results) |

| Tab | Shows |
|---|---|
| 🧭 Agent Trace | The path taken through the graph (retries highlighted), the task plan and every agent step |
| 📡 Drift Analysis | PSI per feature, prediction drift, test table with p-values, distributions, performance |
| 🛡️ Decision & Critic | Decision, rule fired, evidence, policy table, every Critic round with pass/fail |
| 📝 Report & Action | Plain-English summary, report download, alert, incident, champion vs challenger, **Approve / Reject**, model registry |
| ✅ Validation | Runs the full validation suite against ground truth |

### Screenshots

| Agent trace (Sudden drift) | Drift analysis |
|---|---|
| ![Agent trace](docs/screenshots/1_agent_trace.png) | ![Drift analysis](docs/screenshots/2_drift_analysis.png) |
| **Decision & Critic (Pipeline bug + injected fault)** | **Report & Action (champion vs challenger)** |
| ![Decision and Critic](docs/screenshots/3_decision_critic.png) | ![Report and action](docs/screenshots/4_report_action.png) |

## 11. Validation and results

Every scenario has a known, injected change, so the agents' output is checked against ground truth
(`python validation.py`, or the **Validation** tab).

**Decision accuracy — 8 / 8** (4 scenarios × labels on/off)

| Scenario | Expected | Agent decision | Critic retries | Detection precision / recall | Retrain AUC gain |
|---|---|---|---|---|---|
| Normal week | No action | ✅ No action | 0 | 100% / 100% | — |
| Gradual drift | Alert | ✅ Alert | 1 (bootstrap re-check of `loan_amount`) | 100% / 100% | — |
| Sudden drift | Retrain | ✅ Retrain | 0 | 100% / 100% | **+0.026** (0.742 → 0.768) |
| Pipeline bug | Fix pipeline | ✅ Fix pipeline | 0 | 100% / 100% | — (not retrained) |

Identical decisions with labels switched off: the system works under label delay.

**Robustness tests — 7 / 7 passed**

| Test | Result |
|---|---|
| Faulty first decision (×4 scenarios) | Critic caught it every time and sent it back; final decision correct |
| Small batch (300 rows) | Critic asked for more data → window widened to 600 rows → No action |
| Tiny batch (100 rows) | Window reached 400 rows after 2 retries → **escalated to a human**, nothing acted on |
| LangGraph vs fallback runner | Same decision and same path for every scenario |

Key observations:
- **Pipeline bug is never retrained.** The Planner re-plans, drift tests are skipped, and the Critic's "never retrain on corrupted data" check blocks a naive retrain.
- **Sudden drift:** accuracy fell from 83.5% to 71.6%; the retrained challenger beat the champion on 2,000 fresh rows, and is deployed only after a human clicks Approve.
- **Gradual drift:** `loan_amount` was statistically significant but its PSI (0.085) was below 0.10. The Critic flagged the disagreement and the Detector settled it with a 300-sample bootstrap.
- A full monitoring cycle takes about 1 second.

Details: [`docs/validation_plan.md`](docs/validation_plan.md)

## 12. Limitations and future scope

**Limitations**
- Synthetic data only
- Per-feature tests can miss combined (multivariate) drift
- Fixed thresholds, not learned
- Label delay not simulated
- Mock LLM explanations are template-based

**Future scope**
- Scheduled or real-time streaming monitoring
- Multivariate drift detection (classifier-based)
- MLflow model registry integration
- Slack / email alerts via MCP
- Canary rollout of new models and adaptive thresholds

## 13. Roadmap

- [x] Problem statement, domain and architecture finalised
- [x] Repository and project structure set up
- [x] Synthetic data generator with 4 scenarios
- [x] Agents: Planner, Profiler, Detector, Decision, Critic, Reporter
- [x] LangGraph orchestration with re-planning, retries and escalation
- [x] Streamlit UI: agent trace, drift analysis, decision & critic, report & action, validation
- [x] Ollama integration with mock fallback
- [x] Validation run across all scenarios (8/8 decisions, 7/7 robustness tests)
- [ ] Final presentation and demo

## 14. Demo guide (5-minute presentation)

See [`docs/demo_guide.md`](docs/demo_guide.md) for a click-by-click script. In short:
1. **Normal week** → ✅ No action (show the clean path in Agent Trace)
2. **Sudden drift** → 🔁 Retrain → Report & Action → champion vs challenger → **Approve**
3. **Pipeline bug** + *Inject a faulty first decision* → the Critic blocks a retrain on broken data → 🛠️ Fix pipeline
4. **Validation** tab → 8/8 and 7/7

---

## 📄 License

[MIT](LICENSE) © 2026 Harsh Sharma, Divyansh Agarwal
