# Demo Guide — 5-minute presentation

Start the app before you present: `streamlit run app.py` (first load trains the champion, ~3 s).
Keep the sidebar open. Leave "True loan outcomes available" **on**.

| Time | Who | What to do | What to say |
|---|---|---|---|
| 0:00–1:00 | Harsh | Slides: problem | Models fail silently; label delay means we must watch input data. |
| 1:00–2:00 | Harsh | Slides: architecture | 6 agents + human, shared state in LangGraph, Critic loop. |
| 2:00–2:30 | Divyansh | Scenario **Normal week** → Run Watchdog → **Agent Trace** tab | Straight path, every test stable → ✅ No action. Nothing to do is also a decision. |
| 2:30–3:30 | Divyansh | **Sudden drift** → Run → **Drift Analysis** (PSI chart) → **Report & Action** → **Approve & deploy** | credit_score and employment_type PSI > 0.25, accuracy −12 pts → Retrain. Challenger beats champion on fresh data (+0.026 AUC). A human approves; registry shows v2. |
| 3:30–4:15 | Divyansh | **Pipeline bug**, open *Demo controls* → tick **Inject a faulty first decision** → Run → **Agent Trace** + **Decision & Critic** | Planner re-plans (drift tests skipped). The Decision agent naively says Retrain; the Critic fails "never retrain on corrupted data" and sends it back → 🛠️ Fix pipeline + incident ticket. |
| 4:15–5:00 | Harsh | **Validation** tab → Run validation suite; then slides: limitations & future | 8/8 decisions, 100% detection, 7/7 robustness tests. Limitations and future scope. |

## Backup demos (if asked)

- **📚 Knowledge Base tab** (after any run) → shows the runbook the agents followed (e.g. RB-03 Retrain), every document
  each agent cited and why, a search box, and the agent memory of past runs.

- **Gradual drift** → the Critic spots that `loan_amount` is statistically significant but has a small PSI, and
  sends it back for a bootstrap re-check (orange dashed edge in the graph) → ⚠️ Alert.
- **Batch size 100** (Demo controls) with Sudden drift → the window widens twice, still under 500 rows →
  🚨 escalated to a human; no automatic retrain.
- Turn **True loan outcomes** off → same decisions from input drift only (label delay).
- **Orchestrator: Fallback runner** → same results without LangGraph.

## Likely viva questions

**Why not let the LLM decide?** Lending is regulated; decisions must be auditable and repeatable. Statistics and a fixed
policy decide, the Critic checks, the LLM only explains.

**What is PSI?** Population Stability Index — compares the share of data in each bin between training and production:
Σ (actual − expected) · ln(actual / expected). < 0.10 stable, 0.10–0.25 moderate, > 0.25 major.

**Why both KS and PSI?** KS tells you whether a change is *statistically significant* (with 1,000 rows even tiny changes
are), PSI tells you whether it is *big enough to matter*. When they disagree on a critical feature, the Critic asks for a
bootstrap re-check.

**Why is the Planner an agent and not a fixed script?** It reads the data first: picks KS vs Chi-square per feature type,
adds the performance check only if labels exist, and re-plans when the data is broken.

**What stops the retry loop running forever?** Max 2 retries, then escalation to a human.

**Why not retrain on the pipeline-bug data?** The data is corrupted, not genuinely different; retraining would teach the
model from broken inputs. The correct action is to fix the feed.

**How do you know the retrained model is better?** It is evaluated on 2,000 fresh rows it never saw, must beat the
champion's AUC by ≥ 0.01, and still needs human approval.

**What is your knowledge base?** 21 documents (drift concepts, decision policy, runbooks, lending domain) indexed with
TF-IDF. Agents retrieve and cite them: the Planner cites the tests it chose, the Decision agent and Critic cite the
policy, and the Reporter retrieves the runbook for the final action and turns it into the recommended steps. Past runs
are stored as agent memory. Retrieval guides and explains actions but never changes a decision.
