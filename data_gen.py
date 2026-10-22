"""Synthetic loan-applicant data, the four drift scenarios and the champion model.

Reference data (5,000 rows) trains the champion credit-risk model.
Each scenario produces a production batch (1,000 rows by default) with a
known, injected change, so every agent decision can be checked against
ground truth:

  1. Normal week   - nothing changes                     -> No action
  2. Gradual drift - incomes +~20%, city tier shifts      -> Alert
  3. Sudden drift  - gig-worker influx, lower credit scores
                     (and gig workers now default more)   -> Retrain
  4. Pipeline bug  - 40% of credit_score values missing    -> Fix pipeline
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from policy import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EMPLOYMENT_TYPES = ["salaried", "self-employed", "gig"]
CITY_TIERS = [1, 2, 3]


@dataclass
class ScenarioSpec:
    name: str
    expected: str
    description: str
    injected: list[str]                       # features with injected drift (ground truth)
    employment_p: tuple = (0.60, 0.30, 0.10)
    city_p: tuple = (0.45, 0.35, 0.20)
    income_mult: float = 1.0
    credit_shift: float = 0.0
    gig_credit_shift: float = 0.0
    missing_credit_rate: float = 0.0
    concept_shift: bool = False
    concept_gig_bias: float = 0.0         # extra log-odds of default for gig workers
    concept_gig_lti: float = 0.0          # extra loan-to-income sensitivity for gig workers
    concept_gig_credit: float = 0.0       # weakens credit-score signal for gig workers (thin files)
    extra: dict = field(default_factory=dict)


SCENARIOS: dict[str, ScenarioSpec] = {
    "Normal week": ScenarioSpec(
        name="Normal week",
        expected="No action",
        description="Nothing changed - production looks like the training data.",
        injected=[],
    ),
    "Gradual drift": ScenarioSpec(
        name="Gradual drift",
        expected="Alert",
        description="Applicant incomes up ~20% (inflation) and more tier-2/3 city applicants.",
        injected=["monthly_income", "loan_amount", "city_tier"],
        city_p=(0.36, 0.38, 0.26),
        income_mult=1.20,
    ),
    "Sudden drift": ScenarioSpec(
        name="Sudden drift",
        expected="Retrain",
        description="New partner channel brings a large influx of gig workers with thinner credit files.",
        injected=["employment_type", "credit_score", "monthly_income", "loan_amount"],
        employment_p=(0.35, 0.22, 0.43),
        credit_shift=-35.0,
        gig_credit_shift=-25.0,
        concept_shift=True,
        concept_gig_bias=0.8,
        concept_gig_lti=2.5,
        concept_gig_credit=0.010,
    ),
    "Pipeline bug": ScenarioSpec(
        name="Pipeline bug",
        expected="Fix pipeline",
        description="An upstream bureau API is failing - 40% of credit_score values arrive empty.",
        injected=["credit_score (missing values)"],
        missing_credit_rate=0.40,
    ),
}

# Backwards-compatible simple mapping: scenario -> expected decision
SCENARIO_EXPECTED = {k: v.expected for k, v in SCENARIOS.items()}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _simulate(n_rows: int, seed: int, spec: ScenarioSpec) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    employment = rng.choice(EMPLOYMENT_TYPES, size=n_rows, p=spec.employment_p)
    is_gig = employment == "gig"
    is_self = employment == "self-employed"

    age = np.clip(rng.normal(38, 10, n_rows), 21, 65).round().astype(int)

    median_income = np.where(is_gig, 30_000, np.where(is_self, 60_000, 55_000))
    income = median_income * np.exp(rng.normal(0, 0.45, n_rows)) * spec.income_mult
    income = (income / 100).round() * 100

    credit_mean = np.where(is_gig, 665, 712) + spec.credit_shift + np.where(is_gig, spec.gig_credit_shift, 0)
    credit = np.clip(rng.normal(credit_mean, 55), 300, 900).round()

    loan_to_income = np.exp(rng.normal(np.log(6), 0.40, n_rows))
    loan = (income * loan_to_income / 1000).round() * 1000

    city_tier = rng.choice(CITY_TIERS, size=n_rows, p=spec.city_p)

    # Ground-truth default process
    lti = np.log(loan / income / 6)
    z = (
        -2.0
        - 0.013 * (credit - 700)
        + 0.9 * lti
        + 0.55 * is_gig
        + 0.20 * is_self
        + 0.30 * (city_tier == 3)
        - 0.010 * (age - 38)
    )
    if spec.concept_shift:
        # New regime: gig workers from the new channel are much riskier,
        # and high loan-to-income hurts them far more than before.
        z = z + is_gig * (
            spec.concept_gig_bias + spec.concept_gig_lti * lti + spec.concept_gig_credit * (credit - 700)
        )
    default = (rng.random(n_rows) < _sigmoid(z)).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "monthly_income": income,
            "loan_amount": loan,
            "credit_score": credit.astype(float),
            "employment_type": employment,
            "city_tier": city_tier,
            TARGET: default,
        }
    )

    if spec.missing_credit_rate > 0:
        mask = rng.random(n_rows) < spec.missing_credit_rate
        df.loc[mask, "credit_score"] = np.nan

    return df


def generate_reference(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Reference (training) data: what the champion model was built on."""
    return _simulate(n_rows, seed, SCENARIOS["Normal week"])


def generate_production(scenario: str, n_rows: int = 1000, seed: int = 7) -> pd.DataFrame:
    """A production batch with the scenario's drift injected."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    return _simulate(n_rows, seed, SCENARIOS[scenario])


def generate_pool(scenario: str, pool_size: int = 3000, seed: int = 7) -> pd.DataFrame:
    """A larger pool of recent production data for the scenario.

    The monitored batch is the most recent slice of this pool; when the
    Critic asks for a larger window, the Profiler reads further back.
    """
    return generate_production(scenario, n_rows=pool_size, seed=seed)


def generate_holdout(scenario: str, n_rows: int = 2000, seed: int = 2024) -> pd.DataFrame:
    """Fresh, unseen data from the same scenario for champion vs challenger."""
    return generate_production(scenario, n_rows=n_rows, seed=seed)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(random_state: int = 0) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["employment_type"]),
            ("num", "passthrough", ["age", "monthly_income", "loan_amount", "credit_score", "city_tier"]),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=20, random_state=random_state, n_jobs=-1
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate(model: Pipeline, df: pd.DataFrame) -> dict:
    proba = model.predict_proba(df[FEATURES])[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(df[TARGET], proba)),
        "accuracy": float(accuracy_score(df[TARGET], pred)),
        "default_rate": float(df[TARGET].mean()),
    }


def train_champion(reference: pd.DataFrame, seed: int = 0):
    """Train the champion on 80% of reference data; keep 20% as a baseline test set.

    Returns (model, reference_test_df, baseline_metrics).
    """
    train, test = train_test_split(reference, test_size=0.2, random_state=seed, stratify=reference[TARGET])
    model = build_model(seed)
    model.fit(train[FEATURES], train[TARGET])
    return model, test.reset_index(drop=True), evaluate(model, test)
