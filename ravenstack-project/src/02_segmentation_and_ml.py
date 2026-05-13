"""
=============================================================
RavenStack SaaS — Step 2: Segmentation + ML + Retention System
=============================================================
Purpose:
- Load master table from Step 1
- Create customer segments
- Engineer stronger features
- Train and compare churn prediction models
- Pick best model by AUC
- Generate churn probability
- Create retention priority system
- Save actionable outputs

How to run:
    python src/02_segmentation_and_ml.py
=============================================================
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 150, "font.family": "DejaVu Sans"})


# ============================================================
# 1. PATH SETUP
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MASTER_PATH = OUTPUT_DIR / "master_table.csv"
if not MASTER_PATH.exists():
    raise FileNotFoundError(
        f"Missing {MASTER_PATH}. Run src/01_eda_and_kpis.py first."
    )


# ============================================================
# 2. LOAD MASTER TABLE
# ============================================================
print("Loading master table...")
master = pd.read_csv(MASTER_PATH)
print(f"Dataset shape: {master.shape}")


# ============================================================
# 3. BASIC CLEANUP + FEATURE ENGINEERING
# ============================================================
print("\nCreating features...")

# Boolean columns to integers if present
bool_cols = [
    "is_trial",
    "sub_is_trial",
    "auto_renew_flag",
    "upgrade_flag",
    "downgrade_flag",
]
for col in bool_cols:
    if col in master.columns:
        master[col] = master[col].astype(int)

# Numeric columns
numeric_candidates = [
    "tenure_days",
    "seats",
    "mrr_amount",
    "arr_amount",
    "ticket_count",
    "avg_resolution_hrs",
    "avg_first_response_mins",
    "avg_satisfaction",
    "escalation_count",
    "urgent_ticket_count",
    "total_usage_events",
    "total_usage_count",
    "avg_duration_secs",
    "total_errors",
    "beta_feature_uses",
    "unique_features",
]
for col in numeric_candidates:
    if col in master.columns:
        master[col] = pd.to_numeric(master[col], errors="coerce")

# Derived business features
if "tenure_days" in master.columns:
    master["tenure_months"] = master["tenure_days"] / 30.0
else:
    master["tenure_months"] = np.nan

if "ticket_count" in master.columns:
    master["tickets_per_month"] = (
        master["ticket_count"] / master["tenure_months"].replace(0, np.nan)
    )

if "total_usage_events" in master.columns:
    master["usage_events_per_month"] = (
        master["total_usage_events"] / master["tenure_months"].replace(0, np.nan)
    )

if "total_usage_count" in master.columns:
    master["avg_usage_per_event"] = (
        master["total_usage_count"] / master["total_usage_events"].replace(0, np.nan)
    )

if "total_errors" in master.columns:
    master["errors_per_100_usage"] = (
        100 * master["total_errors"] / master["total_usage_count"].replace(0, np.nan)
    )

if "unique_features" in master.columns:
    master["usage_per_feature"] = (
        master["total_usage_count"] / master["unique_features"].replace(0, np.nan)
    )

if "seats" in master.columns:
    master["revenue_per_seat"] = (
        master["mrr_amount"] / master["seats"].replace(0, np.nan)
    )

if "ticket_count" in master.columns and "escalation_count" in master.columns:
    master["escalation_rate"] = (
        master["escalation_count"] / master["ticket_count"].replace(0, np.nan)
    )

master.replace([np.inf, -np.inf], np.nan, inplace=True)


# ============================================================
# 4. CUSTOMER SEGMENTATION
# ============================================================
print("\nCreating customer segments...")

if "mrr_amount" in master.columns:
    master["value_segment"] = pd.qcut(
        master["mrr_amount"].fillna(0).rank(method="first"),
        q=3,
        labels=["Low Value", "Mid Value", "High Value"],
    )
else:
    master["value_segment"] = "Unknown"

if "tenure_days" in master.columns:
    master["tenure_segment"] = pd.cut(
        master["tenure_days"],
        bins=[-1, 180, 540, 999999],
        labels=["New (<6m)", "Mid (6-18m)", "Loyal (>18m)"],
    )
else:
    master["tenure_segment"] = "Unknown"

if "total_usage_events" in master.columns:
    master["engagement_segment"] = pd.cut(
        master["total_usage_events"].fillna(0),
        bins=[-1, 5, 20, 999999],
        labels=["Low Engagement", "Mid Engagement", "High Engagement"],
    )
else:
    master["engagement_segment"] = "Unknown"

print("\nSegment distribution:")
print(master["value_segment"].value_counts(dropna=False).to_string())


# ============================================================
# 5. MACHINE LEARNING — CHURN PREDICTION
# ============================================================
print("\nTraining churn prediction models...")

if "churn_flag" not in master.columns:
    raise ValueError("Target column 'churn_flag' not found in master table.")

y = master["churn_flag"].astype(int)

candidate_numeric = [
    "tenure_days",
    "seats",
    "mrr_amount",
    "arr_amount",
    "ticket_count",
    "avg_resolution_hrs",
    "avg_first_response_mins",
    "avg_satisfaction",
    "escalation_count",
    "urgent_ticket_count",
    "total_usage_events",
    "total_usage_count",
    "avg_duration_secs",
    "total_errors",
    "beta_feature_uses",
    "unique_features",
    "tenure_months",
    "tickets_per_month",
    "usage_events_per_month",
    "avg_usage_per_event",
    "errors_per_100_usage",
    "usage_per_feature",
    "revenue_per_seat",
    "escalation_rate",
    "is_trial",
    "sub_is_trial",
    "auto_renew_flag",
    "upgrade_flag",
    "downgrade_flag",
]

candidate_categorical = [
    "industry",
    "country",
    "referral_source",
    "sub_plan_tier",
    "billing_frequency",
]

numeric_features = [c for c in candidate_numeric if c in master.columns]
categorical_features = [c for c in candidate_categorical if c in master.columns]

feature_cols = numeric_features + categorical_features
X = master[feature_cols].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# OneHotEncoder compatibility across sklearn versions
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            numeric_features,
        ),
        (
            "cat",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", ohe),
            ]),
            categorical_features,
        ),
    ],
    remainder="drop",
)

models = {
    "Logistic Regression": Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="liblinear",
            )),
        ]
    ),
    "Random Forest": Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=3,
                random_state=42,
                class_weight="balanced_subsample",
            )),
        ]
    ),
    "Extra Trees": Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", ExtraTreesClassifier(
                n_estimators=600,
                max_depth=10,
                min_samples_leaf=3,
                random_state=42,
                class_weight="balanced",
            )),
        ]
    ),
}

results = {}

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Better threshold than 0.50 for imbalanced churn
    threshold = 0.35
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "pipeline": pipeline,
        "auc": auc,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }

    print(f"\n[{name}] AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

best_model_name = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_model_name]["pipeline"]
best_auc = results[best_model_name]["auc"]

print("\nBest model:", best_model_name)
print(f"Best validation AUC: {best_auc:.3f}")


# ============================================================
# 6. FIT BEST MODEL ON FULL DATA + SCORE ALL CUSTOMERS
# ============================================================
print("\nScoring all customers...")

best_model.fit(X, y)
master["churn_probability"] = best_model.predict_proba(X)[:, 1]


# ============================================================
# 7. RETENTION PRIORITY SYSTEM
# ============================================================
print("\nCreating retention priority system...")

def assign_priority(row):
    value = row.get("value_segment", "Unknown")
    prob = row.get("churn_probability", 0)

    if value == "High Value" and prob >= 0.60:
        return "P1 — High Value, High Risk"
    elif value == "Mid Value" and prob >= 0.50:
        return "P2 — Mid Value, High Risk"
    elif prob >= 0.40:
        return "P3 — Low Value, High Risk"
    else:
        return "P4 — Low Risk"

master["retention_priority"] = master.apply(assign_priority, axis=1)

print("\nPriority distribution:")
print(master["retention_priority"].value_counts().to_string())


# ============================================================
# 8. BUSINESS METRICS
# ============================================================
revenue_at_risk_pred = master.loc[master["churn_probability"] >= 0.40, "mrr_amount"].sum()
p1_revenue = master.loc[
    master["retention_priority"] == "P1 — High Value, High Risk",
    "mrr_amount"
].sum()

print("\nBusiness impact:")
print(f"Predicted Revenue at Risk: ${revenue_at_risk_pred:,.0f}")
print(f"P1 Revenue at Risk: ${p1_revenue:,.0f}")

high_value_count = master.loc[
    master["retention_priority"] == "P1 — High Value, High Risk"
].shape[0]

print(f"High-value customers at risk: {high_value_count}")
# ============================================================
# 9. RETENTION ACTION PLAN
# ============================================================
actions = {
    "P1 — High Value, High Risk": "Immediate call + custom discount",
    "P2 — Mid Value, High Risk": "Targeted email + support follow-up",
    "P3 — Low Value, High Risk": "Automated nudges / offers",
    "P4 — Low Risk": "Upsell / loyalty program",
}

top_customers = (
    master.sort_values("churn_probability", ascending=False)
    .head(20)
    .loc[:, [
        "account_name",
        "industry",
        "mrr_amount",
        "churn_probability",
        "retention_priority",
    ]]
    .copy()
)

top_customers["recommended_action"] = top_customers["retention_priority"].map(actions)

master.to_csv(OUTPUT_DIR / "master_with_predictions.csv", index=False)
top_customers.to_csv(OUTPUT_DIR / "retention_actions.csv", index=False)

print("\nTop 10 Customers for Retention:")
print(top_customers.head(10).to_string(index=False))

print("\nSaved:")
print("- outputs/master_with_predictions.csv")
print("- outputs/retention_actions.csv")


# ============================================================
# 10. VISUALS
# ============================================================
print("\nSaving charts...")

priority_order = [
    "P1 — High Value, High Risk",
    "P2 — Mid Value, High Risk",
    "P3 — Low Value, High Risk",
    "P4 — Low Risk",
]

priority_counts = master["retention_priority"].value_counts().reindex(priority_order, fill_value=0)

plt.figure(figsize=(9, 4))
bars = plt.bar(
    priority_counts.index,
    priority_counts.values,
    color=["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71"]
)
plt.title("Retention Priority Distribution", fontsize=13, fontweight="bold")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20, ha="right")
plt.bar_label(bars, padding=3, fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "11_retention_priority_distribution.png", bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 4))
priority_prob = (
    master.groupby("retention_priority")["churn_probability"]
    .mean()
    .reindex(priority_order)
)
bars = plt.bar(priority_prob.index, priority_prob.values, color="#3498DB")
plt.title("Average Churn Probability by Retention Priority", fontsize=13, fontweight="bold")
plt.ylabel("Average Churn Probability")
plt.xticks(rotation=20, ha="right")
plt.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "12_churn_probability_by_priority.png", bbox_inches="tight")
plt.close()

plt.figure(figsize=(9, 5))
revenue_by_priority = (
    master.groupby("retention_priority")["mrr_amount"]
    .sum()
    .reindex(priority_order)
)
bars = plt.bar(revenue_by_priority.index, revenue_by_priority.values, color="#8E44AD")
plt.title("Revenue by Retention Priority", fontsize=13, fontweight="bold")
plt.ylabel("Total MRR (USD)")
plt.xticks(rotation=20, ha="right")
plt.bar_label(bars, fmt="$%.0f", padding=3, fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "13_revenue_by_priority.png", bbox_inches="tight")
plt.close()

if best_model_name in ["Random Forest", "Extra Trees"]:
    try:
        feature_names = best_model.named_steps["preprocess"].get_feature_names_out()
        importances = best_model.named_steps["model"].feature_importances_
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)

        plt.figure(figsize=(9, 6))
        feat_imp.sort_values().plot(kind="barh")
        plt.title(f"Top 15 Feature Importances — {best_model_name}", fontsize=13, fontweight="bold")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "14_feature_importance.png", bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Could not save feature importance chart: {e}")

try:
    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_test,
        results[best_model_name]["y_prob"],
        name=f"{best_model_name} (AUC={best_auc:.3f})",
    )
    plt.title("ROC Curve — Churn Prediction", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "15_roc_curve.png", bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Could not save ROC curve: {e}")

print("Charts saved to outputs/")
print("Done.")