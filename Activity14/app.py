"""ShopSmart: Predicting Customer Churn with Regularized Logistic Regression
— Activity 14 (Session 24).

Run locally with:
    streamlit run app.py
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from churn_data import ALL_FEATURES, FEATURE_LABELS, REAL_FEATURES, TEST_IDX, TRAIN_IDX
from logistic_engine import CANONICAL_L1, CANONICAL_L1_LAMBDA, CANONICAL_L2, CANONICAL_L2_LAMBDA, CANONICAL_PLAIN, fit

st.set_page_config(page_title="ShopSmart Churn Prediction", page_icon="📉", layout="wide")

st.title("📉 ShopSmart: Predicting Customer Churn")
st.markdown(
    "**ShopSmart** (from Activity 8) wants to predict which customers are about to **churn** — "
    "cancel their account or stop buying. Someone on the data team included 4 genuinely useful "
    "behavioral signals (tenure, spending, support tickets, days since last purchase) alongside "
    f"4 that are almost certainly useless (favorite color, a 'lucky number,' etc.), and there "
    f"are only **{len(TRAIN_IDX)} training customers** to learn from (the other {len(TEST_IDX)} "
    "are held back to test generalization). Same overfitting setup as Activity 13 — this time "
    "for a **classification** problem."
)


def coefficient_chart(weights, feature_names, title):
    colors = ["#4C8BF5" if f in REAL_FEATURES else "#E94F64" for f in feature_names]
    labels = [FEATURE_LABELS[f] for f in feature_names]
    fig = go.Figure(go.Bar(x=labels, y=weights, marker_color=colors))
    fig.update_layout(title=title, yaxis_title="Coefficient (normalized-feature scale)", height=420, margin=dict(t=50, b=100))
    return fig


tab_problem, tab_l2, tab_l1 = st.tabs(["📊 The Overfitting Problem", "🎯 L2 (Ridge)", "🎯 L1 (Lasso)"])

# --------------------------------------------------------------- Problem tab
with tab_problem:
    st.header("Meet the Unregularized Model")
    st.write("Blue bars are the 4 **real** behavioral features; red bars are the 4 **noise** features.")
    st.plotly_chart(coefficient_chart(CANONICAL_PLAIN["weights"], ALL_FEATURES, "Plain Logistic Regression — Coefficients"), use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Train accuracy", f"{CANONICAL_PLAIN['train_accuracy']:.0%}")
    col2.metric("Test accuracy", f"{CANONICAL_PLAIN['test_accuracy']:.0%}")
    col3.metric("Train loss", f"{CANONICAL_PLAIN['train_loss']:.3f}")
    col4.metric("Test loss", f"{CANONICAL_PLAIN['test_loss']:.3f}")
    st.warning(
        "**100% train accuracy** but noticeably lower test accuracy, and a test loss over "
        "**130x** the training loss — a textbook overfitting signature. With only 20 training "
        "customers and 8 features (4 of them noise), the model memorized the training set "
        "instead of learning the real churn pattern."
    )
    st.info(
        "The fix is the same idea as Session 23, applied to logistic regression's cross-entropy "
        "cost function instead of mean squared error: **L2 (Ridge)** shrinks every coefficient "
        "smoothly; **L1 (Lasso)** can drive coefficients to exactly zero."
    )

# --------------------------------------------------------------- L2 tab
with tab_l2:
    st.header("L2 (Ridge) Regularization")
    st.markdown("Adds `(λ/2) × Σw²` to the cost function — shrinks every coefficient smoothly toward zero.")

    st.subheader("🎛️ Playground")
    st.caption("Exploration only — does not affect your graded questions below.")
    lam_l2 = st.slider("λ (regularization strength)", 0.0, 1.0, 0.05, 0.01, key="l2_lambda")
    playground_l2 = fit(ALL_FEATURES, lam_l2, "l2")

    st.plotly_chart(coefficient_chart(playground_l2["weights"], ALL_FEATURES, f"L2 Coefficients at λ={lam_l2}"), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Train accuracy", f"{playground_l2['train_accuracy']:.0%}")
    c2.metric("Test accuracy", f"{playground_l2['test_accuracy']:.0%}")

    st.divider()
    st.subheader("📌 Canonical L2 Model (graded reference)")
    st.caption(f"Fixed setting: λ = {CANONICAL_L2_LAMBDA}")
    st.plotly_chart(coefficient_chart(CANONICAL_L2["weights"], ALL_FEATURES, f"Canonical L2 Coefficients (λ={CANONICAL_L2_LAMBDA})"), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Train accuracy", f"{CANONICAL_L2['train_accuracy']:.0%}", delta=f"{(CANONICAL_L2['train_accuracy']-CANONICAL_PLAIN['train_accuracy'])*100:.0f} pts vs. plain")
    c2.metric("Test accuracy", f"{CANONICAL_L2['test_accuracy']:.0%}", delta=f"{(CANONICAL_L2['test_accuracy']-CANONICAL_PLAIN['test_accuracy'])*100:.0f} pts vs. plain")

# --------------------------------------------------------------- L1 tab
with tab_l1:
    st.header("L1 (Lasso) Regularization")
    st.markdown("Adds `λ × Σ|w|` to the cost function — can drive coefficients to **exactly zero**: automatic feature selection.")

    st.subheader("🎛️ Playground")
    st.caption("Exploration only — does not affect your graded questions below.")
    lam_l1 = st.slider("λ (regularization strength)", 0.0, 0.5, 0.05, 0.01, key="l1_lambda")
    playground_l1 = fit(ALL_FEATURES, lam_l1, "l1")

    st.plotly_chart(coefficient_chart(playground_l1["weights"], ALL_FEATURES, f"L1 Coefficients at λ={lam_l1}"), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Train accuracy", f"{playground_l1['train_accuracy']:.0%}")
    c2.metric("Test accuracy", f"{playground_l1['test_accuracy']:.0%}")
    c3.metric("Features zeroed out", f"{playground_l1['num_zeroed']} / {len(ALL_FEATURES)}")

    st.divider()
    st.subheader("📌 Canonical L1 Model (graded reference)")
    st.caption(f"Fixed setting: λ = {CANONICAL_L1_LAMBDA}")
    st.plotly_chart(coefficient_chart(CANONICAL_L1["weights"], ALL_FEATURES, f"Canonical L1 Coefficients (λ={CANONICAL_L1_LAMBDA})"), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Train accuracy", f"{CANONICAL_L1['train_accuracy']:.0%}")
    c2.metric("Test accuracy", f"{CANONICAL_L1['test_accuracy']:.0%}")
    c3.metric("Features zeroed out", f"{CANONICAL_L1['num_zeroed']} / {len(ALL_FEATURES)}")
    zeroed_names = [FEATURE_LABELS[f] for f, w in zip(ALL_FEATURES, CANONICAL_L1["weights"]) if abs(w) < 1e-6]
    st.success(f"Zeroed-out features: {', '.join(zeroed_names)}")

st.divider()
st.info("When you're done exploring all three tabs, fill out **A14_ReflectionQuestions.md** with your results.")
