"""Mérida Homes: Taming Overfitting with Regularization — Activity 13
(Session 23).

Run locally with:
    streamlit run app.py
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from house_data import ALL_FEATURES, FEATURE_LABELS, NOISE_FEATURES, REAL_FEATURES, TEST_IDX, TRAIN_IDX
from regularization_engine import CANONICAL_LASSO, CANONICAL_LASSO_LAMBDA, CANONICAL_OLS, CANONICAL_RIDGE, CANONICAL_RIDGE_LAMBDA, lasso_fit, ridge_fit

st.set_page_config(page_title="Mérida Homes: Regularization", page_icon="🧮", layout="wide")

st.title("🧮 Mérida Homes: Taming Overfitting with Regularization")
st.markdown(
    "This time, **Mérida Homes** gave you a messier spreadsheet: alongside the 4 features that "
    "genuinely predict price (size, bedrooms, age, distance), someone also included 4 columns "
    "that have **nothing to do with price** — a street number, the owner's 'lucky number,' a "
    "paint color code, and a zodiac score. Worse, you only have "
    f"**{len(TRAIN_IDX)} training houses** to learn from (the other {len(TEST_IDX)} are held back "
    "to test how well the model generalizes). That's exactly the setup where a model can "
    "**overfit** — and where **regularization** earns its keep."
)


def coefficient_chart(weights, feature_names, title):
    colors = ["#4C8BF5" if f in REAL_FEATURES else "#E94F64" for f in feature_names]
    labels = [FEATURE_LABELS[f] for f in feature_names]
    fig = go.Figure(go.Bar(x=labels, y=weights, marker_color=colors))
    fig.update_layout(title=title, yaxis_title="Coefficient (normalized-feature scale)", height=420, margin=dict(t=50, b=80))
    return fig


tab_problem, tab_ridge, tab_lasso = st.tabs(["📊 The Overfitting Problem", "🎯 Ridge (L2)", "🎯 Lasso (L1)"])

# --------------------------------------------------------------- Problem tab
with tab_problem:
    st.header("Meet the Unregularized Model")
    st.write(
        "Blue bars are the 4 **real** features; red bars are the 4 **noise** features that "
        "should, ideally, end up with a coefficient near zero."
    )
    st.plotly_chart(coefficient_chart(CANONICAL_OLS["weights"], ALL_FEATURES, "Plain Linear Regression — Coefficients"), use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Train R²", f"{CANONICAL_OLS['train_r2']:.4f}")
    col2.metric("Test R²", f"{CANONICAL_OLS['test_r2']:.4f}")
    gap = CANONICAL_OLS["train_r2"] - CANONICAL_OLS["test_r2"]
    st.warning(
        f"Train R² is **{gap:.3f} points higher** than Test R² — the model fits the training "
        "houses noticeably better than it predicts brand-new ones. That gap is the signature of "
        "**overfitting**: with only 20 training examples and 8 features (some of them pure "
        "noise), the model has room to fit noise instead of the real pattern."
    )
    st.info(
        "**Regularization** fixes this by adding a penalty for large coefficients to the cost "
        "function, discouraging the model from leaning on any one feature (especially a noisy "
        "one) too heavily. The next two tabs cover the two standard ways to do this: **Ridge "
        "(L2)** and **Lasso (L1)**."
    )

# --------------------------------------------------------------- Ridge tab
with tab_ridge:
    st.header("Ridge (L2) Regularization")
    st.markdown(
        "Ridge adds a penalty of `(λ/2) × Σw²` to the cost function. It **shrinks every "
        "coefficient toward zero**, smoothly, but essentially never makes one *exactly* zero."
    )

    st.subheader("🎛️ Playground")
    st.caption("Exploration only — does not affect your graded questions below.")
    lam_ridge = st.slider("λ (regularization strength)", 0.0, 2.0, 0.03, 0.01, key="ridge_lambda")
    playground_ridge = ridge_fit(ALL_FEATURES, lam_ridge if lam_ridge > 0 else 1e-8)

    st.plotly_chart(coefficient_chart(playground_ridge["weights"], ALL_FEATURES, f"Ridge Coefficients at λ={lam_ridge}"), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Train R²", f"{playground_ridge['train_r2']:.4f}")
    c2.metric("Test R²", f"{playground_ridge['test_r2']:.4f}")

    st.divider()
    st.subheader("📌 Canonical Ridge Model (graded reference)")
    st.caption(f"Fixed setting: λ = {CANONICAL_RIDGE_LAMBDA}")
    st.plotly_chart(coefficient_chart(CANONICAL_RIDGE["weights"], ALL_FEATURES, f"Canonical Ridge Coefficients (λ={CANONICAL_RIDGE_LAMBDA})"), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Train R²", f"{CANONICAL_RIDGE['train_r2']:.4f}", delta=f"{CANONICAL_RIDGE['train_r2'] - CANONICAL_OLS['train_r2']:.4f} vs. plain")
    c2.metric("Test R²", f"{CANONICAL_RIDGE['test_r2']:.4f}", delta=f"{CANONICAL_RIDGE['test_r2'] - CANONICAL_OLS['test_r2']:.4f} vs. plain")

# --------------------------------------------------------------- Lasso tab
with tab_lasso:
    st.header("Lasso (L1) Regularization")
    st.markdown(
        "Lasso adds a penalty of `λ × Σ|w|` to the cost function. Unlike Ridge, this can drive "
        "coefficients to **exactly zero** — Lasso doubles as automatic feature selection."
    )

    st.subheader("🎛️ Playground")
    st.caption("Exploration only — does not affect your graded questions below.")
    lam_lasso = st.slider("λ (regularization strength)", 0.0, 100.0, 30.0, 1.0, key="lasso_lambda")
    playground_lasso = lasso_fit(ALL_FEATURES, lam_lasso)

    st.plotly_chart(coefficient_chart(playground_lasso["weights"], ALL_FEATURES, f"Lasso Coefficients at λ={lam_lasso}"), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Train R²", f"{playground_lasso['train_r2']:.4f}")
    c2.metric("Test R²", f"{playground_lasso['test_r2']:.4f}")
    c3.metric("Features zeroed out", f"{playground_lasso['num_zeroed']} / {len(ALL_FEATURES)}")

    st.divider()
    st.subheader("📌 Canonical Lasso Model (graded reference)")
    st.caption(f"Fixed setting: λ = {CANONICAL_LASSO_LAMBDA}")
    st.plotly_chart(coefficient_chart(CANONICAL_LASSO["weights"], ALL_FEATURES, f"Canonical Lasso Coefficients (λ={CANONICAL_LASSO_LAMBDA})"), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Train R²", f"{CANONICAL_LASSO['train_r2']:.4f}")
    c2.metric("Test R²", f"{CANONICAL_LASSO['test_r2']:.4f}")
    c3.metric("Features zeroed out", f"{CANONICAL_LASSO['num_zeroed']} / {len(ALL_FEATURES)}")
    zeroed_names = [FEATURE_LABELS[f] for f, w in zip(ALL_FEATURES, CANONICAL_LASSO["weights"]) if abs(w) < 1e-6]
    st.success(f"Zeroed-out features: {', '.join(zeroed_names)}")

st.divider()
st.info("When you're done exploring all three tabs, fill out **A13_ReflectionQuestions.md** with your results.")
