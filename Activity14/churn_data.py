"""The ShopSmart customer-churn dataset (Session 24: Regularized Logistic
Regression).

Same fictional company as Activity 8 (ShopSmart), now trying to predict
whether a customer will churn (cancel/stop buying) using 4 real behavioral
features and 4 irrelevant ones — plus a deliberately small training set, to
invite the same kind of overfitting Activity 13 explored for regression.
There is no randomness a student can trigger; everything is generated once,
with a fixed seed, at import time.
"""

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


REAL_FEATURES = ["tenure_months", "monthly_spend", "support_tickets", "days_since_last"]
NOISE_FEATURES = ["favorite_color_code", "lucky_number", "signup_weekday", "referral_code_len"]
ALL_FEATURES = REAL_FEATURES + NOISE_FEATURES

FEATURE_LABELS = {
    "tenure_months": "Tenure (months as a customer)",
    "monthly_spend": "Average monthly spend ($)",
    "support_tickets": "Support tickets filed",
    "days_since_last": "Days since last purchase",
    "favorite_color_code": "Favorite color (code)",
    "lucky_number": "Customer's 'lucky number'",
    "signup_weekday": "Day of week they signed up",
    "referral_code_len": "Length of their referral code",
}


def _generate(seed: int = 123, n: int = 50):
    rng = np.random.default_rng(seed)

    tenure_months = rng.uniform(1, 60, n)
    monthly_spend = rng.uniform(10, 200, n)
    support_tickets = rng.integers(0, 10, n).astype(float)
    days_since_last = rng.uniform(0, 90, n)

    favorite_color_code = rng.uniform(1, 5, n)
    lucky_number = rng.uniform(1, 100, n)
    signup_weekday = rng.integers(0, 7, n).astype(float)
    referral_code_len = rng.integers(4, 12, n).astype(float)

    z_true = (
        -0.05 * tenure_months
        - 0.02 * monthly_spend
        + 0.35 * support_tickets
        + 0.03 * days_since_last
        - 1.0
    )
    prob_true = sigmoid(z_true)
    churned = (rng.uniform(0, 1, n) < prob_true).astype(float)

    data = {
        "tenure_months": tenure_months,
        "monthly_spend": monthly_spend,
        "support_tickets": support_tickets,
        "days_since_last": days_since_last,
        "favorite_color_code": favorite_color_code,
        "lucky_number": lucky_number,
        "signup_weekday": signup_weekday,
        "referral_code_len": referral_code_len,
        "churned": churned,
    }

    perm = rng.permutation(n)
    train_idx, test_idx = perm[:20], perm[20:]
    return data, train_idx, test_idx


DATA, TRAIN_IDX, TEST_IDX = _generate()
CHURNED = DATA["churned"]


def feature_matrix(feature_names: list[str]) -> np.ndarray:
    return np.column_stack([DATA[name] for name in feature_names])


def train_test_split(feature_names: list[str]):
    X = feature_matrix(feature_names)
    return X[TRAIN_IDX], CHURNED[TRAIN_IDX], X[TEST_IDX], CHURNED[TEST_IDX]
