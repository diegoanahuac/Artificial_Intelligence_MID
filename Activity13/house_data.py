"""The Mérida Homes 'noisy' dataset (Session 23: Regularized Linear Regression).

Same fictional agency as Activity 12, but this dataset deliberately includes
four IRRELEVANT features alongside the four real ones, and a small training
set relative to the number of features — exactly the setup that invites
overfitting, so regularization has something real to fix. There is no
randomness a student can trigger; everything is generated once, with a fixed
seed, at import time.
"""

import numpy as np

REAL_FEATURES = ["size_m2", "bedrooms", "age_years", "distance_km"]
NOISE_FEATURES = ["street_number", "lucky_number", "paint_color_code", "zodiac_score"]
ALL_FEATURES = REAL_FEATURES + NOISE_FEATURES

FEATURE_LABELS = {
    "size_m2": "Size (m²)",
    "bedrooms": "Bedrooms",
    "age_years": "Age (years)",
    "distance_km": "Distance to downtown (km)",
    "street_number": "Street number",
    "lucky_number": "Owner's 'lucky number'",
    "paint_color_code": "Front door paint color code",
    "zodiac_score": "Owner's zodiac sign score",
}
TARGET_LABEL = "Price (thousands of MXN)"


def _generate(seed: int = 99, n: int = 50):
    rng = np.random.default_rng(seed)

    size_m2 = rng.uniform(60, 300, n)
    bedrooms = rng.integers(1, 7, n).astype(float)
    age_years = rng.uniform(0, 40, n)
    distance_km = rng.uniform(0.5, 20, n)

    # Pure noise: numbers with NO real relationship to price whatsoever.
    street_number = rng.uniform(1, 200, n)
    lucky_number = rng.uniform(1, 100, n)
    paint_color_code = rng.uniform(1, 5, n)
    zodiac_score = rng.uniform(1, 12, n)

    true_price = 8.0 * size_m2 + 60.0 * bedrooms - 9.0 * age_years - 18.0 * distance_km + 300
    noise = rng.normal(0, 120, n)
    price = np.clip(true_price + noise, 300, None)

    data = {
        "size_m2": size_m2,
        "bedrooms": bedrooms,
        "age_years": age_years,
        "distance_km": distance_km,
        "street_number": street_number,
        "lucky_number": lucky_number,
        "paint_color_code": paint_color_code,
        "zodiac_score": zodiac_score,
        "price": price,
    }

    # Fixed train/test split — deliberately small training set (20 of 50)
    # relative to 8 features, to invite real overfitting.
    perm = rng.permutation(n)
    train_idx, test_idx = perm[:20], perm[20:]
    return data, train_idx, test_idx


DATA, TRAIN_IDX, TEST_IDX = _generate()
PRICE = DATA["price"]


def feature_matrix(feature_names: list[str]) -> np.ndarray:
    return np.column_stack([DATA[name] for name in feature_names])


def train_test_split(feature_names: list[str]):
    X = feature_matrix(feature_names)
    return X[TRAIN_IDX], PRICE[TRAIN_IDX], X[TEST_IDX], PRICE[TEST_IDX]
