# Activity 13 — Reflection Questions: Regularization at Mérida Homes

**Student:** Diego Kamil Olea Ruiz
**Course:** Introduction to Artificial Intelligence — Universidad Anáhuac Mayab
**Session:** 23 · Regularized Linear Regression (Ridge & Lasso)

All numbers come from the **Canonical Model** sections of the app unless a question asks about the
playground. Screenshots are in [`screenshots/`](screenshots/).

---

## 1. Plain (unregularized) model — train R², test R² and gap

| Metric | Value |
|---|---|
| Train R² | **0.9871** |
| Test R² | **0.8851** |
| Gap (train − test) | **0.1021** (the app shows it rounded as 0.102) |

The model explains almost all of the variance in the 20 training houses but noticeably less in the
30 unseen test houses. That gap is the signature of overfitting: with only 20 examples and 8
features, the model also fits noise coming from the 4 irrelevant columns.

*Screenshot: `screenshots/A13_01_overfitting_problem.png`*

## 2. Canonical Ridge Model (λ = 0.03) — all 8 coefficients

| Feature | Type | Coefficient |
|---|---|---|
| Size (m²) | real | 540.35 |
| Bedrooms | real | 45.00 |
| Age (years) | real | −88.91 |
| Distance to downtown (km) | real | −142.48 |
| Street number | noise | −31.65 |
| Owner's "lucky number" | noise | −2.80 |
| Front door paint color code | noise | 34.37 |
| Owner's zodiac sign score | noise | 44.73 |

*Screenshot: `screenshots/A13_04_canonical_ridge.png`*

## 3. Canonical Ridge Model — train R² and test R²

| Metric | Ridge (λ = 0.03) | Plain model | Difference |
|---|---|---|---|
| Train R² | **0.9861** | 0.9871 | −0.0010 |
| Test R² | **0.8926** | 0.8851 | **+0.0075** |

The test R² is **better** than the plain model's. Ridge gave up a tiny bit of training fit
(−0.0010) in exchange for better generalization on the unseen houses (+0.0075), and the
train-test gap shrank from 0.1021 to 0.0935.

## 4. Canonical Lasso Model (λ = 30) — all 8 coefficients and zeroed features

| Feature | Type | Coefficient |
|---|---|---|
| Size (m²) | real | 543.01 |
| Bedrooms | real | 14.28 |
| Age (years) | real | −65.27 |
| Distance to downtown (km) | real | −129.93 |
| Street number | noise | −20.25 |
| Owner's "lucky number" | noise | **0.00** |
| Front door paint color code | noise | **0.00** |
| Owner's zodiac sign score | noise | 38.04 |

**Features driven exactly to zero (2 / 8):** Owner's "lucky number" and Front door paint color code.
All 4 real features stayed nonzero.

*Screenshot: `screenshots/A13_05_canonical_lasso.png`*

## 5. Canonical Lasso Model — train R² and test R²

| Metric | Lasso (λ = 30) |
|---|---|
| Train R² | **0.9777** |
| Test R² | **0.8760** |

In this case Lasso's test R² is slightly **lower** than the plain model's (−0.0091). Removing two
noise features made the model simpler and more interpretable, but λ = 30 also shrank real
features such as *Bedrooms* (45.81 → 14.28), so part of the useful signal was lost too.

## 6. Ridge vs. Lasso as λ grows

- **Ridge (L2)** shrinks **every** coefficient **smoothly and proportionally**. As λ grows, the bars
  get shorter, but they never land exactly on zero. Even at λ = 2.0, all 8 bars (including the red
  noise ones) were still nonzero.
- **Lasso (L1)** pushes coefficients toward zero and **eliminates** them: the weakest features hit
  **exactly 0** and disappear from the model. As λ grew, the zeroed-out counter went 0 → 1 (λ = 10)
  → 2 (λ = 30) → 4 (λ = 80) → 5 (λ = 100).

So Ridge *turns everything down*, while Lasso *selects features* by switching some of them off
completely.

## 7. True or False: "increasing λ always improves test R²"

**False.** In my playground exploration:

| Model | λ | Test R² |
|---|---|---|
| Ridge | 0.00 | 0.8851 |
| Ridge | 0.03 | 0.8926 ← best |
| Ridge | 0.10 | 0.8819 |
| Ridge | 0.50 | 0.7707 |
| Ridge | 1.00 | 0.6635 |
| Ridge | 2.00 | 0.5175 |
| Lasso | 10 | 0.8911 |
| Lasso | 30 | 0.8760 |
| Lasso | 80 | 0.8527 |

A small λ helped (Ridge λ = 0.03 → 0.8926), but past that point test R² went **down**. For example,
at **Ridge λ = 1.0, test R² fell to 0.6635**, far worse than the plain model's 0.8851. Too much
regularization **underfits**: it also crushes the real features (Size went from 566.8 to 262.7),
so the model can no longer capture the true pattern. The goal is to find the λ that balances
bias and variance, not to make λ as large as possible.

*Screenshots: `screenshots/A13_02_ridge_large_lambda.png`, `screenshots/A13_03_lasso_zeroed.png`*

## 8. Real-world scenario

**Scenario:** an IT team predicting **server or cloud costs** (or incident volume) from monitoring
logs. A single dashboard can export hundreds of metrics: CPU, memory and traffic, but also
log-file sizes, host IDs, timestamps encoded as numbers, and team tags. Many of them are
irrelevant or redundant, and there may only be a few months of history, which is the same
"many features, few examples" setup as Mérida Homes.

**First choice: Lasso (L1).** Because I'd suspect many columns are pure noise, I'd want the model
to **drop them automatically**. That makes the model easier to explain to the team ("cost depends
on these 5 metrics") and cheaper to maintain, since fewer metrics need to be collected. If the
remaining metrics turned out to be strongly correlated with each other (e.g., CPU and load
average), I'd then consider Ridge or Elastic Net, because Lasso tends to keep one correlated
feature arbitrarily and drop the others.

---

### Declaration of AI use

In accordance with Universidad Anáhuac's academic integrity declaration, I declare that I used an
AI assistant (Claude) to help organize the workflow, run the provided app locally, and draft and
format these answers. The numeric results come from the course's Canonical Models, verified
against my own screenshots, and I reviewed and edited every answer.
