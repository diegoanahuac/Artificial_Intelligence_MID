# Activity 14 — Reflection Questions: ShopSmart Churn Prediction

**Student:** Diego Kamil Olea Ruiz
**Course:** Introduction to Artificial Intelligence — Universidad Anáhuac Mayab
**Session:** 24 · Regularized Logistic Regression (L1 & L2)

All numbers come from the **Canonical Model** sections of the app unless a question asks about the
playground.

---

## 1. Plain (unregularized) model

- Train accuracy: **100%** (20 / 20)
- Test accuracy: **77%** (76.67%, 23 / 30)
- Train loss: **0.022**
- Test loss: **2.887**

The model is perfect on the training customers but misses about 1 in 4 new customers. Its test
loss is roughly **130× its training loss**, which means it is very confident in many wrong
predictions. That is a clear case of overfitting.

![Plain model](screenshots/A14_01_overfitting_problem.png)

Without regularization the model hits 100% train accuracy but only 77% on test. Train loss is 0.022
and test loss is 2.887, so it memorized the training customers instead of learning the real
pattern.

## 2. Canonical L2 (Ridge) Model (λ = 0.05) — all 8 coefficients

Real features:
- Tenure (months as a customer): −1.33
- Average monthly spend ($): −0.15
- Support tickets filed: 0.78
- Days since last purchase: 0.48

Noise features:
- Favorite color (code): −0.20
- Customer's "lucky number": −0.62
- Day of week they signed up: −0.18
- Length of their referral code: −0.25

![Canonical L2 Model](screenshots/A14_04_canonical_l2.png)

This is the graded L2 model at λ = 0.05. Every coefficient is much smaller than in the plain model,
but all 8 are still there.

## 3. Canonical L2 Model — train and test accuracy

- Train accuracy: **95%** (5 points lower than the plain model's 100%)
- Test accuracy: **83%** (83.33%, 7 points higher than the plain model's 77%)
- Test loss: 0.519 (vs. 2.887 for the plain model)

L2 gives up a little training accuracy (it now misclassifies 1 of the 20 training customers) but
**generalizes better**: test accuracy rises from 77% to 83%, and test loss drops from 2.887 to 0.519.
The coefficients are much smaller overall (e.g., Support tickets 8.62 → 0.78), so the model is
far less overconfident.

## 4. Canonical L1 (Lasso) Model (λ = 0.05) — all 8 coefficients and zeroed features

Real features:
- Tenure (months as a customer): −2.11
- Average monthly spend ($): **0.00**
- Support tickets filed: 0.40
- Days since last purchase: 0.28

Noise features:
- Favorite color (code): **0.00**
- Customer's "lucky number": **0.00**
- Day of week they signed up: **0.00**
- Length of their referral code: **0.00**

**Features driven exactly to zero (5 / 8):** Average monthly spend, Favorite color, Customer's
"lucky number", Day of week they signed up, and Length of their referral code.

All 4 noise features were eliminated, but so was one real feature (Average monthly spend). Its
signal was already weak (−1.30 in the plain model, −0.15 under L2), so L1 judged it not worth
keeping at this λ.

![Canonical L1 Model](screenshots/A14_05_canonical_l1.png)

This is the graded L1 model at λ = 0.05. It zeroed out 5 of the 8 features, and only tenure,
support tickets and days since last purchase are left.

## 5. Canonical L1 Model — train and test accuracy

- Train accuracy: **95%**
- Test accuracy: **73%** (73.33%)

L1 produced the simplest model (only 3 active features), but at this λ its test accuracy is lower
than both the plain model (77%) and L2 (83%). Its test loss (0.639) is still far better than the
plain model's (2.887), so its probabilities are much better calibrated even though it gets a
couple more customers wrong.

## 6. L2 vs. L1 as λ grows

- **L2** shrinks **all** coefficients gradually and evenly. In the bar chart, every bar gets shorter,
  but none reaches exactly zero (even at λ = 0.5 and 1.0 all 8 features remained, with 0 zeroed).
- **L1** shrinks coefficients **and switches features off** completely. The zeroed-out counter
  climbed fast: 4 features at λ = 0.02 (all the noise features), 5 at λ = 0.05, 7 at λ = 0.1, and
  all 8 at λ = 0.5 (the model ends up predicting only from the bias).

L2 is like turning down the volume on every feature; L1 is like muting the ones it considers
unimportant.

![L2 playground at λ = 0.5](screenshots/A14_02_l2_large_lambda_0.5.png)

At λ = 0.5 all 8 bars shrank a lot but none of them reached zero. Train accuracy went down to 85%
and test stayed at 80%.

![L1 playground at λ = 0.05](screenshots/A14_03_l1_zeroed.png)

At λ = 0.05, L1 already zeroed out 5 of 8 features, all 4 noise ones plus average monthly spend.

![L1 playground at λ = 0.5](screenshots/A14_06_l1_lambda_0.5_all_zeroed.png)

Pushing L1 to λ = 0.5 zeroes out all 8 features, so the model only predicts from the bias. That is
where L1 goes too far.

## 7. True or False: "100% training accuracy means it's the best model to deploy"

**False.** From Question 1, the plain model has **100% train accuracy but only 77% test accuracy**,
and its **test loss (2.887) is about 130× its train loss (0.022)**. It memorized the 20 training
customers, including patterns in the useless columns, instead of learning what really drives churn.
The Canonical L2 model, with *lower* training accuracy (95%), does better on new customers (83% test
accuracy, test loss 0.519). Deployment is about performance on **new** customers, so the regularized
model is the better choice.

## 8. Real-world scenario

**Scenario:** a bank's **fraud-detection classifier** for credit-card transactions (fraud vs.
legitimate). Transaction records can include dozens or hundreds of fields: amount, merchant
category, time, location, device ID, browser version, terminal ID, form-filling speed, and so on.
Many are irrelevant or only correlate with fraud by coincidence, and confirmed fraud cases are
rare, so the model effectively learns from few positive examples. That is exactly when it can
memorize noise.

**First choice: L1 (Lasso).** It would automatically drop the irrelevant fields and leave a small
set of clear risk factors. That matters in banking, where analysts and regulators need to
understand *why* a transaction was flagged, and fewer features also means faster real-time scoring.
If several important features were highly correlated (e.g., transaction amount and amount in the
last 24 h), I'd switch to L2 or Elastic Net so the model doesn't arbitrarily drop one of them.

---

### Declaration of AI use

In accordance with Universidad Anáhuac's academic integrity declaration, I declare that I used an
AI assistant (Claude) to help organize the workflow, run the provided app locally, and draft and
format these answers. The numeric results come from the course's Canonical Models, verified
against my own screenshots, and I reviewed and edited every answer.
