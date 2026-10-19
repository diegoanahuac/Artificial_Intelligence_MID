# Activity 14: ShopSmart — Predicting Customer Churn (Regularized Logistic Regression)
## Sessions 24
## Due date (mm/dd/yyyy): 10/18/2026
## Delivery Format: [] Video URL | [X] Markdown file | [] Jupyter Notebook file

---

# Activity Description

## The Story

**ShopSmart** (the fictional online store from Activity 8) wants to predict which customers are
about to **churn** — cancel their account or simply stop buying. The data team gave you 4
genuinely useful behavioral signals (how long they've been a customer, how much they spend,
how many support tickets they've filed, how long since their last purchase) plus 4 that are
almost certainly useless (favorite color, a "lucky number," signup day of week, referral code
length) — and only 20 training customers to learn from.

This is the same overfitting setup as Activity 13, now applied to **classification** with
**Regularized Logistic Regression**.

**App link:** https://uam-aiclass-a14.streamlit.app/

If you'd rather run it on your own machine instead of using the shared link, see
**Running It Yourself** below.

### The App

Three tabs:

1. **📊 The Overfitting Problem** — the unregularized model achieves **100% training accuracy**
   but clearly worse test accuracy, plus a train-vs-test loss comparison.
2. **🎯 L2 (Ridge)** — a live λ slider showing smooth coefficient shrinkage, plus a fixed
   Canonical L2 Model.
3. **🎯 L1 (Lasso)** — a live λ slider showing exact zeroing of coefficients, plus a fixed
   Canonical L1 Model and a live "features zeroed out" counter.

### Your Tasks

1. **In the Overfitting Problem tab**, note the training accuracy vs. test accuracy, and the
   training loss vs. test loss. Take a screenshot.

2. **In the L2 tab, try at least 3 different λ values.** Take a screenshot at a large λ (0.5+).

3. **In the L1 tab, try at least 3 different λ values**, and find one where at least 3 features
   get zeroed out. Take a screenshot.

4. **Record the Canonical L2 Model's coefficients, train accuracy, and test accuracy.** Take a
   screenshot.

5. **Record the Canonical L1 Model's coefficients, train/test accuracy, and which features got
   zeroed out.** Take a screenshot.

6. **Fill out `A14_ReflectionQuestions.md`**, using the exact numbers from your Canonical Model
   screenshots, and submit it along with your labeled screenshots.

### Running It Yourself (optional)

```bash
conda activate ai_uam
cd Activity14
pip install -r requirements.txt
streamlit run app.py
```

# References:
- [Streamlit documentation](https://docs.streamlit.io/)
- [Markdown Guide](https://www.markdownguide.org/basic-syntax/)
