# Activity 13: Mérida Homes — Taming Overfitting with Regularization
## Sessions 23
## Due date (mm/dd/yyyy): 10/18/2026
## Delivery Format: [] Video URL | [X] Markdown file | [] Jupyter Notebook file

---

# Activity Description

## The Story

**Mérida Homes** gave you a messier spreadsheet this time. Alongside the 4 features that
genuinely predict price (size, bedrooms, age, distance), someone also included 4 columns that
have **nothing to do with price**: a street number, the owner's "lucky number," a paint color
code, and a zodiac score. Worse, you only have 20 training houses to learn from (30 more are
held back to test how well the model generalizes).

That combination — a small training set and several irrelevant features — is exactly what
causes a linear regression model to **overfit**: fitting noise in the training data instead of
the real underlying pattern. **Regularization** is how you fix that.

This activity covers Session 23 in one app. Just like Activity 12, you can freely explore in a
**Playground** — but every graded question is based on a separate, fixed **Canonical Model**.

**App link:** https://uam-aiclass-a13.streamlit.app/

If you'd rather run it on your own machine instead of using the shared link, see
**Running It Yourself** below.

### The App

Three tabs:

1. **📊 The Overfitting Problem** — the unregularized model's coefficients (color-coded: blue
   for the 4 real features, red for the 4 noise features) and its train-vs-test R² gap.
2. **🎯 Ridge (L2)** — a live λ slider showing how Ridge shrinks every coefficient smoothly,
   plus a fixed Canonical Ridge Model.
3. **🎯 Lasso (L1)** — a live λ slider showing how Lasso can drive coefficients to *exactly*
   zero, plus a fixed Canonical Lasso Model and a live "features zeroed out" counter.

### Your Tasks

1. **In the Overfitting Problem tab**, note the train R² vs. test R² gap for the plain
   (unregularized) model. Take a screenshot.

2. **In the Ridge tab, try at least 3 different λ values in the playground.** Watch how the red
   (noise) bars behave differently from the blue (real) bars as λ grows. Take a screenshot at a
   large λ (e.g. 1.0+).

3. **In the Lasso tab, try at least 3 different λ values**, and find one where at least 2
   features get zeroed out while all 4 real features are still nonzero. Take a screenshot.

4. **Record the Canonical Ridge Model's coefficients, train R², and test R².** Take a screenshot.

5. **Record the Canonical Lasso Model's coefficients, train R², test R², and which features got
   zeroed out.** Take a screenshot.

6. **Fill out `A13_ReflectionQuestions.md`**, using the exact numbers from your Canonical Model
   screenshots, and submit it along with your labeled screenshots.

### Running It Yourself (optional)

```bash
conda activate ai_uam
cd Activity13
pip install -r requirements.txt
streamlit run app.py
```

# References:
- [Streamlit documentation](https://docs.streamlit.io/)
- [Markdown Guide](https://www.markdownguide.org/basic-syntax/)
