# Artificial Intelligence — Activities (Midterm II period)

**Student:** Diego Kamil Olea Ruiz · Universidad Anáhuac Mayab
**Course:** Introduction to Artificial Intelligence (Prof. Miguel A. Guirao Aguilera)
**Course repository (instructions & starter code):** https://github.com/UAMayab/ai

Each activity folder contains:

- `README.md`: the professor's original instructions
- the starter app code (`app.py` + data/engine modules) so it can be run locally
- `AXX_ReflectionQuestions.md`: **graded deliverable** with my answers
- `screenshots/`: labeled screenshots of my run of the app

| Activity | Session | Topic | Deliverable |
|---|---|---|---|
| [Activity 13](Activity13/) | 23 | Mérida Homes: Regularized Linear Regression (Ridge / Lasso) | [A13_ReflectionQuestions.md](Activity13/A13_ReflectionQuestions.md) |
| [Activity 14](Activity14/) | 24 | ShopSmart: Regularized Logistic Regression for churn (L2 / L1) | [A14_ReflectionQuestions.md](Activity14/A14_ReflectionQuestions.md) |
| [Activity 15](Activity15/) | 25–27 | Digit Lab: neural networks, back-propagation and handwriting recognition | [A15_ReflectionQuestions.md](Activity15/A15_ReflectionQuestions.md) |

## Running an activity locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r Activity13/requirements.txt
streamlit run Activity13/app.py
```

The starter code belongs to the course repository [UAMayab/ai](https://github.com/UAMayab/ai) and is
distributed under the GNU GPL v3 license.
