# Activity 16: Customer Groups — Finding ShopSmart's Customer Segments with k-means
## Session 29
## Due date (mm/dd/yyyy): 11/01/2026
## Delivery Format: [] Video URL | [X] Markdown file | [] Jupyter Notebook file

---

# Activity Description

## The Story

**ShopSmart** (the online store from Activities 8 and 14) sends the same promotion to every
customer, and most of them ignore it. The marketing team suspects the store has several very
different kinds of customers, but nobody has ever labeled them. Your job: use **k-means
clustering** to discover the customer groups hidden in the data, decide how many segments the
store should use, and recommend what to do with each one.

Because nobody tells k-means what the groups are, this is **unsupervised learning**: the
algorithm has to find the structure on its own. You will see exactly how it does that, when it
works well, and when it doesn't.

**No programming background is required.** Everything happens by clicking, moving sliders, and
reading charts in the app. The customers are **fictional**, generated for this activity.

**App link:** https://uam-aiclass-a16.streamlit.app/

If you'd rather run it on your own machine instead of using the shared link, see
**Running It Yourself** below.

### The App

Six tabs:

1. **🛒 Meet the Customers** — 300 customers, each one a dot: visits per month vs. average spend
   per visit. No labels, no colors.
2. **👣 Step by Step** — watch k-means run one step at a time. You choose k, how the starting
   centroids are picked (random customers, k-means++, or **pick your own** by clicking customers),
   the random seed, and the two stopping rules: a **maximum number of iterations** and a
   **tolerance**.
3. **🔁 Restarts** — run k-means many times from different starting centroids and compare the
   results.
4. **📐 Choosing k & Segments** — the **elbow method** and the **silhouette score** for choosing k,
   plus a segment report (size, visits, spend, revenue) for the k you pick.
5. **⚠️ Where k-means Fails** — two datasets with known groups that k-means gets wrong.
6. **🖼️ Bonus: Photo Compression** — k-means picks the k colors that best represent a photo. You
   can try your own photo too.

> 💡 **Hint:** When the app first opens, wait until it finishes loading (the running indicator in
> the top-right corner disappears) before switching tabs.

### Your Tasks

1. **In Meet the Customers**, before running anything, decide how many groups you see by eye and
   write down a plain-words description of each one.

   > 💡 **Hint:** There is no wrong answer here. Hover over any dot to see that customer's numbers.
   > You will compare this guess with what k-means finds in Task 6.

2. **In Step by Step, leave the default settings** (k = 4, Random customers, seed 4). Click
   **Next step ▶** through at least the first two iterations (4 clicks), then **⏭ Run to the end**.
   Take a screenshot of the final chart, the stopping message, and the iteration log.

   > 💡 **Hint:** In each **assignment** step, watch which customers change color. In each
   > **update** step, watch where the ✕ centroids move: each one jumps to the middle (average) of
   > its own customers. The dotted lines are the centroids' trails.

3. **Test the two stopping rules.** Set **Maximum iterations** to 3 and run to the end. Then put it
   back to 30, set **Tolerance** to 0.1, and run to the end again. Take a screenshot of each.

   > 💡 **Hint:** Compare each final WCSS with the one from Task 2. In the full run's iteration log
   > from Task 2, look at the "largest centroid move" column: does a small move always mean the
   > clusters are finished?

4. **Starting centroids matter.** With Random customers, change the **seed** until you find a run
   whose final WCSS is clearly worse than in Task 2. Take a screenshot. Then choose **Pick my own**,
   click 4 starting customers that you think are a *bad* choice, and run to the end. Take a
   screenshot.

   > 💡 **Hint:** Try putting all your starting customers in the same corner of the chart. Clicking
   > a customer adds it as a starting centroid (a yellow star); **Clear my picks** starts over.

5. **In Restarts**, keep k = 4 and 20 runs. Take a screenshot of the four numbers and the dot chart.

   > 💡 **Hint:** Each dot is one *complete* run that converged. If every run converged, why don't
   > they all have the same WCSS?

6. **In Choosing k & Segments**, read the elbow and silhouette charts, then set **Your choice of k**
   to 4 and study the segment report. Take a screenshot of both charts and of the report.

   > 💡 **Hint:** The *elbow* is where the WCSS curve bends and then flattens. For the silhouette, look
   > for the highest point. In the report, compare each cluster's **share of customers** with its
   > **share of revenue**.

7. **In Where k-means Fails**, look at both datasets. Take a screenshot of each one, including the
   three numbers below the charts.

   > 💡 **Hint:** Compare "WCSS of k-means' answer" with "WCSS of the true groups". Which one is
   > lower, and what does that say about what k-means is trying to do?

8. **In Bonus: Photo Compression**, use the coffee photo with k = 4, 16, and 32. Optionally, upload
   a photo of your own. Take a screenshot at the smallest k you would accept.

   > 💡 **Hint:** The sizes below the images depend only on the image size and on k. Compare the
   > pictures, not just the numbers.

9. **Fill out `A16_ReflectionQuestions.md`**, using the exact numbers from your screenshots, and
   submit it along with your labeled screenshots (a `.zip` with the `.md` file and the images is
   fine).

   > 💡 **Hint:** Before submitting, check each answer against **How Your Grade Is Calculated**
   > below. It lists exactly what earns full points on every question.

### How Your Grade Is Calculated

Your grade is out of **100 points**, split across the 12 reflection questions. There are three
kinds of questions, and each kind is graded differently:

- **Exact answers** (Q2, Q4, Q8, and the numbers in Q6, Q9, Q10, Q11): there is one correct value,
  read from the app. With the settings the question gives, everyone sees exactly the same numbers.
  You get full points for correct values and partial credit when some parts are right. Small
  rounding differences are accepted.
- **Your own experiments** (Q5, Q7): every student's choices are different, so there is no single
  right number. But the app always gives exactly the same result for the same settings, so any run
  you report can be re-run and checked. You are graded on whether your numbers match your settings
  and on your description. Numbers that no setting can reproduce get no credit.
- **Explain in your own words** (Q1, Q3, Q12, and the explanations in Q6, Q9, Q10, Q11): graded on
  four levels.
  - **Excellent:** complete, correct, and specific to your own results.
  - **Proficient:** correct, but missing a required part or not tied to your own results.
  - **Developing:** vague, incomplete, or partly wrong.
  - **Insufficient:** missing or wrong.

| # | Question | Points | Full points if you… | Partial credit |
|---|---|---:|---|---|
| 1 | Your first guess | 4 | say how many groups you see **and** describe each one in plain words | 2–3: a number with little or no description |
| 2 | The default run | 8 | report the iterations, the stopping rule, and the final WCSS correctly | 3–6: 1 or 2 of the 3 correct |
| 3 | Assignment and update | 10 | explain both steps in your own words and why WCSS can never go up, using your iteration log | Proficient 6–8 · Developing 3–5 · Insufficient 0–2 |
| 4 | Stopping early | 10 | report the stopping rule, iterations, and final WCSS for both runs, and explain the risk of stopping early using the log | 4–7: values right but no explanation, or one run wrong |
| 5 | A bad seed | 8 | report a seed whose WCSS is worse, with the matching WCSS, a screenshot, and what went wrong | Proficient 5–7 · Developing 2–4 · Insufficient 0–1 |
| 6 | Restarts and the lecture's claim | 12 | report all four restart numbers correctly and use them to judge the lecture's statement about convergence | Proficient 8–10 · Developing 4–7 · Insufficient 0–3 |
| 7 | Your own bad start | 6 | report your 4 customer numbers and the final WCSS (reproducible), with a screenshot and what went wrong | Proficient 4–5 · Developing 2–3 · Insufficient 0–1 |
| 8 | Elbow and silhouette | 6 | give the k each method suggests and compare it with your guess from Q1 | 3–4: one method right, or no comparison |
| 9 | Business segments | 12 | name every cluster, identify the top-revenue cluster with its share of customers and of revenue, and propose one specific action per cluster | Proficient 8–10 · Developing 4–7 · Insufficient 0–3 |
| 10 | Where k-means fails | 10 | report the agreement % and both WCSS values for both datasets, and explain what the WCSS comparison means | 4–7: numbers right but weak explanation, or one dataset missing |
| 11 | Photo compression | 6 | report bits per pixel and size for k = 4, 16, and 32 (plus the original's size), and justify your smallest acceptable k | 3–4: sizes right but no justification, or one k wrong |
| 12 | Another application | 8 | describe a real use: what each data point is, which features, how you'd choose k, and what you'd do with the clusters | Proficient 5–6 · Developing 2–4 · Insufficient 0–1 |
| | **Total** | **100** | | |

**Delivery format:** your answers must be in a Markdown file (`A16_ReflectionQuestions.md`); a `.zip`
with that file and your screenshots is fine. Answers delivered in another format (PDF, Word, `.txt`)
lose **5 points** from the total.

### Running It Yourself (optional)

```bash
conda activate ai_uam
cd Activity16
pip install -r requirements.txt
streamlit run app.py
```

### About the Data and Photos

The 300 ShopSmart customers and the two "Where k-means Fails" datasets are fictional, generated
with a fixed random seed so that everyone sees exactly the same data. The photos come from the
scikit-image project's sample data:
- *coffee*: Rachel Michetti, courtesy of Pikolo Espresso Bar (CC0).
- *chelsea*: Stéfan van der Walt (CC0).
- *astronaut*: NASA, astronaut Eileen Collins (public domain).

# References:
- [Streamlit documentation](https://docs.streamlit.io/)
- [Arthur & Vassilvitskii (2007), "k-means++: The Advantages of Careful Seeding"](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)
- [Markdown Guide](https://www.markdownguide.org/basic-syntax/)
