# Activity 16 — Reflection Questions: Customer Groups (k-means)

**Diego Kamil Olea Ruiz** · Artificial Intelligence · Session 29

All numbers come from the app with its default settings unless the question says otherwise. The app
always gives the same result for the same settings, so every run below can be re-run and checked.

---

## Part A — The data and the algorithm

### 1. My first guess

By eye I see **4 groups** in *Meet the Customers*:

- People who almost never come and spend little each time (bottom-left corner, around 2 visits a month and a few hundred pesos).
- Regulars who come all the time but buy small baskets (far right, around 15 visits a month, low spend).
- Occasional shoppers who come a few times a month but spend a lot per visit (top-left, more than $1,500 per visit).
- Frequent big spenders who come often *and* spend a lot (upper middle, around 11 visits and more than $1,200 per visit).

### 2. The default run

With k = 4, Random customers, seed 4, maximum iterations 30 and tolerance 0:

- (a) It took **10 iterations**.
- (b) It was stopped by **convergence**: in iteration 10 no customer changed cluster, so the centroids stopped moving.
- (c) The final WCSS is **34.82**.

### 3. Assignment and update

The **assignment step** looks at every customer and paints it with the color of the closest centroid. The **update step** then moves each centroid ✕ to the average (middle) of the customers that now belong to it. Repeat until nothing changes.

In my iteration log the WCSS after each update went **91.04 → 89.88 → 89.08 → 87.64 → 84.70 → 71.30 → 39.04 → 34.86 → 34.82 → 34.82**. It only went down or stayed the same, never up.

It can never go up because both steps can only make the total smaller or leave it equal. In the assignment step a customer only switches if the new centroid is closer, so its squared distance can only shrink. In the update step the mean is exactly the point that gives the smallest sum of squared distances for that group, so moving the centroid there can't make the group worse. Two steps that never increase WCSS means the whole loop never increases it.

### 4. Stopping early

- (a) **Maximum iterations = 3**: stopped by the **maximum iterations** rule after **3 iterations**, final WCSS **88.51**.
- (b) **Maximum iterations = 30, tolerance = 0.1**: stopped by the **tolerance** rule after **2 iterations**, final WCSS **89.45**.

Both are about 2.5 times worse than the full run in Question 2 (34.82). The risk is that a small move does not mean the clusters are finished. In the full run's log the largest centroid move was only 0.079 in iteration 2 and 0.075 in iteration 3 (both under 0.1), but then it grew again to 0.243 in iteration 5 and 0.505 in iteration 6, and that is exactly when WCSS fell from 84.70 to 39.04. k-means was crawling before a big reorganization, so stopping early froze it with clusters that were still clearly wrong.

## Part B — Starting centroids

### 5. A bad seed

With Random customers and k = 4, **seed 8** gives a final WCSS of **387.35**, more than 11 times worse than 34.82 (it converged in that bad state).

What went wrong is where the centroids started: three of them landed in the same natural group, so they ended up splitting it into three small clusters (36, 28 and 32 customers), while the fourth centroid had to cover everyone else, **204 customers**, which is basically three real groups merged into one. Once that happened, no customer had a closer centroid to jump to, so k-means "converged" and stayed stuck there.

### 6. Restarts and the lecture's claim

In *Restarts* with k = 4 and 20 runs:

- (a) Best WCSS found: **34.82**
- (b) Random starts that reached it: **11 of 20**
- (c) k-means++ starts that reached it: **17 of 20**
- (d) Worst random-start WCSS: **387.35**

The statement is **not completely true**. Every one of these 40 runs converged, but 9 random runs and 3 k-means++ runs converged to a WCSS worse than 34.82, the worst one at 387.35. Converging only means the result is **stable** (no customer wants to change cluster), not that it is **optimal**. k-means finds a local minimum that depends on the starting centroids, which is why we run it many times and keep the lowest WCSS, and why k-means++ (smarter starts) hits the best answer more often than random starts.

### 7. My own bad start

With **Pick my own** I chose customers **#140, #267, #297 and #163**, all in the same bottom-left corner (between 0.5 and 1.3 visits a month and $82 to $272 per visit). k-means converged after 10 iterations with a final WCSS of **93.08**, almost 3 times worse than 34.82.

What it got wrong: the final clusters have 42, 82, 53 and 123 customers. Two centroids stayed in the low-spend area and split the rare cheap shoppers between them (centroids around 1.5 visits / $248 and 2.5 visits / $364), and the two big-spender groups (the occasional big spenders and the frequent big spenders) got merged into one cluster of 123 customers whose centroid sits in the empty space between them, at 6.9 visits and $1,489 per visit. The most valuable customers ended up mixed together.

## Part C — Choosing k and making business decisions

### 8. Elbow and silhouette

- **Elbow method: k = 4.** WCSS drops a lot from k = 1 to 4 (600.00 → 320.19 → 94.58 → 34.82) and then flattens out (28.41 for k = 5, 23.01 for k = 6), so the bend is at 4.
- **Silhouette score: k = 4.** It is the highest point of the curve, **0.746** (k = 3 gives 0.700 and k = 5 gives 0.689).

Both methods agree with each other and with my guess of 4 groups in Question 1.

### 9. Business segments (k = 4)

- **Cluster 1 · "Rare bargain visitors"**: 95 customers (32%), 2.1 visits a month, $313 per visit, $656 a month, **4%** of revenue.
- **Cluster 2 · "Everyday small-basket regulars"**: 81 customers (27%), 15.1 visits, $241 per visit, $3,630 a month, **19%** of revenue.
- **Cluster 3 · "Occasional big-ticket shoppers"**: 71 customers (24%), 3.9 visits, $1,653 per visit, $6,493 a month, **30%** of revenue.
- **Cluster 4 · "VIP loyal big spenders"**: 53 customers (18%), 11.1 visits, $1,258 per visit, $13,914 a month, **47%** of revenue.

(b) **Cluster 4** brings in the largest share: only **18% of the customers** produce **47% of the total monthly revenue**.

(c) One action per cluster:

- Rare bargain visitors: a cheap automated email with a time-limited coupon for their *next* purchase, to try to get a second visit a month without spending much on them.
- Everyday small-basket regulars: "add one more item" offers, like free shipping above a basket threshold or bundles, to raise the ticket of people who already come every day.
- Occasional big-ticket shoppers: reminders before paydays and seasons plus "meses sin intereses" on big purchases, to get them to come more often.
- VIP loyal big spenders: a loyalty tier with early access and priority service instead of discounts. They already buy a lot; the goal is not to lose them.

## Part D — Limits of k-means

### 10. Where k-means fails

- **Two moons**: k-means put **75.0%** of the points in their true group. WCSS of k-means' answer: **121.2**. WCSS of the true groups: **182.1**.
- **One big group, two small ones**: **81.1%** of the points in their true group. WCSS of k-means' answer: **1051.8**. WCSS of the true groups: **1345.6**.

In both datasets k-means' answer has a **lower WCSS than the true groups**. So k-means did not fail at its own job: it found a better answer to the question it is built to answer (make clusters as compact as possible). The problem is that the real groups are not round blobs of similar size (curved moons, or one big spread-out group next to two small dense ones), so the most compact split is not the real one. Minimizing WCSS and finding the true groups are not the same thing when the groups aren't round and equal.

## Part E — Another application

### 11. Photo compression

Coffee photo (600 × 400 pixels), k-means++, seed 0. The original takes **703 KB** uncompressed (24 bits per pixel).

- **k = 4**: **2 bits per pixel**, **59 KB** (12.0x smaller)
- **k = 16**: **4 bits per pixel**, **117 KB** (6.0x smaller)
- **k = 32**: **5 bits per pixel**, **147 KB** (4.8x smaller)

The smallest k I'd accept is **16**. With 4 colors the picture becomes a poster: smooth shading turns into flat bands. At 16 the coffee photo already looks close to the original at normal size, and going to 32 adds 30 KB for a difference that is hard to notice. 16 is where the compression is still big (6x) and the photo still looks like a photo.

### 12. Another real-world use

**Grouping homes by electricity use** (for example for CFE in Yucatán, where air conditioning dominates the bill).

- **Each data point**: one household with a smart meter.
- **Features**: average daily kWh, share of consumption during peak hours (6 to 10 pm), and the summer-to-winter ratio. I would standardize them like the app does, because kWh and percentages are on very different scales.
- **Choosing k**: run k = 2 to 10 with several k-means++ restarts, look at the elbow and the silhouette, and keep a k whose clusters can be explained in plain words (for example "AC all night", "small apartment", "home business during the day").
- **What I'd do with the clusters**: send each group a different energy-saving recommendation and test time-of-day tariffs on the "AC all night" group, which is the one that drives peak demand.

---

## Declaration of AI use

Following the course's academic integrity policy: I used Claude (Anthropic) as an assistant for this activity. It ran the activity's own `kmeans_engine.py` with the app's settings so I could check the exact numbers (WCSS, iteration log, restarts, segment report, failure datasets and photo sizes) and it helped me draft the wording of the answers. I reviewed the answers and the numbers before submitting.
