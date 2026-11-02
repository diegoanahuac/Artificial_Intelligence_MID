"""Customer Groups: Finding ShopSmart's Customer Segments with k-means
— Activity 16 (Session 29).

Run locally with:
    streamlit run app.py
"""

import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import kmeans_engine as ke
from customer_data import CUSTOMER_IDS, FEATURE_LABELS, X
from failure_data import DATASETS

st.set_page_config(page_title="Customer Groups: k-means", page_icon="🛒", layout="wide")

COLORS = ["#4C8BF5", "#E94F64", "#2BB673", "#F5A623", "#9B59B6", "#17BECF", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22"]
INIT_LABELS = {"Random customers": "random", "k-means++": "k-means++", "Pick my own": "manual"}
STOP_TEXT = {
    "converged": "**Converged**: no customer changed cluster in the last iteration, so the centroids stopped moving.",
    "tolerance": "**Stopped by the tolerance rule**: the largest centroid move was no more than the tolerance. "
                 "The clusters may not be finished.",
    "max iterations": "**Stopped at the maximum number of iterations** before converging. The clusters may not be finished.",
}
IMAGES_DIR = Path(__file__).with_name("images")
PHOTOS = {
    "Coffee (a product photo)": ("coffee.png", "Photo: Rachel Michetti, courtesy of Pikolo Espresso Bar (CC0)."),
    "Chelsea the cat": ("chelsea.png", "Photo: Stéfan van der Walt (CC0)."),
    "Astronaut Eileen Collins": ("astronaut.png", "Photo: NASA (public domain)."),
}
UPLOAD = "Upload your own photo"

Z, Z_MEAN, Z_STD = ke.standardize(X)


def to_original(C):
    """Centroids from standard units back to visits and pesos (exact: the mean commutes with rescaling)."""
    return C * Z_STD + Z_MEAN


st.title("🛒 Customer Groups: Finding ShopSmart's Customer Segments")
st.markdown(
    "**ShopSmart** (the online store from Activities 8 and 14) wants to stop sending the same promotion "
    f"to everyone. The data team has **{len(X)} customers** (fictional) and two numbers for each one: how "
    "often they buy and how much they spend per visit. Nobody has labeled the customers, so there is no "
    "\"right answer\" to learn from: this is **unsupervised learning**. The **k-means** algorithm has to "
    "discover the groups on its own."
)


def customer_chart(labels=None, centroids=None, trails=None, picks=(), title="", height=520):
    fig = go.Figure()
    hover = "Customer #%{customdata[0]}<br>%{x} visits/month<br>$%{y:,.0f} MXN per visit<extra></extra>"
    if labels is None:
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", name="Customers", customdata=CUSTOMER_IDS[:, None],
                                 marker=dict(color="#9AA5B1", size=7, opacity=0.8), hovertemplate=hover))
    else:
        for j in range(int(labels.max()) + 1 if centroids is None else len(centroids)):
            m = labels == j
            fig.add_trace(go.Scatter(x=X[m, 0], y=X[m, 1], mode="markers", name=f"Cluster {j + 1} ({m.sum()})",
                                     customdata=CUSTOMER_IDS[m][:, None], hovertemplate=hover,
                                     marker=dict(color=COLORS[j % len(COLORS)], size=7, opacity=0.75)))
    if trails is not None:
        for j in range(trails.shape[1]):
            fig.add_trace(go.Scatter(x=trails[:, j, 0], y=trails[:, j, 1], mode="lines", showlegend=False, hoverinfo="skip",
                                     line=dict(color=COLORS[j % len(COLORS)], dash="dot", width=2)))
    if centroids is not None:
        fig.add_trace(go.Scatter(
            x=centroids[:, 0], y=centroids[:, 1], mode="markers+text", name="Centroids",
            text=[str(j + 1) for j in range(len(centroids))], textposition="top right", textfont=dict(size=14),
            marker=dict(symbol="x", size=18, color=[COLORS[j % len(COLORS)] for j in range(len(centroids))],
                        line=dict(width=2, color="black")),
            hovertemplate="Centroid %{text}<br>%{x:.1f} visits/month<br>$%{y:,.0f} MXN per visit<extra></extra>"))
    if picks:
        P = X[[p - 1 for p in picks]]
        fig.add_trace(go.Scatter(x=P[:, 0], y=P[:, 1], mode="markers+text", name="Your picks",
                                 text=[str(i + 1) for i in range(len(picks))], textposition="top right",
                                 marker=dict(symbol="star", size=18, color="#FFD400", line=dict(width=1, color="black")),
                                 hoverinfo="skip"))
    fig.update_layout(title=title, xaxis_title=FEATURE_LABELS[0], yaxis_title=FEATURE_LABELS[1], height=height,
                      margin=dict(t=50), legend=dict(orientation="h", y=-0.15))
    return fig


@st.cache_data(max_entries=256, show_spinner=False)
def cached_run(k, init, seed, max_iter, tol, manual_idx):
    return ke.run(Z, k, init, seed, max_iter, tol, list(manual_idx) if manual_idx else None)


@st.cache_data(show_spinner=False)
def cached_restarts(k, n_runs):
    return ke.restarts(Z, k, "random", n_runs), ke.restarts(Z, k, "k-means++", n_runs)


@st.cache_data(show_spinner=False)
def cached_elbow():
    return ke.elbow_table(Z)


@st.cache_data(show_spinner=False)
def cached_best(k):
    return ke.best_of(Z, k)


@st.cache_data(show_spinner=False)
def cached_failure(name):
    Xf, yf = DATASETS[name]
    k = len(np.unique(yf))
    best = ke.best_of(Xf, k)
    true_centroids = np.array([Xf[yf == g].mean(axis=0) for g in range(k)])
    return best, ke.wcss(Xf, yf, true_centroids), ke.match_accuracy(best["labels"], yf)


tab_meet, tab_steps, tab_restarts, tab_k, tab_fail, tab_photo = st.tabs([
    "🛒 Meet the Customers", "👣 Step by Step", "🔁 Restarts", "📐 Choosing k & Segments",
    "⚠️ Where k-means Fails", "🖼️ Bonus: Photo Compression",
])

# ------------------------------------------------------------ Meet the Customers
with tab_meet:
    st.header("Every dot is one customer")
    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", f"{len(X)}")
    col2.metric("Average visits per month", f"{X[:, 0].mean():.1f}")
    col3.metric("Average spend per visit", f"${X[:, 1].mean():,.0f} MXN")
    st.plotly_chart(customer_chart(title="ShopSmart customers: no labels, no colors (yet)"), width="stretch", key="meet_chart")
    st.info(
        "**Why the app rescales the data first.** k-means groups customers by *distance*. A difference of "
        "1 visit and a difference of 1 peso are not comparable, and pesos run into the thousands, so without "
        "rescaling the spend would decide everything. Before clustering, each feature is converted to "
        f"**standard units** (how many standard deviations above or below average): 1 standard unit = "
        f"{Z_STD[0]:.2f} visits per month, or ${Z_STD[1]:,.0f} MXN per visit. The charts always show the "
        "real units."
    )

# ------------------------------------------------------------ Step by Step
with tab_steps:
    st.header("Watch k-means work, one step at a time")
    st.markdown(
        "Each **iteration** has two steps: **1. Assignment**: every customer joins the cluster of its "
        "nearest centroid (✕). **2. Update**: every centroid moves to the average of its customers. "
        "The **WCSS** (within-cluster sum of squares) adds up every customer's squared distance to its "
        "centroid, in standard units: lower means tighter clusters."
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    k = c1.slider("Number of clusters (k)", 2, 8, 4, key="steps_k")
    init_label = c2.radio("Starting centroids", list(INIT_LABELS), index=0, key="steps_init")
    init = INIT_LABELS[init_label]
    seed = c3.number_input("Random seed", 0, 999, 4, key="steps_seed", disabled=init == "manual",
                           help="Changes which customers are picked at random as starting centroids.")
    max_iter = c4.slider("Maximum iterations", 1, 30, 30, key="steps_max_iter")
    tol = c5.select_slider("Tolerance (standard units)", [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0], 0.0, key="steps_tol",
                           help="Stop as soon as no centroid moves more than this. 0 = stop only when nothing changes.")

    if "picks" not in st.session_state:
        st.session_state.picks = []
        st.session_state.pick_round = 0

    def clear_picks():
        st.session_state.picks = []
        st.session_state.pick_round += 1

    run_result = None
    if init == "manual":
        picks = st.session_state.picks[:k]
        st.markdown(f"**Click {k} customers** on the chart to use them as your starting centroids "
                    f"({len(picks)} of {k} picked).")
        st.button("Clear my picks", on_click=clear_picks, key="clear_picks")
        if len(picks) < k:
            event = st.plotly_chart(customer_chart(picks=picks, title=f"Click {k - len(picks)} more customer(s)"),
                                    width="stretch", on_select="rerun", selection_mode="points",
                                    key=f"pick_chart_{st.session_state.pick_round}")
            for point in event.selection.points:
                if point.get("curve_number") != 0:  # trace 0 holds every customer, in customer-ID order
                    continue
                cid = int(CUSTOMER_IDS[point["point_index"]])
                if cid not in st.session_state.picks and len(st.session_state.picks) < k:
                    st.session_state.picks.append(cid)
                    st.rerun()
        else:
            run_result = cached_run(k, init, 0, max_iter, tol, tuple(p - 1 for p in picks))
            st.caption(f"Your starting customers: {', '.join('#' + str(p) for p in picks)}")
    else:
        run_result = cached_run(k, init, int(seed), max_iter, tol, None)

    if run_result is not None:
        r = run_result
        n_stages = 2 * r["iterations"]
        signature = (k, init, int(seed), max_iter, tol, tuple(st.session_state.picks[:k]) if init == "manual" else ())
        if st.session_state.get("steps_signature") != signature:
            st.session_state.steps_signature = signature
            st.session_state.stage = 0

        def go_to(n):
            st.session_state.stage = min(max(n, 0), n_stages)

        stage = st.session_state.stage
        b1, b2, b3, b4, _ = st.columns([1, 1, 1, 1, 4])
        b1.button("◀ Back", on_click=go_to, args=(stage - 1,), disabled=stage == 0, key="steps_back")
        b2.button("Next step ▶", on_click=go_to, args=(stage + 1,), disabled=stage == n_stages, type="primary", key="steps_next")
        b3.button("⏭ Run to the end", on_click=go_to, args=(n_stages,), disabled=stage == n_stages, key="steps_end")
        b4.button("Reset", on_click=go_to, args=(0,), key="steps_reset")

        history = [r["start_centroids"]] + [s["centroids"] for s in r["steps"]]
        if stage == 0:
            labels, C, label = None, history[0], "Start: the starting centroids are placed"
            stage_wcss = None
        else:
            it = (stage + 1) // 2
            step = r["steps"][it - 1]
            labels = step["labels"]
            if stage % 2 == 1:
                C = history[it - 1]
                label = f"Iteration {it}, step 1 (assignment): {step['changed']} customer(s) joined a different cluster"
                if it == 1:
                    label = "Iteration 1, step 1 (assignment): every customer joins its nearest centroid"
                stage_wcss = ke.wcss(Z, labels, C)
            else:
                C = history[it]
                label = f"Iteration {it}, step 2 (update): centroids moved to the average of their customers (largest move {step['max_move']:.3f})"
                stage_wcss = step["wcss"]
        trails =np.array([to_original(c) for c in history[:((stage // 2) + 1)]]) if stage >= 2 else None
        st.progress(stage / n_stages, text=label)

        col_chart, col_info = st.columns([3, 1])
        with col_chart:
            st.plotly_chart(customer_chart(labels, to_original(C), trails, title=label), width="stretch", key="steps_chart")
        with col_info:
            st.metric("Iterations in this run", r["iterations"])
            if stage_wcss is not None:
                st.metric("WCSS right now", f"{stage_wcss:.2f}")
            if stage == n_stages:
                st.metric("Final WCSS", f"{r['wcss']:.2f}")
                (st.success if r["stop_reason"] == "converged" else st.warning)(STOP_TEXT[r["stop_reason"]])
                if r["stop_reason"] != "converged":
                    st.caption("Because it stopped early, each customer was assigned once more to its nearest final "
                               "centroid, so the final clusters match the final centroids.")
            if any(s["empty"] for s in r["steps"][:max(1, (stage + 1) // 2)]):
                st.error("A centroid ended up with no customers (an empty cluster). It stays where it is until "
                         "some customer is nearer to it than to any other centroid.")

        done = r["steps"][:stage // 2]
        if done:
            st.subheader("Iteration log")
            st.dataframe(pd.DataFrame({
                "Iteration": [s["iteration"] for s in done],
                "Customers that changed cluster": [s["changed"] for s in done],
                "Largest centroid move (standard units)": [round(s["max_move"], 3) for s in done],
                "WCSS after the update": [round(s["wcss"], 2) for s in done],
            }), hide_index=True, width="stretch")

# ------------------------------------------------------------ Restarts
with tab_restarts:
    st.header("Same data, different starting centroids, different answers")
    st.markdown(
        "Every run below uses the same customers and the same k. The only difference is the **random seed**, "
        "which decides the starting centroids. Each run keeps going until it converges."
    )
    c1, c2 = st.columns(2)
    k_r = c1.slider("Number of clusters (k)", 2, 8, 4, key="restarts_k")
    n_runs = c2.slider("Number of runs (seeds 0, 1, 2, ...)", 1, 50, 20, key="restarts_n")
    runs_random, runs_pp = cached_restarts(k_r, n_runs)
    best_wcss = min(r["wcss"] for r in runs_random + runs_pp)
    reached = lambda runs: sum(r["wcss"] <= best_wcss * (1 + 1e-9) for r in runs)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best WCSS found (lowest)", f"{best_wcss:.2f}")
    col2.metric("Random starts that reached it", f"{reached(runs_random)} of {n_runs}")
    col3.metric("k-means++ starts that reached it", f"{reached(runs_pp)} of {n_runs}")
    col4.metric("Worst random-start WCSS", f"{max(r['wcss'] for r in runs_random):.2f}")

    fig = go.Figure()
    for row, (name, runs, color) in enumerate([("Random customers", runs_random, "#E94F64"), ("k-means++", runs_pp, "#4C8BF5")]):
        # a small fixed vertical offset per seed, so runs with the same WCSS don't hide behind each other
        fig.add_trace(go.Scatter(x=[r["wcss"] for r in runs], y=[row + 0.07 * (r["seed"] % 5 - 2) for r in runs],
                                 mode="markers", name=name, marker=dict(size=11, color=color, opacity=0.6),
                                 customdata=[r["seed"] for r in runs], hovertemplate="seed %{customdata}: WCSS %{x:.2f}<extra></extra>"))
    fig.add_vline(x=best_wcss, line_dash="dash", annotation_text="best found")
    fig.update_layout(title="Final WCSS of every run (each dot is one run; lower is better)", xaxis_title="Final WCSS",
                      height=300, margin=dict(t=50), showlegend=False,
                      yaxis=dict(tickvals=[0, 1], ticktext=["Random customers", "k-means++"], range=[-0.5, 1.5]))
    st.plotly_chart(fig, width="stretch", key="restarts_strip")

    worst = max(runs_random, key=lambda r: r["wcss"])
    col_best, col_worst = st.columns(2)
    best_run = min(runs_random + runs_pp, key=lambda r: r["wcss"])
    col_best.plotly_chart(customer_chart(best_run["labels"], to_original(best_run["centroids"]), height=430,
                                         title=f"Best: {best_run['init']} seed {best_run['seed']}, WCSS {best_run['wcss']:.2f}"),
                          width="stretch", key="restarts_best")
    col_worst.plotly_chart(customer_chart(worst["labels"], to_original(worst["centroids"]), height=430,
                                          title=f"Worst random start: seed {worst['seed']}, WCSS {worst['wcss']:.2f}"),
                           width="stretch", key="restarts_worst")
    with st.expander("Every run, seed by seed"):
        st.dataframe(pd.DataFrame({
            "Seed": [r["seed"] for r in runs_random],
            "Random customers: final WCSS": [round(r["wcss"], 2) for r in runs_random],
            "k-means++: final WCSS": [round(r["wcss"], 2) for r in runs_pp],
        }), hide_index=True, width="stretch")

# ------------------------------------------------------------ Choosing k & Segments
with tab_k:
    st.header("How many segments should ShopSmart use?")
    st.markdown(
        "For each k from 1 to 10, the app keeps the best of 10 k-means++ runs. **Elbow method**: WCSS always "
        "goes down as k grows (more centroids means every customer is closer to one), so look for the k where "
        "adding another cluster stops helping much. **Silhouette score** (−1 to 1): for each customer, compares "
        "the average distance to its own cluster with the average distance to the nearest other cluster; "
        "higher means clearer, better-separated clusters."
    )
    table = cached_elbow()
    col_e, col_s = st.columns(2)
    fig = go.Figure(go.Scatter(x=[r["k"] for r in table], y=[r["wcss"] for r in table], mode="lines+markers", line_color="#4C8BF5"))
    fig.update_layout(title="Elbow method: WCSS for each k", xaxis_title="k", yaxis_title="WCSS", height=360, margin=dict(t=50), xaxis_dtick=1)
    col_e.plotly_chart(fig, width="stretch", key="elbow_chart")
    sil = [r for r in table if r["silhouette"] is not None]
    fig = go.Figure(go.Scatter(x=[r["k"] for r in sil], y=[r["silhouette"] for r in sil], mode="lines+markers", line_color="#2BB673"))
    fig.update_layout(title="Silhouette score for each k", xaxis_title="k", yaxis_title="Silhouette score", height=360, margin=dict(t=50), xaxis_dtick=1)
    col_s.plotly_chart(fig, width="stretch", key="silhouette_chart")
    with st.expander("The numbers behind both charts"):
        st.dataframe(pd.DataFrame({"k": [r["k"] for r in table], "WCSS": [round(r["wcss"], 2) for r in table],
                                   "Silhouette": [None if r["silhouette"] is None else round(r["silhouette"], 3) for r in table]}),
                     hide_index=True)

    st.header("Segment report")
    k_seg = st.slider("Your choice of k", 2, 10, 4, key="segments_k")
    best = cached_best(k_seg)
    revenue = X[:, 0] * X[:, 1]
    rows = []
    for j in range(k_seg):
        m = best["labels"] == j
        rows.append({
            "Segment": f"Cluster {j + 1}",
            "Customers": int(m.sum()),
            "Share of customers": f"{m.mean():.0%}",
            "Avg. visits per month": round(X[m, 0].mean(), 1),
            "Avg. spend per visit (MXN)": round(X[m, 1].mean()),
            "Avg. monthly spend per customer (MXN)": round(revenue[m].mean()),
            "Share of total monthly revenue": f"{revenue[m].sum() / revenue.sum():.0%}",
        })
    st.plotly_chart(customer_chart(best["labels"], to_original(best["centroids"]), height=460,
                                   title=f"Best k-means++ result for k = {k_seg} (WCSS {best['wcss']:.2f})"),
                    width="stretch", key="segments_chart")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Monthly spend per customer = visits per month × spend per visit, computed for each customer and then "
                      "averaged. The revenue share adds it up over all customers in the segment.")

# ------------------------------------------------------------ Where k-means Fails
with tab_fail:
    st.header("Groups that k-means can't see")
    st.markdown(
        "k-means works best when groups are **round blobs of similar size**. Each dataset below has known "
        "true groups. k-means (best of 10 k-means++ runs, with k equal to the true number of groups) gets the "
        "same data without the answers."
    )
    name = st.radio("Dataset", list(DATASETS), key="fail_dataset")
    Xf, yf = DATASETS[name]
    best_f, true_wcss, agree = cached_failure(name)

    def plain_chart(labels, centroids, title):
        fig = go.Figure()
        for g in range(int(labels.max()) + 1):
            m = labels == g
            fig.add_trace(go.Scatter(x=Xf[m, 0], y=Xf[m, 1], mode="markers", name=f"Group {g + 1} ({m.sum()})",
                                     marker=dict(color=COLORS[g], size=6, opacity=0.75), hoverinfo="skip"))
        if centroids is not None:
            fig.add_trace(go.Scatter(x=centroids[:, 0], y=centroids[:, 1], mode="markers", name="Centroids",
                                     marker=dict(symbol="x", size=16, color="black", line=dict(width=2, color="white"))))
        fig.update_layout(title=title, height=430, margin=dict(t=50), legend=dict(orientation="h", y=-0.12),
                          yaxis_scaleanchor="x")
        return fig

    col_true, col_km = st.columns(2)
    col_true.plotly_chart(plain_chart(yf, None, "The true groups"), width="stretch", key="fail_true")
    col_km.plotly_chart(plain_chart(best_f["labels"], best_f["centroids"], "What k-means found"), width="stretch", key="fail_km")
    col1, col2, col3 = st.columns(3)
    col1.metric("Points k-means put in their true group", f"{agree:.1%}")
    col2.metric("WCSS of k-means' answer", f"{best_f['wcss']:.1f}")
    col3.metric("WCSS of the true groups", f"{true_wcss:.1f}")
    if best_f["wcss"] < true_wcss:
        st.warning(
            "k-means' answer has a **lower WCSS than the true groups**. So k-means did exactly what it is built to "
            "do: it found clusters with a smaller total squared distance than the real groups have. Running it "
            "longer or from better starting points would not fix this. The problem is the goal itself: "
            "\"smallest total squared distance to a center\" doesn't describe these groups."
        )

# ------------------------------------------------------------ Bonus: Photo Compression
with tab_photo:
    st.header("Bonus: k-means picks the colors")
    st.markdown(
        "A photo is a long list of pixels, and every pixel is a point in **color space** (red, green, blue, each "
        "0–255). Run k-means on the pixels and every centroid is a color: replace each pixel by its centroid "
        "and the photo uses only **k colors**. This is called *color quantization*, and it is how GIF images "
        "and many apps shrink photos."
    )
    c1, c2, c3, c4 = st.columns(4)
    choice = c1.selectbox("Photo", list(PHOTOS) + [UPLOAD], key="photo_choice")
    k_p = c2.slider("Number of colors (k)", 2, 32, 8, key="photo_k")
    init_p = c3.radio("Starting centroids", ["k-means++", "random"], key="photo_init")
    seed_p = c4.number_input("Random seed", 0, 999, 0, key="photo_seed")

    image_bytes, credit = None, ""
    if choice == UPLOAD:
        uploaded = st.file_uploader("Choose a JPG or PNG photo", type=["jpg", "jpeg", "png"], key="photo_upload")
        st.caption("Your photo is processed in memory to make the compressed version. This app does not save it.")
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
    else:
        file_name, credit = PHOTOS[choice]
        image_bytes = (IMAGES_DIR / file_name).read_bytes()

    @st.cache_data(max_entries=32, show_spinner=False)
    def quantize(data, k, init, seed):
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.thumbnail((600, 600))
        pixels = np.asarray(img, dtype=float).reshape(-1, 3)
        rng = np.random.default_rng(seed)
        sample = pixels[rng.choice(len(pixels), size=min(10_000, len(pixels)), replace=False)]
        r = ke.run(sample, k, init, seed, max_iter=50)
        labels = np.concatenate([ke.assign(chunk, r["centroids"]) for chunk in np.array_split(pixels, max(1, len(pixels) // 20_000))])
        palette = np.clip(np.round(r["centroids"]), 0, 255).astype(np.uint8)
        out = palette[labels].reshape(img.size[1], img.size[0], 3)
        rms = float(np.sqrt(np.mean((pixels - palette[labels].astype(float)) ** 2)))
        return np.asarray(img), out, palette, rms, r["iterations"], r["stop_reason"]

    if image_bytes is None:
        st.info("Upload a photo to try it on your own picture.")
    else:
        original, compressed, palette, rms, iters, stop = quantize(image_bytes, k_p, init_p, int(seed_p))
        h, w = original.shape[:2]
        bits_per_pixel = math.ceil(math.log2(k_p))
        original_kb = w * h * 24 / 8 / 1024
        compressed_kb = (w * h * bits_per_pixel + k_p * 24) / 8 / 1024
        col_o, col_c = st.columns(2)
        col_o.image(original, caption=f"Original ({w} x {h} pixels, up to 16.7 million possible colors). {credit}", width="stretch")
        col_c.image(compressed, caption=f"Only {k_p} colors (k-means: {iters} iterations, {stop})", width="stretch")
        st.image(np.repeat(np.repeat(palette[None, :, :], 40, axis=0), 40, axis=1), caption="The palette: the k centroids, as colors")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Bits per pixel (original: 24)", f"{bits_per_pixel}")
        col2.metric("Uncompressed size, original", f"{original_kb:,.0f} KB")
        col3.metric(f"Uncompressed size, {k_p} colors", f"{compressed_kb:,.0f} KB ({original_kb / compressed_kb:.1f}x smaller)")
        col4.metric("Average error per color channel", f"{rms:.1f}", help="Root-mean-square difference between original and compressed pixels, on the 0–255 scale.")
        st.caption(
            "Sizes are *uncompressed*: 24 bits per pixel for the original; for k colors, each pixel stores only which "
            f"palette color it uses (⌈log₂ k⌉ = {bits_per_pixel} bits) plus the palette itself (k × 24 bits). "
            "1 KB = 1,024 bytes. PNG or JPEG files would be smaller still, because they add further compression."
        )
