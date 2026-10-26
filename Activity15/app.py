"""Digit Lab: Teaching a Neural Network to Read Handwriting
— Activity 15 (Sessions 25-27).

Run locally with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import backprop_demo as bp
from mnist_data import IMAGE_SIDE, TEST_IMAGES, TRAIN_IMAGES, Y_TEST, Y_TRAIN
from nn_engine import (
    ACTIVATIONS, CANONICAL, CANONICAL_CONFIG, CANONICAL_TEST_PREDICTIONS, CANONICAL_TEST_PROBS,
    INITS, predict_proba, train,
)

st.set_page_config(page_title="Digit Lab: Neural Networks", page_icon="🔢", layout="wide")

TRAIN_COLOR = "#4C8BF5"
TEST_COLOR = "#E94F64"

st.title("🔢 Digit Lab: Teaching a Neural Network to Read Handwriting")
st.markdown(
    "Banks read the amounts on checks and post offices read ZIP codes with exactly this kind of "
    "system: a **neural network** that looks at the pixels of a handwritten digit and decides which "
    f"of the 10 digits it is. This network learns from **{len(Y_TRAIN):,} training images** of real "
    f"handwriting (the classic MNIST dataset) and is checked on **{len(Y_TEST):,} test images** it "
    "never sees during training."
)


def upscale(img, scale):
    """Nearest-neighbor enlargement so tiny 28x28 images stay crisp on screen."""
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)


def weight_image(w):
    """784 incoming weights -> 28x28 RGB: red = positive, blue = negative, white = ~0."""
    w = w.reshape(IMAGE_SIDE, IMAGE_SIDE) / (np.abs(w).max() or 1.0)
    pos, neg = np.clip(w, 0, 1), np.clip(-w, 0, 1)
    rgb = np.stack([1 - neg, 1 - pos - neg, 1 - pos], axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def curves_chart(history, key, title, yaxis):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["epoch"], y=history[f"train_{key}"], name="Train", mode="lines+markers", line_color=TRAIN_COLOR))
    fig.add_trace(go.Scatter(x=history["epoch"], y=history[f"test_{key}"], name="Test", mode="lines+markers", line_color=TEST_COLOR))
    fig.update_layout(title=title, xaxis_title="Epoch (0 = before training)", yaxis_title=yaxis, height=360, margin=dict(t=50))
    if key == "accuracy":
        fig.update_yaxes(tickformat=".0%")
    return fig


def probability_chart(probs, title):
    best = int(np.argmax(probs))
    colors = [TRAIN_COLOR if d == best else "#B0B7C3" for d in range(10)]
    fig = go.Figure(go.Bar(x=[str(d) for d in range(10)], y=probs, marker_color=colors,
                           text=[f"{p:.1%}" for p in probs], textposition="outside"))
    fig.update_layout(title=title, xaxis_title="Digit", yaxis_title="Probability", yaxis_range=[0, 1.15],
                      yaxis_tickformat=".0%", height=320, margin=dict(t=50), xaxis_type="category")
    return fig


def preprocess_drawing(rgba):
    """Turn a 280x280 canvas drawing into a 28x28 MNIST-style image, or None if blank.

    Same recipe MNIST used: crop to the digit, shrink it to fit a 20x20 box,
    then place it in a 28x28 frame so its center of mass sits in the middle.
    """
    ink = rgba[:, :, :3].astype(float).mean(axis=2) * (rgba[:, :, 3] / 255.0)
    if ink.max() < 25:
        return None
    rows = np.flatnonzero(ink.max(axis=1) > 25)
    cols = np.flatnonzero(ink.max(axis=0) > 25)
    crop = ink[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    h, w = crop.shape
    scale = 20 / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    small = np.asarray(Image.fromarray(crop.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS), dtype=float)

    ys, xs = np.indices(small.shape)
    total = small.sum()
    top = int(np.clip(round(IMAGE_SIDE / 2 - (ys * small).sum() / total), 0, IMAGE_SIDE - new_h))
    left = int(np.clip(round(IMAGE_SIDE / 2 - (xs * small).sum() / total), 0, IMAGE_SIDE - new_w))
    frame = np.zeros((IMAGE_SIDE, IMAGE_SIDE))
    frame[top:top + new_h, left:left + new_w] = small
    return frame / frame.max()


@st.cache_data(max_entries=64, show_spinner=False)
def cached_train(hidden_layers, activation, learning_rate, epochs, batch_size, init):
    return train(hidden_layers, activation, learning_rate, epochs, batch_size, init)


tab_meet, tab_backprop, tab_lab, tab_draw = st.tabs(
    ["🧠 Meet the Network", "🔁 Back-propagation Step by Step", "🎛️ Training Lab", "✏️ Draw & Classify"]
)

# ------------------------------------------------------------ Meet the Network tab
with tab_meet:
    st.header("1. What the network sees")
    st.markdown(
        "A biological neuron receives signals from other neurons, and *fires* when the combined "
        "signal is strong enough. An artificial neuron does the same with numbers: it multiplies "
        "each input by a **weight**, adds them up (plus a **bias**), and passes the total through an "
        "**activation function**. Here, the inputs are pixels."
    )
    examples = [int(np.flatnonzero(Y_TRAIN == d)[0]) for d in range(10)]
    st.image([upscale(TRAIN_IMAGES[i], 3) for i in examples], caption=[f"Label: {d}" for d in range(10)], width=84)

    col_img, col_grid = st.columns([1, 3])
    with col_img:
        digit = st.selectbox("Pick a training example to look inside", range(10), index=3, key="meet_digit")
        image = TRAIN_IMAGES[examples[digit]]
        st.image(upscale(image, 8), caption=f"A handwritten {digit}, 28 x 28 pixels", width=224)
    with col_grid:
        fig = go.Figure(go.Heatmap(z=image, colorscale="gray", showscale=False, text=image, texttemplate="%{text}",
                                   textfont={"size": 7}, hovertemplate="row %{y}, col %{x}: %{z}<extra></extra>"))
        fig.update_layout(title="The same image as the network receives it: 784 numbers (0 = black, 255 = white)",
                          height=560, margin=dict(t=50), yaxis_autorange="reversed")
        st.plotly_chart(fig, width="stretch")

    st.header("2. The canonical network")
    hidden = CANONICAL_CONFIG["hidden_layers"][0]
    st.graphviz_chart(
        f"""digraph {{ rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#EEF3FE", fontname="Helvetica"];
        input [label="Input layer\\n784 neurons\\n(one per pixel)"];
        hidden [label="Hidden layer\\n{hidden} neurons\\n({CANONICAL_CONFIG['activation']} activation)"];
        output [label="Output layer\\n10 neurons\\n(softmax: one per digit 0-9)"];
        input -> hidden [label="784 x {hidden} weights\\n+ {hidden} biases"];
        hidden -> output [label="{hidden} x 10 weights\\n+ 10 biases"]; }}"""
    )
    st.markdown(
        f"Fully connected: every input neuron connects to every hidden neuron, and every hidden neuron "
        f"to every output neuron. Trained with **{CANONICAL_CONFIG['epochs']} epochs**, learning rate "
        f"**{CANONICAL_CONFIG['learning_rate']}**, batch size **{CANONICAL_CONFIG['batch_size']}**, "
        f"**{CANONICAL_CONFIG['init']}** initial weights."
    )
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Trainable parameters", f"{CANONICAL['num_parameters']:,}")
    col2.metric("Train accuracy", f"{CANONICAL['train_accuracy']:.1%}")
    col3.metric("Test accuracy", f"{CANONICAL['test_accuracy']:.1%}")
    col4.metric("Train loss", f"{CANONICAL['train_loss']:.3f}")
    col5.metric("Test loss", f"{CANONICAL['test_loss']:.3f}")

    col_loss, col_acc = st.columns(2)
    col_loss.plotly_chart(curves_chart(CANONICAL["history"], "loss", "Cross-entropy loss per epoch", "Loss"), width="stretch", key="canon_loss")
    col_acc.plotly_chart(curves_chart(CANONICAL["history"], "accuracy", "Accuracy per epoch", "Accuracy"), width="stretch", key="canon_acc")

    confusion = np.zeros((10, 10), dtype=int)
    np.add.at(confusion, (Y_TEST, CANONICAL_TEST_PREDICTIONS), 1)
    fig = go.Figure(go.Heatmap(z=confusion, x=[str(d) for d in range(10)], y=[str(d) for d in range(10)], colorscale="Blues",
                               text=confusion, texttemplate="%{text}", showscale=False,
                               hovertemplate="true %{y}, predicted %{x}: %{z} images<extra></extra>"))
    fig.update_layout(title="Confusion matrix on the 1,000 test images (diagonal = correct)", xaxis_title="Predicted digit",
                      yaxis_title="True digit", yaxis_autorange="reversed", height=480, margin=dict(t=50), xaxis_type="category", yaxis_type="category")
    st.plotly_chart(fig, width="stretch")

    st.header("3. What did the hidden neurons learn?")
    st.markdown(
        f"Each hidden neuron has 784 incoming weights, one per pixel, so its weights can be drawn as a "
        f"28 x 28 image. **Red** pixels push the neuron to fire when that spot is bright (ink); **blue** "
        f"pixels push against it. Nobody programmed these patterns: back-propagation found them."
    )
    first_layer = CANONICAL["layers"][0][0]
    st.image([upscale(weight_image(first_layer[:, k]), 3) for k in range(first_layer.shape[1])],
             caption=[f"Neuron {k}" for k in range(first_layer.shape[1])], width=84)

# ------------------------------------------------------------ Back-propagation tab
with tab_backprop:
    st.header("One training step, every number shown")
    st.markdown(
        "Training the digit network repeats the same step thousands of times. To see what happens in "
        "*one* step, here is a tiny network: **2 inputs, 2 hidden neurons, 1 output**, all using the "
        f"sigmoid activation. The input is x1 = **{bp.INPUTS['x1']}**, x2 = **{bp.INPUTS['x2']}**, and "
        f"the correct answer (target) is **{bp.TARGET}**. The error is the squared error "
        "E = ½ (target − output)²."
    )
    learning_rate = st.number_input("Learning rate (η)", 0.01, 10.0, bp.DEFAULT_LEARNING_RATE, 0.1, key="bp_lr",
                                    help=f"Graded answers use the default of {bp.DEFAULT_LEARNING_RATE}.")
    stages = ["Start", "1. Forward pass", "2. Error", "3. Output delta", "4. Hidden deltas & gradients", "5. Update weights"]
    if "bp_stage" not in st.session_state:
        st.session_state.bp_stage = 0

    def go_to_stage(n):
        st.session_state.bp_stage = min(max(n, 0), len(stages) - 1)

    stage = st.session_state.bp_stage
    col_prev, col_next, col_reset, _ = st.columns([1, 1, 1, 5])
    col_prev.button("◀ Back", disabled=stage == 0, on_click=go_to_stage, args=(stage - 1,))
    col_next.button("Next step ▶", type="primary", disabled=stage == len(stages) - 1, on_click=go_to_stage, args=(stage + 1,))
    col_reset.button("Reset", on_click=go_to_stage, args=(0,))
    st.progress(stage / (len(stages) - 1), text=f"Step {stage} of {len(stages) - 1}: {stages[stage]}")

    w = bp.INITIAL_WEIGHTS
    r = bp.update(w, learning_rate)
    f = r["forward"]
    show_new = stage >= 5

    def edge(name):
        return f"{name} = {r['new_weights'][name]:.4f}" if show_new else f"{name} = {w[name]:.4f}"

    def node(label, value, bias=None):
        lines = [label]
        if bias:
            lines.append(edge(bias))
        if stage >= 1 and value is not None:
            lines.append(f"= {value:.4f}")
        return "\\n".join(lines)

    st.graphviz_chart(
        f"""digraph {{ rankdir=LR; splines=line; nodesep=0.8; ranksep=2.2;
        node [shape=circle, style=filled, fontname="Helvetica", fontsize=11, width=1.3, fixedsize=true];
        x1 [label="x1\\n{bp.INPUTS['x1']}", fillcolor="#EEF3FE"]; x2 [label="x2\\n{bp.INPUTS['x2']}", fillcolor="#EEF3FE"];
        h1 [label="{node('h1', f['h1'], 'b1')}", fillcolor="#FFF4D6"]; h2 [label="{node('h2', f['h2'], 'b2')}", fillcolor="#FFF4D6"];
        o [label="{node('o', f['o'], 'b3')}", fillcolor="#E3F6E8"];
        edge [fontname="Helvetica", fontsize=10];
        x1 -> h1 [label="{edge('w1')}"]; x2 -> h1 [label="{edge('w2')}"];
        x1 -> h2 [label="{edge('w3')}"]; x2 -> h2 [label="{edge('w4')}"];
        h1 -> o [label="{edge('w5')}"]; h2 -> o [label="{edge('w6')}"]; }}"""
    )

    if show_new:
        st.caption("The connections now show the **updated** weights. The neuron values are still the ones from "
                   "the forward pass with the old weights.")
    if stage == 0:
        st.info("Press **Next step ▶** to send the input forward through the network.")
    if stage >= 1:
        st.subheader("1. Forward pass")
        st.markdown(
            f"- h1: net = w1·x1 + w2·x2 + b1 = **{f['net_h1']:.4f}**, sigmoid → h1 = **{f['h1']:.4f}**\n"
            f"- h2: net = w3·x1 + w4·x2 + b2 = **{f['net_h2']:.4f}**, sigmoid → h2 = **{f['h2']:.4f}**\n"
            f"- o: net = w5·h1 + w6·h2 + b3 = **{f['net_o']:.4f}**, sigmoid → output o = **{f['o']:.4f}**"
        )
    if stage >= 2:
        st.subheader("2. Error")
        st.markdown(f"E = ½ ({bp.TARGET} − {f['o']:.4f})² = **{f['error']:.4f}**. The network's answer is too low.")
    if stage >= 3:
        st.subheader("3. Output delta (start of the backward pass)")
        st.markdown(
            f"δo = (o − target) · o · (1 − o) = **{r['delta_o']:.4f}**. The sign says which way the output "
            "neuron's net input should move to lower the error, and the size says how strongly."
        )
    if stage >= 4:
        st.subheader("4. Send the blame backward")
        st.markdown(
            f"Each hidden neuron gets a share of δo through the weight that connects it to the output, "
            f"times its own sigmoid slope h · (1 − h):\n"
            f"- δh1 = δo · w5 · h1 · (1 − h1) = **{r['delta_h1']:.4f}**\n"
            f"- δh2 = δo · w6 · h2 · (1 − h2) = **{r['delta_h2']:.4f}**\n\n"
            "Each weight's **gradient** is the delta of the neuron it feeds into, times the value "
            "flowing along that connection (for example, gradient of w5 = δo · h1)."
        )
    if stage >= 5:
        st.subheader("5. Update every weight: new = old − η · gradient")
        g = r["gradients"]
        st.dataframe(pd.DataFrame({
            "Weight": list(w),
            "Old value": [round(w[k], 4) for k in w],
            "Gradient": [round(g[k], 4) for k in w],
            "New value": [round(r["new_weights"][k], 4) for k in w],
        }), hide_index=True)
        col_a, col_b = st.columns(2)
        col_a.metric("Error before the update", f"{f['error']:.4f}")
        col_b.metric("Error after one update", f"{r['error_after']:.4f}", f"{r['error_after'] - f['error']:.4f}", delta_color="inverse")
        st.success("One step of back-propagation is done. Training is just this step, repeated.")

        errors = bp.error_curve(w, learning_rate, 100)
        fig = go.Figure(go.Scatter(x=list(range(len(errors))), y=errors, mode="lines", line_color=TRAIN_COLOR))
        fig.update_layout(title=f"Repeat the same step 100 times (η = {learning_rate})", xaxis_title="Number of updates",
                          yaxis_title="Error E", height=320, margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------ Training Lab tab
with tab_lab:
    st.header("Train your own network")
    st.markdown(
        "Change the **hyperparameters** (the settings you choose *before* training) and press **Train**. "
        "Training is deterministic: the same settings always give exactly the same network, so leaving "
        "every setting at its default reproduces the canonical network."
    )
    col1, col2, col3, col4 = st.columns(4)
    num_layers = col1.radio("Hidden layers", [1, 2], index=0, horizontal=True)
    neurons = col1.select_slider("Neurons per hidden layer", [8, 16, 32, 64, 128], value=CANONICAL_CONFIG["hidden_layers"][0])
    activation = col2.selectbox("Activation function", ACTIVATIONS, index=ACTIVATIONS.index(CANONICAL_CONFIG["activation"]))
    init = col2.selectbox("Initial weights", INITS, index=INITS.index(CANONICAL_CONFIG["init"]))
    lr = col3.select_slider("Learning rate", [0.001, 0.01, 0.1, 0.5, 1.0, 3.0, 10.0], value=CANONICAL_CONFIG["learning_rate"])
    epochs = col3.slider("Epochs", 1, 20, CANONICAL_CONFIG["epochs"])
    batch_size = col4.select_slider("Batch size", [16, 32, 64, 128, 256], value=CANONICAL_CONFIG["batch_size"])
    col4.write("")
    train_clicked = col4.button("🚀 Train", type="primary", width="stretch")

    if "lab_runs" not in st.session_state:
        st.session_state.lab_runs = []
    if train_clicked:
        with st.spinner("Training... (a few seconds)"):
            model = cached_train((neurons,) * num_layers, activation, lr, epochs, batch_size, init)
        st.session_state.lab_model = model
        st.session_state.lab_runs.append({
            "Run": len(st.session_state.lab_runs) + 1,
            "Hidden layers": " x ".join([str(neurons)] * num_layers),
            "Activation": activation,
            "Learning rate": lr,
            "Epochs": epochs,
            "Batch size": batch_size,
            "Initial weights": init,
            "Train accuracy": f"{model['train_accuracy']:.1%}",
            "Test accuracy": f"{model['test_accuracy']:.1%}",
            "Test loss": round(model["test_loss"], 3),
        })

    model = st.session_state.get("lab_model")
    if model is None:
        st.info("Choose your settings and press **🚀 Train**.")
    else:
        c = model["config"]
        st.subheader(f"Run {len(st.session_state.lab_runs)}: {' x '.join(map(str, c['hidden_layers']))} hidden, "
                     f"{c['activation']}, learning rate {c['learning_rate']}, {c['epochs']} epochs, "
                     f"batch {c['batch_size']}, {c['init']} weights")
        if model["diverged"]:
            st.error("Training blew up: some weights became infinitely large (not a number), so training stopped. "
                     "The learning rate is far too high for these settings.")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Train accuracy", f"{model['train_accuracy']:.1%}")
        col2.metric("Test accuracy", f"{model['test_accuracy']:.1%}",
                    f"{(model['test_accuracy'] - CANONICAL['test_accuracy']) * 100:+.1f} pts vs canonical")
        col3.metric("Train − test gap", f"{(model['train_accuracy'] - model['test_accuracy']) * 100:.1f} pts")
        col4.metric("Trainable parameters", f"{model['num_parameters']:,}")
        if model["test_accuracy"] <= 0.2:
            st.warning("This network is about as good as guessing at random (10%). Something about these "
                       "settings stopped it from learning.")
        col_loss, col_acc = st.columns(2)
        col_loss.plotly_chart(curves_chart(model["history"], "loss", "Cross-entropy loss per epoch", "Loss"), width="stretch", key="lab_loss")
        col_acc.plotly_chart(curves_chart(model["history"], "accuracy", "Accuracy per epoch", "Accuracy"), width="stretch", key="lab_acc")

    if st.session_state.lab_runs:
        st.subheader("Your run history")
        st.dataframe(pd.DataFrame(st.session_state.lab_runs), hide_index=True, width="stretch")
        if st.button("Clear run history"):
            st.session_state.lab_runs = []
            st.session_state.pop("lab_model", None)
            st.rerun()

# ------------------------------------------------------------ Draw & Classify tab
with tab_draw:
    lab_model = st.session_state.get("lab_model")
    st.header("1. Classify a test image")
    col_pick, col_probs = st.columns([1, 2])
    with col_pick:
        idx = st.number_input("Test image number (0-999)", 0, len(Y_TEST) - 1, 0, key="test_idx")
        st.image(upscale(TEST_IMAGES[idx], 6), width=168)
        probs = CANONICAL_TEST_PROBS[idx]
        pred = int(CANONICAL_TEST_PREDICTIONS[idx])
        st.markdown(f"**True label:** {Y_TEST[idx]}  \n**Canonical prediction:** {pred} ({probs[pred]:.1%} confident)")
        if pred == Y_TEST[idx]:
            st.success("Correct")
        else:
            st.error("Wrong")
    with col_probs:
        st.plotly_chart(probability_chart(probs, f"Canonical network: test image #{idx}"), width="stretch")
        if lab_model is not None:
            st.plotly_chart(probability_chart(predict_proba(lab_model, TEST_IMAGES[idx].reshape(-1) / 255.0)[0],
                                              "Your latest Training Lab network"), width="stretch")

    st.header("2. Draw your own digit")
    st.markdown(
        "Draw one digit, large and centered, with your mouse or finger. Use the canvas toolbar to undo or "
        "clear it. The app shrinks your drawing to 28 x 28 pixels the same way MNIST was prepared."
    )
    col_canvas, col_seen, col_result = st.columns([2, 1, 3])
    with col_canvas:
        canvas = st_canvas(stroke_width=20, stroke_color="#FFFFFF", background_color="#000000", height=280, width=280,
                           drawing_mode="freedraw", return_image_data=True, key="draw_canvas")
    drawing = preprocess_drawing(canvas.image_data) if canvas.image_data is not None else None
    if drawing is None:
        col_result.info("Draw a digit on the black canvas.")
    else:
        col_seen.image(upscale((drawing * 255).astype(np.uint8), 5), caption="What the network sees (28 x 28)", width=140)
        flat = drawing.reshape(-1)
        canon_probs = predict_proba(CANONICAL, flat)[0]
        col_result.plotly_chart(probability_chart(canon_probs, f"Canonical network says: {int(np.argmax(canon_probs))}"), width="stretch")
        if lab_model is not None:
            lab_probs = predict_proba(lab_model, flat)[0]
            col_result.plotly_chart(probability_chart(lab_probs, f"Your Training Lab network says: {int(np.argmax(lab_probs))}"), width="stretch")

    st.header("3. Where the canonical network gets it wrong")
    wrong = np.flatnonzero(CANONICAL_TEST_PREDICTIONS != Y_TEST)
    st.markdown(f"The canonical network misclassifies **{len(wrong)} of {len(Y_TEST):,}** test images. Each caption reads **image number: true label → what the network said**.")
    st.image([upscale(TEST_IMAGES[i], 3) for i in wrong],
             caption=[f"#{i}: {Y_TEST[i]} → {CANONICAL_TEST_PREDICTIONS[i]}" for i in wrong], width=84)
