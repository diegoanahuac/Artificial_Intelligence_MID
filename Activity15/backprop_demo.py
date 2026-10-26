"""One back-propagation update on a tiny 2-2-1 network, with every intermediate
number exposed (Session 26).

    x1 --w1--> h1        h1 --w5--> o
    x2 --w2--> h1        h2 --w6--> o
    x1 --w3--> h2
    x2 --w4--> h2        biases: b1 (h1), b2 (h2), b3 (o)

Every neuron uses the sigmoid activation, and the error is the squared error
E = 1/2 (target - o)^2. Inputs, target, and starting weights are fixed, so the
default learning rate always produces the same numbers (the graded ones).
"""

import math

INPUTS = {"x1": 0.8, "x2": 0.2}
TARGET = 1.0
INITIAL_WEIGHTS = {
    "w1": 0.4, "w2": -0.3, "w3": 0.2, "w4": 0.7, "w5": 0.5, "w6": -0.6,
    "b1": 0.1, "b2": -0.1, "b3": 0.2,
}
DEFAULT_LEARNING_RATE = 0.5


def _sigmoid(z):
    return 1 / (1 + math.exp(-z))


def forward(w):
    x1, x2 = INPUTS["x1"], INPUTS["x2"]
    net_h1 = w["w1"] * x1 + w["w2"] * x2 + w["b1"]
    net_h2 = w["w3"] * x1 + w["w4"] * x2 + w["b2"]
    h1, h2 = _sigmoid(net_h1), _sigmoid(net_h2)
    net_o = w["w5"] * h1 + w["w6"] * h2 + w["b3"]
    o = _sigmoid(net_o)
    return {"net_h1": net_h1, "h1": h1, "net_h2": net_h2, "h2": h2, "net_o": net_o, "o": o,
            "error": 0.5 * (TARGET - o) ** 2}


def update(w, learning_rate):
    """One full forward + backward pass. Returns every intermediate value."""
    x1, x2 = INPUTS["x1"], INPUTS["x2"]
    f = forward(w)
    h1, h2, o = f["h1"], f["h2"], f["o"]

    # Output delta: how the error changes with the output neuron's net input.
    delta_o = (o - TARGET) * o * (1 - o)
    # Hidden deltas: the output delta sent back through w5/w6, times each sigmoid's slope.
    delta_h1 = delta_o * w["w5"] * h1 * (1 - h1)
    delta_h2 = delta_o * w["w6"] * h2 * (1 - h2)

    gradients = {
        "w1": delta_h1 * x1, "w2": delta_h1 * x2, "w3": delta_h2 * x1, "w4": delta_h2 * x2,
        "w5": delta_o * h1, "w6": delta_o * h2,
        "b1": delta_h1, "b2": delta_h2, "b3": delta_o,
    }
    new_weights = {k: w[k] - learning_rate * gradients[k] for k in w}
    return {
        "forward": f,
        "delta_o": delta_o,
        "delta_h1": delta_h1,
        "delta_h2": delta_h2,
        "gradients": gradients,
        "new_weights": new_weights,
        "error_after": forward(new_weights)["error"],
    }


def error_curve(w, learning_rate, steps):
    """Error before each of `steps` repeated updates, plus the final error."""
    errors = [forward(w)["error"]]
    for _ in range(steps):
        w = update(w, learning_rate)["new_weights"]
        errors.append(forward(w)["error"])
    return errors
