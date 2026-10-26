"""A small fully-connected neural network trained with back-propagation,
implemented from scratch in NumPy (Sessions 25-26). No scikit-learn, no PyTorch.

- Hidden layers use sigmoid, tanh, or ReLU; the output layer is softmax over the
  10 digits, trained with the cross-entropy cost.
- Training is mini-batch gradient descent. Weight initialization and the batch
  shuffle order both come from one seeded random generator, so the same
  settings always produce exactly the same network.

The **canonical model** (fixed settings, trained once at import time) is what
every reflection question is graded against.
"""

import numpy as np

from mnist_data import NUM_CLASSES, NUM_PIXELS, X_TEST, X_TRAIN, Y_TEST, Y_TRAIN

ACTIVATIONS = ("sigmoid", "tanh", "relu")
INITS = ("small random", "all zeros", "large random")
SEED = 42

CANONICAL_CONFIG = {
    "hidden_layers": (64,),
    "activation": "sigmoid",
    "learning_rate": 0.5,
    "epochs": 10,
    "batch_size": 32,
    "init": "small random",
}


def _activate(z, name):
    if name == "sigmoid":
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    if name == "tanh":
        return np.tanh(z)
    return np.maximum(z, 0.0)


def _activation_slope(a, name):
    """Derivative of the activation, written in terms of its output a."""
    if name == "sigmoid":
        return a * (1 - a)
    if name == "tanh":
        return 1 - a ** 2
    return (a > 0).astype(float)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _init_layers(sizes, init, rng):
    layers = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        if init == "small random":
            W = rng.normal(0.0, np.sqrt(1.0 / fan_in), (fan_in, fan_out))
        elif init == "large random":
            W = rng.normal(0.0, 1.0, (fan_in, fan_out))
        else:
            W = np.zeros((fan_in, fan_out))
        layers.append((W, np.zeros(fan_out)))
    return layers


def _forward(layers, X, activation):
    """Returns the list of every layer's output: [input, hidden..., softmax]."""
    outputs = [X]
    for i, (W, b) in enumerate(layers):
        z = outputs[-1] @ W + b
        outputs.append(_softmax(z) if i == len(layers) - 1 else _activate(z, activation))
    return outputs


def _loss_and_accuracy(layers, X, y, activation):
    p = _forward(layers, X, activation)[-1]
    loss = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))
    return loss, float(np.mean(p.argmax(axis=1) == y))


def train(hidden_layers, activation, learning_rate, epochs, batch_size, init, seed=SEED) -> dict:
    assert activation in ACTIVATIONS and init in INITS
    rng = np.random.default_rng(seed)
    sizes = [NUM_PIXELS, *hidden_layers, NUM_CLASSES]
    layers = _init_layers(sizes, init, rng)
    targets = np.eye(NUM_CLASSES)[Y_TRAIN]
    n = len(X_TRAIN)

    history = {"epoch": [], "train_loss": [], "test_loss": [], "train_accuracy": [], "test_accuracy": []}
    diverged = False

    with np.errstate(over="ignore", invalid="ignore"):
        for epoch in range(epochs + 1):
            if epoch > 0:
                order = rng.permutation(n)
                for start in range(0, n, batch_size):
                    idx = order[start:start + batch_size]
                    outputs = _forward(layers, X_TRAIN[idx], activation)

                    # Backward pass: softmax + cross-entropy gives (prediction - target)
                    # at the output; each earlier layer's delta is the next layer's
                    # delta sent back through its weights, times the activation slope.
                    delta = (outputs[-1] - targets[idx]) / len(idx)
                    for i in range(len(layers) - 1, -1, -1):
                        W, b = layers[i]
                        grad_W = outputs[i].T @ delta
                        grad_b = delta.sum(axis=0)
                        if i > 0:
                            delta = (delta @ W.T) * _activation_slope(outputs[i], activation)
                        layers[i] = (W - learning_rate * grad_W, b - learning_rate * grad_b)

            train_loss, train_acc = _loss_and_accuracy(layers, X_TRAIN, Y_TRAIN, activation)
            test_loss, test_acc = _loss_and_accuracy(layers, X_TEST, Y_TEST, activation)
            if not all(np.all(np.isfinite(W)) for W, _ in layers):
                diverged = True
                break
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["train_accuracy"].append(train_acc)
            history["test_accuracy"].append(test_acc)

    return {
        "config": {
            "hidden_layers": tuple(hidden_layers),
            "activation": activation,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "init": init,
        },
        "layers": layers,
        "history": history,
        "diverged": diverged,
        "num_parameters": int(sum(W.size + b.size for W, b in layers)),
        "train_loss": history["train_loss"][-1],
        "test_loss": history["test_loss"][-1],
        "train_accuracy": history["train_accuracy"][-1],
        "test_accuracy": history["test_accuracy"][-1],
    }


def predict_proba(model, X):
    """Probabilities for each of the 10 digits, one row per image."""
    with np.errstate(over="ignore", invalid="ignore"):
        return _forward(model["layers"], np.atleast_2d(X), model["config"]["activation"])[-1]


CANONICAL = train(**CANONICAL_CONFIG)
CANONICAL_TEST_PROBS = predict_proba(CANONICAL, X_TEST)
CANONICAL_TEST_PREDICTIONS = CANONICAL_TEST_PROBS.argmax(axis=1)
