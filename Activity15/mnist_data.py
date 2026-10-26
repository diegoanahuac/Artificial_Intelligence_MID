"""MNIST subset for Activity 15 (Sessions 25-27).

6,000 training images (600 per digit) and 1,000 test images (100 per digit),
taken once with a fixed seed from the classic MNIST handwritten-digit dataset
(LeCun, Cortes & Burges). Each image is 28 x 28 grayscale pixels, stored as
0-255 and scaled here to 0.0 (black) - 1.0 (white).
"""

from pathlib import Path

import numpy as np

IMAGE_SIDE = 28
NUM_PIXELS = IMAGE_SIDE * IMAGE_SIDE  # 784 input neurons
NUM_CLASSES = 10                      # 10 output neurons, one per digit 0-9

with np.load(Path(__file__).with_name("mnist_subset.npz"), allow_pickle=False) as _data:
    TRAIN_IMAGES = _data["x_train"]
    Y_TRAIN = _data["y_train"].astype(np.int64)
    TEST_IMAGES = _data["x_test"]
    Y_TEST = _data["y_test"].astype(np.int64)

X_TRAIN = TRAIN_IMAGES.reshape(len(TRAIN_IMAGES), NUM_PIXELS) / 255.0
X_TEST = TEST_IMAGES.reshape(len(TEST_IMAGES), NUM_PIXELS) / 255.0
