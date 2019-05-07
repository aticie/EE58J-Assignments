import numpy as np
from utils import load_data
from sklearn.tree import DecisionTreeClassifier
import os

cwd = os.getcwd()
train_folder = os.path.join(cwd, "8x8mini")
test_folder = os.path.join(train_folder, "test")

X, y = load_data(train_folder)
X_test, Y_test = load_data(test_folder)

num_iterations = 100

# input: dataset X and labels y (in {+1, -1})
hypotheses = []
hypothesis_weights = []

N, _ = X.shape
d = np.ones(N) / N

for t in range(num_iterations):
    h = DecisionTreeClassifier(max_depth=1)

    h.fit(X, y, sample_weight=d)
    pred = h.predict(X)

    eps = d.dot(pred != y)
    alpha = (np.log(1 - eps) - np.log(eps)) / 2

    d = d * np.exp(- alpha * y * pred)
    d = d / d.sum()

    hypotheses.append(h)
    hypothesis_weights.append(alpha)