"""Bernoulli, Multinomial and Gaussian Naive Bayes on scikit-learn's 8x8 digits (1,797 images, 80/20 split, seed 0)."""
import json

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

X, y = load_digits(return_X_y=True)
X = X / 16.0  # the 8x8 digits are 0-16, so /16 scales to [0, 1]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

res = {}
bern = BernoulliNB().fit(np.where(Xtr > 0.5, 1, 0), ytr)
res["BernoulliNB (binary, threshold 0.5)"] = accuracy_score(yte, bern.predict(np.where(Xte > 0.5, 1, 0)))
tri = lambda a: np.digitize(a, bins=[1 / 3, 2 / 3])  # white / grey / black
mult = MultinomialNB().fit(tri(Xtr), ytr)
res["MultinomialNB (3 levels)"] = accuracy_score(yte, mult.predict(tri(Xte)))
res["GaussianNB (raw 0-1)"] = accuracy_score(yte, GaussianNB().fit(Xtr, ytr).predict(Xte))
for k, v in res.items():
    print(f"{k:40s} {v:.4f}")
json.dump({k: round(v, 4) for k, v in res.items()}, open("results/digits_nb.json", "w"), indent=2)
