"""Ordinary least squares from the normal equation, w = (X'X)^-1 X'y, with a bias column.

The original assignment fit student exam scores on hours studied (25 rows) and obtained
intercept 2.4837 and slope 9.7758. That CSV is not in this repo, so this script checks the same closed-form
code against numpy.linalg.lstsq on seeded synthetic data. Use np.linalg.solve or lstsq in production rather than inv.
"""
import numpy as np


def ols(x, y):
    Xb = np.c_[np.ones(len(x)), x]
    return np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    hours = rng.uniform(1, 9.5, 25)
    scores = 2.5 + 9.8 * hours + rng.normal(0, 5, 25)
    w = ols(hours, scores)
    ref = np.linalg.lstsq(np.c_[np.ones(25), hours], scores, rcond=None)[0]
    print("normal equation :", w.round(4))
    print("numpy lstsq     :", ref.round(4))
    assert np.allclose(w, ref)
