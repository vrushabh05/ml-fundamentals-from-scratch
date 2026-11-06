"""Gradient descent: (1) a univariate non-convex function, (2) multiple linear regression from scratch.

Part 2 shows why feature scaling matters. The original assignment ran it on California housing (MedInc, Population)
with unscaled features and lr=0.001, and the loss overflowed after ~40 iterations. California housing cannot be
downloaded here, so this script demonstrates the same failure and the fix on a seeded synthetic dataset with the
same kind of scale mismatch. It does NOT reproduce the housing numbers.
"""
import json

import numpy as np


# ---- 1. f(x) = x^4 + x^3 - 2x^2 ---------------------------------------------------------------
def f(x): return x ** 4 + x ** 3 - 2 * x ** 2
def df(x): return 4 * x ** 3 + 3 * x ** 2 - 4 * x


def descend(x0, lr=0.01, iters=20):
    xs = [x0]
    for _ in range(iters):
        xs.append(xs[-1] - lr * df(xs[-1]))
    return xs


# ---- 2. multiple linear regression ----------------------------------------------------------------
def gradient_descent(X, y, lr=0.001, n_iters=100, seed=0):
    rng = np.random.default_rng(seed)
    W, b = rng.normal(0, 0.1, X.shape[1]), rng.normal(0, 0.1)
    losses = []
    for _ in range(n_iters):
        err = y - (X @ W + b)
        loss = np.mean(err ** 2)
        losses.append(float(loss))
        if not np.isfinite(loss):
            break
        W = W + lr * (2 / len(y)) * (X.T @ err)   # = W - lr * dL/dW
        b = b + lr * (2 / len(y)) * err.sum()
    return W, b, losses


if __name__ == "__main__":
    out = {}
    for x0 in (2, -2):
        xs = descend(x0)
        out[f"x0={x0}"] = dict(end_x=round(xs[-1], 4), end_f=round(f(xs[-1]), 4))
    roots = sorted([0.0, (-3 + 73 ** 0.5) / 8, (-3 - 73 ** 0.5) / 8])
    out["stationary_points"] = {f"x={r:.4f}": dict(f=round(f(r), 4), kind="min" if df(r + 1e-3) > 0 > df(r - 1e-3) else ("max" if df(r + 1e-3) < 0 < df(r - 1e-3) else "-")) for r in roots}
    print(json.dumps(out, indent=2))

    rng = np.random.default_rng(0)
    n = 2000
    income = rng.gamma(4, 1, n) + 0.5                  # ~0.5-15, like MedInc
    population = rng.gamma(2, 700, n) + 3              # ~3-10,000+, like Population
    y = 0.4 * income + 0.0002 * population + 0.5 + rng.normal(0, 0.3, n)
    X = np.c_[income, population]
    Xs = (X - X.mean(0)) / X.std(0)
    _, _, raw = gradient_descent(X, y, lr=0.001, n_iters=50)
    Ws, bs, scaled = gradient_descent(Xs, y, lr=0.1, n_iters=200)
    out2 = {
        "unscaled_lr0.001": dict(iterations_run=len(raw), final_loss=raw[-1], diverged=not np.isfinite(raw[-1]) or raw[-1] > raw[0]),
        "scaled_lr0.1": dict(iterations_run=len(scaled), start_loss=round(scaled[0], 4), final_loss=round(scaled[-1], 4)),
        "noise_floor_mse": 0.09,
    }
    print(json.dumps(out2, indent=2, default=float))
    json.dump({"univariate": out, "regression_demo": out2}, open("results/gradient_descent.json", "w"), indent=2, default=float)
