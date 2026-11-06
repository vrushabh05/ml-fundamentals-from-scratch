import json, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
from gradient_descent import f, descend, gradient_descent

ens = json.load(open("results/ensembles.json")); nb = json.load(open("results/digits_nb.json")); knn = json.load(open("results/knn.json"))
fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.5), dpi=150)

a = ax[0, 0]; xs = np.linspace(-2.2, 1.6, 300); a.plot(xs, f(xs), color="#555")
for x0, c in ((2, "#2a7f62"), (-2, "#c2410c")):
    p = np.array(descend(x0)); a.plot(p, f(p), "o-", ms=3, color=c, label=f"start x={x0} (20 steps, lr 0.01)")
for r, l in ((-1.443, "global min -1.443"), (0.693, "local min 0.693")): a.axvline(r, color="#999", ls=":", lw=1); a.text(r + .03, 8, l, fontsize=7, rotation=90, va="top")
a.set_title("Gradient descent on x^4 + x^3 - 2x^2", fontsize=10); a.legend(frameon=False, fontsize=7, loc="upper center")

b = ax[0, 1]
rng = np.random.default_rng(0); n = 2000
inc = rng.gamma(4, 1, n) + .5; pop = rng.gamma(2, 700, n) + 3; y = .4 * inc + .0002 * pop + .5 + rng.normal(0, .3, n)
X = np.c_[inc, pop]; Xs = (X - X.mean(0)) / X.std(0)
_, _, raw = gradient_descent(X, y, 0.001, 50); _, _, sc = gradient_descent(Xs, y, 0.1, 200)
raw = np.array(raw); finite = raw[np.isfinite(raw)]
b.plot(range(len(finite)), np.log10(finite), color="#c2410c", label=f"unscaled, lr 0.001 (overflows to inf at step {len(raw)-1})")
b.plot(range(len(sc)), np.log10(sc), color="#2a7f62", label="standardised, lr 0.1")
b.axhline(np.log10(0.09), color="#999", ls=":", lw=1); b.text(110, np.log10(0.09) + 0.6, "noise floor (MSE 0.09)", fontsize=7)
b.set_ylim(-1.5, 12); b.text(2, 10.5, "unscaled curve keeps rising to ~1e300, then inf", fontsize=7, color="#c2410c")
b.set_xlabel("iteration"); b.set_ylabel("log10 MSE"); b.set_title("Why feature scaling matters (synthetic data)", fontsize=10); b.legend(frameon=False, fontsize=7)

c = ax[1, 0]; names = list(ens); vals = [ens[k]["mean"] for k in names]
c.barh(names[::-1], vals[::-1], color="#1d4ed8", height=.55); c.set_xlim(.9, 1.0)
for i, v in enumerate(vals[::-1]): c.text(v + .001, i, f"{v:.3f}", va="center", fontsize=8)
c.set_title("Iris, 5-fold CV accuracy: trees vs ensembles", fontsize=10)

d = ax[1, 1]; names = ["BernoulliNB\n(binary)", "MultinomialNB\n(3 levels)", "GaussianNB"]; vals = list(nb.values())
d.bar(names, vals, color=["#9ca3af", "#1d4ed8", "#9ca3af"], width=.55); d.set_ylim(.7, 1.0)
for i, v in enumerate(vals): d.text(i, v + .004, f"{v:.3f}", ha="center", fontsize=8)
d.set_title("8x8 digits (1,797 images): Naive Bayes variants", fontsize=10)
for x in ax.flat: [x.spines[k].set_visible(False) for k in ("top", "right")]
plt.tight_layout(); plt.savefig("assets/overview.png")
