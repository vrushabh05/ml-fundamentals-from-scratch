"""K-nearest neighbours written with NumPy, compared with scikit-learn on Iris (70/30 stratified split, seed 42)."""
import json
from collections import Counter
from time import perf_counter

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier


class MyKNN:
    def __init__(self, n_neighbors=5, weights="uniform"):
        assert isinstance(n_neighbors, int) and n_neighbors >= 1
        assert weights in ("uniform", "distance")
        self.n_neighbors, self.weights = n_neighbors, weights

    def fit(self, X, y):
        self._X, self._y = np.asarray(X, float), np.asarray(y)
        return self

    @staticmethod
    def _euclidean(A, B):  # vectorised pairwise distances via |a|^2 + |b|^2 - 2ab
        d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.sqrt(np.maximum(d2, 0))

    def predict(self, X):
        dists = self._euclidean(np.asarray(X, float), self._X)
        preds = np.empty(len(dists), dtype=self._y.dtype)
        for i, d in enumerate(dists):
            idx = np.argpartition(d, self.n_neighbors)[: self.n_neighbors]
            labels = self._y[idx]
            if self.weights == "uniform":
                votes = Counter(labels)
            else:
                votes = {}
                for lab, w in zip(labels, 1.0 / (d[idx] + 1e-9)):
                    votes[lab] = votes.get(lab, 0.0) + w
            top = max(votes.values())
            preds[i] = min(lab for lab, v in votes.items() if np.isclose(v, top))  # ties -> smallest label
        return preds


def timed(clf, Xtr, ytr, Xte):
    t = perf_counter(); clf.fit(Xtr, ytr); fit = perf_counter() - t
    t = perf_counter(); pred = clf.predict(Xte); pr = perf_counter() - t
    return pred, fit * 1000, pr * 1000


if __name__ == "__main__":
    np.random.seed(42)
    X, y = datasets.load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    out = {}
    for w in ("uniform", "distance"):
        for name, clf in (("MyKNN", MyKNN(5, w)), ("sklearn", KNeighborsClassifier(5, weights=w))):
            pred, fit_ms, pred_ms = timed(clf, Xtr, ytr, Xte)
            out[f"{name}-{w}"] = dict(accuracy=round(accuracy_score(yte, pred), 4), fit_ms=round(fit_ms, 2), predict_ms=round(pred_ms, 2),
                                      confusion=confusion_matrix(yte, pred).tolist())
            print(f"{name:8s} {w:9s} acc {out[f'{name}-{w}']['accuracy']:.4f}  fit {fit_ms:.2f} ms  predict {pred_ms:.2f} ms")
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    mine = np.array([accuracy_score(y[te], MyKNN(5).fit(X[tr], y[tr]).predict(X[te])) for tr, te in cv.split(X, y)])
    sk = cross_val_score(KNeighborsClassifier(5), X, y, cv=cv)
    out["cv"] = dict(MyKNN=[round(mine.mean(), 4), round(mine.std(), 4)], sklearn=[round(sk.mean(), 4), round(sk.std(), 4)])
    print("5-fold CV  MyKNN %.4f +- %.4f | sklearn %.4f +- %.4f" % (mine.mean(), mine.std(), sk.mean(), sk.std()))
    json.dump(out, open("results/knn.json", "w"), indent=2)
