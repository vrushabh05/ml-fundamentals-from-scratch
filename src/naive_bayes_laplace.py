"""Hand-written categorical Naive Bayes with and without Laplace smoothing on a synthetic loan dataset.

The labels in this dataset are drawn independently of the features, so there is no signal to learn:
a classifier cannot beat always predicting the majority class (about 0.70).
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)
size = 500
loan = pd.DataFrame({
    "Income(Thousand)": np.random.normal(75, 5, size).round(0),
    "Education": np.random.choice(["High School", "Bachelor", "Masters"], size, p=[0.4, 0.5, 0.1]),
    "CreditHistory": np.random.choice(["Good", "Average", "Poor"], size, p=[0.5, 0.3, 0.2]),
    "LoanApproval": np.random.choice(["Accepted", "Rejected"], size, p=[0.7, 0.3]),
})
loan["IncomeBin"] = pd.qcut(loan["Income(Thousand)"], 4, labels=["Low", "Medium", "High", "Very High"])
train, test = train_test_split(loan, test_size=0.2, stratify=loan["LoanApproval"], random_state=2025)
features, target = ["IncomeBin", "Education", "CreditHistory"], "LoanApproval"

priors = train[target].value_counts(normalize=True).to_dict()


def conditional(smooth):
    table = {}
    for f in features:
        table[f] = {}
        values = train[f].unique()
        for cls in train[target].unique():
            counts = train.loc[train[target] == cls, f].value_counts()
            total = counts.sum()
            if smooth:
                table[f][cls] = {v: (counts.get(v, 0) + 1) / (total + len(values)) for v in values}
            else:
                table[f][cls] = (counts / total).to_dict()
    return table


def predict(row, cond):
    scores = {c: np.log(p) + sum(np.log(cond[f][c].get(row[f], 1e-6)) for f in features) for c, p in priors.items()}
    return max(scores, key=scores.get)


res = {}
for name, smooth in (("without Laplace", False), ("with Laplace", True)):
    cond = conditional(smooth)
    preds = test[features].apply(lambda r: predict(r, cond), axis=1)
    res[name] = round(float((preds == test[target]).mean()), 3)
res["majority-class baseline"] = round(float((test[target] == train[target].mode()[0]).mean()), 3)
print(json.dumps(res, indent=2))
json.dump(res, open("results/naive_bayes_laplace.json", "w"), indent=2)
