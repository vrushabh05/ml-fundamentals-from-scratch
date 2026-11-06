"""Decision tree vs bagging, random forest, AdaBoost and gradient boosting on Iris, 5-fold CV."""
import json

from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
}
res = {}
for name, m in models.items():
    s = cross_val_score(m, X, y, cv=5, scoring="accuracy")
    res[name] = dict(mean=round(s.mean(), 4), std=round(s.std(), 4), folds=[round(v, 4) for v in s])
    print(f"{name:18s} {s.mean():.4f} +- {s.std():.4f}")
json.dump(res, open("results/ensembles.json", "w"), indent=2)
