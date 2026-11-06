"""Exploratory analysis of the California housing CSV (housing.csv, ~20k districts).

Code from the original assignment. The CSV is not included, so this file has NOT been re-run for this repo;
the findings in the README for this module are the ones recorded in the original notebook.
usage: python src/housing_eda.py path/to/housing.csv
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

sns.set(style="whitegrid")
df = pd.read_csv(sys.argv[1])

plt.figure(figsize=(12, 6)); sns.histplot(df["median_house_value"], bins=50, kde=True)
plt.title("Distribution of Median House Values"); plt.savefig("assets/housing_value_hist.png")

df["ocean_proximity"] = LabelEncoder().fit_transform(df["ocean_proximity"])
corr = df.corr()
plt.figure(figsize=(10, 8)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .6})
plt.title("Correlation Heatmap"); plt.savefig("assets/housing_corr.png")
print(corr["median_house_value"].sort_values(ascending=False))

for x in ("latitude", "median_income"):
    plt.figure(figsize=(12, 6)); sns.scatterplot(x=x, y="median_house_value", data=df, alpha=.5)
    plt.title(f"{x} vs median house value"); plt.savefig(f"assets/housing_{x}.png")
