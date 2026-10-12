
import pandas as pd
import numpy as np
import re
from pathlib import Path

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_val_score
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

dataset_path = (
    PROJECT_ROOT
    / "data"
    / "tickets"
    / "02_improved_support_tickets.csv"
)

df = pd.read_csv(dataset_path)

# Group queries that differ only in common variable details
def normalize_query(query):
    query = query.lower().strip()
    query = re.sub(r"#tkt\d+", "<ticket>", query)
    query = re.sub(
        r"\d{1,2}-[a-z]{3}-\d{4}",
        "<date>",
        query
    )
    query = re.sub(r"₹[\d,]+", "<amount>", query)
    return query

groups = df["query_text"].apply(normalize_query)

model = Pipeline([
    ("features", FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True
        ))
    ])),
    ("classifier", LinearSVC(
        class_weight="balanced",
        random_state=42
    ))
])

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    df["query_text"],
    df["category"],
    groups=groups,
    cv=cv,
    scoring="accuracy"
)

print("\nSEVEN-CATEGORY INTENT CLASSIFICATION")
print("-" * 50)

for i, score in enumerate(scores, start=1):
    print(f"Fold {i}: {score:.2%}")

print(f"\nAverage Accuracy: {np.mean(scores):.2%}")
print(f"Standard Deviation: {np.std(scores):.2%}")


# Step 25 - Category-wise model evaluation

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

predictions = cross_val_predict(
    model,
    df["query_text"],
    df["category"],
    groups=groups,
    cv=cv
)

print("\nSEVEN-CATEGORY CLASSIFICATION REPORT")
print("-" * 50)

print(
    classification_report(
        df["category"],
        predictions,
        zero_division=0
    )
)

print("\nCONFUSION MATRIX")
print("-" * 50)

print(
    pd.crosstab(
        df["category"],
        predictions,
        rownames=["Actual"],
        colnames=["Predicted"]
    ).to_string()
)



# Step 26 - Inspect misclassified fraud queries

results = df[["query_text", "category"]].copy()

results["predicted_category"] = predictions

fraud_errors = results[
    (results["category"] == "Fraud/Unauthorized")
    & (results["predicted_category"] != "Fraud/Unauthorized")
]

print("\nMISCLASSIFIED FRAUD QUERIES")
print("-" * 70)

for _, row in fraud_errors.iterrows():
    print("\nCustomer Query:", row["query_text"])
    print("Actual Category:", row["category"])
    print("Predicted Category:", row["predicted_category"])

print("\nTotal Fraud Misclassifications:", len(fraud_errors))
