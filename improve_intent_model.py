
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(
    PROJECT_ROOT / "data" / "tickets" / "01_support_tickets.csv"
)

# Reuse the existing normalization logic for validation groups
import re

def normalize_query(query):
    query = query.lower()
    query = re.sub(r"#tkt\d+", "<ticket>", query)
    query = re.sub(r"\d{1,2}-[a-z]{3}-\d{4}", "<date>", query)
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

print("\nImproved Model — Template-Grouped CV")

for i, score in enumerate(scores, start=1):
    print(f"Fold {i}: {score:.2%}")

print(f"\nAverage Accuracy: {scores.mean():.2%}")
print(f"Standard Deviation: {scores.std():.2%}")



from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

predictions = cross_val_predict(
    model,
    df["query_text"],
    df["category"],
    groups=groups,
    cv=cv
)

print("\nLinearSVC Classification Report:")
print(
    classification_report(
        df["category"],
        predictions,
        zero_division=0
    )
)

print("\nLinearSVC Confusion Matrix:")
print(
    pd.crosstab(
        df["category"],
        predictions,
        rownames=["Actual"],
        colnames=["Predicted"]
    )
)


# Step 5 - Deduplicated LinearSVC Evaluation

from sklearn.base import clone
from sklearn.metrics import accuracy_score
import numpy as np

dedup_scores = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(
        df["query_text"],
        df["category"],
        groups=groups
    ),
    start=1
):
    # Remove duplicate queries from training data only
    train_data = df.iloc[train_idx].drop_duplicates(
        subset=["query_text"]
    )

    test_data = df.iloc[test_idx]

    # Create a fresh model for each fold
    fold_model = clone(model)

    fold_model.fit(
        train_data["query_text"],
        train_data["category"]
    )

    predictions = fold_model.predict(
        test_data["query_text"]
    )

    accuracy = accuracy_score(
        test_data["category"],
        predictions
    )

    dedup_scores.append(accuracy)

    print(f"Fold {fold}: {accuracy:.2%}")

print("\nDeduplicated LinearSVC Results:")
print(f"Average Accuracy: {np.mean(dedup_scores):.2%}")
print(f"Standard Deviation: {np.std(dedup_scores):.2%}")


#Prepare for dataset improvement

print("\nDataset Columns:")
print(df.columns.tolist())

print("\nDataset Category Counts:")
print(df["category"].value_counts())

print("\nUnique Queries by Category:")
print(
    df.groupby("category")["query_text"].nunique()
)

# Dataset improvement
#Review the Account Access training data
