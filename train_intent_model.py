import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent

file_path = (
    PROJECT_ROOT
    / "data"
    / "tickets"
    / "01_support_tickets.csv"
)

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Total tickets:", len(df))
print("Columns:", df.columns.tolist())

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        df,
        groups=df["query_text"]
    )
)

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

print("\nTraining tickets:", len(train_df))
print("Testing tickets:", len(test_df))

print("\nOverlapping queries:",
      len(set(train_df["query_text"]) &
          set(test_df["query_text"])))


model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

print("\nNLP pipeline created successfully!")
print(model)


X_train = train_df["query_text"]
y_train = train_df["category"]

model.fit(X_train, y_train)

print("\nModel training completed successfully!")
print("Training samples:", len(X_train))
print("Learned categories:", model.classes_)

## Test the trained model
from sklearn.metrics import accuracy_score

X_test = test_df["query_text"]
y_test = test_df["category"]

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Testing samples:", len(X_test))
print(f"Test Accuracy: {accuracy * 100:.2f}%")

#Generate Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

cm_df = pd.DataFrame(
    cm,
    index=model.classes_,
    columns=model.classes_
)

print("\nConfusion Matrix:")
print(cm_df.to_string())


## Identify misclassifed cx queries

results = test_df[["query_text", "category"]].copy()

results["predicted_category"] = y_pred

errors = results[
    results["category"] != results["predicted_category"]
]

print("\nMisclassified Test Queries:")

print(
    errors.drop_duplicates()
    .to_string(index=False)
)

## Calculate accuracy on unique test queries
unique_results = results.drop_duplicates(
    subset=["query_text"]
)

unique_accuracy = accuracy_score(
    unique_results["category"],
    unique_results["predicted_category"]
)

print("\nUnique Query Evaluation:")
print("Unique test queries:", len(unique_results))
print(
    f"Unique query accuracy: {unique_accuracy * 100:.2f}%"
)

## Reduce duplicatiom

unique_train_df = train_df.drop_duplicates(
    subset=["query_text"]
)

print("\nTraining Data Analysis:")
print("Original training tickets:", len(train_df))
print("Unique training queries:", len(unique_train_df))
print(
    "Duplicate training tickets:",
    len(train_df) - len(unique_train_df)
)

unique_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

unique_model.fit(
    unique_train_df["query_text"],
    unique_train_df["category"]
)

unique_pred = unique_model.predict(X_test)

unique_model_accuracy = accuracy_score(
    y_test,
    unique_pred
)

print("\nDeduplicated Model Evaluation:")
print("Training samples:", len(unique_train_df))
print(f"Test Accuracy: {unique_model_accuracy * 100:.2f}%")

from sklearn.metrics import classification_report

print("\nDeduplicated Model Classification Report:")

print(
    classification_report(
        y_test,
        unique_pred,
        zero_division=0
    )
)

print("\nOriginal Training Category Distribution:")
print(train_df["category"].value_counts())

print("\nUnique Training Category Distribution:")
print(unique_train_df["category"].value_counts())

# Model

balanced_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    ))
])

balanced_model.fit(
    unique_train_df["query_text"],
    unique_train_df["category"]
)

balanced_pred = balanced_model.predict(X_test)

balanced_accuracy = accuracy_score(
    y_test,
    balanced_pred
)

print("\nBalanced Model Evaluation:")
print(f"Test Accuracy: {balanced_accuracy * 100:.2f}%")

print("\nBalanced Model Classification Report:")

print(
    classification_report(
        y_test,
        balanced_pred,
        zero_division=0
    )
)

## identify mistakes of balance model

balanced_results = test_df[
    ["query_text", "category"]
].copy()

balanced_results["predicted_category"] = balanced_pred

balanced_errors = balanced_results[
    balanced_results["category"] !=
    balanced_results["predicted_category"]
]

print("\nBalanced Model Misclassifications:")

print(
    balanced_errors.drop_duplicates()
    .to_string(index=False)
)

## Inspect prediction confidence

error_queries = balanced_errors["query_text"].drop_duplicates()

for query in error_queries:
    probabilities = balanced_model.predict_proba([query])[0]

    print("\nCustomer Query:", query)

    for category, probability in zip(
        balanced_model.classes_,
        probabilities
    ):
        print(f"{category}: {probability:.2%}")


# confidence value

confidence_threshold = 0.60

probabilities = balanced_model.predict_proba(X_test)

max_confidences = probabilities.max(axis=1)

review_count = sum(
    confidence < confidence_threshold
    for confidence in max_confidences
)

print("\nConfidence Threshold Evaluation:")
print("Threshold:", confidence_threshold)
print("Tickets requiring review:", review_count)
print("Tickets automatically classified:", len(X_test) - review_count)


print("\nPrediction Confidence Distribution:")

print(f"Minimum: {max_confidences.min():.2%}")
print(f"Maximum: {max_confidences.max():.2%}")
print(f"Average: {max_confidences.mean():.2%}")
print(f"Median: {pd.Series(max_confidences).median():.2%}")

#Compare different confidence thresholds

print("\nConfidence Threshold Comparison:")

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:

    accepted = max_confidences >= threshold

    accepted_count = accepted.sum()

    coverage = accepted_count / len(X_test)

    if accepted_count > 0:
        accepted_accuracy = accuracy_score(
            y_test[accepted],
            balanced_pred[accepted]
        )

        accuracy_text = f"{accepted_accuracy * 100:.2f}%"
    else:
        accuracy_text = "N/A"

    print(
        f"Threshold: {threshold:.0%} | "
        f"Accepted: {accepted_count} | "
        f"Coverage: {coverage:.2%} | "
        f"Accepted Accuracy: {accuracy_text}"
    )

##Evaluate thresholds using unique queries

unique_test = test_df.drop_duplicates(
    subset=["query_text"]
)

unique_probs = balanced_model.predict_proba(
    unique_test["query_text"]
)

unique_preds = balanced_model.predict(
    unique_test["query_text"]
)

unique_confidences = unique_probs.max(axis=1)

print("\nUnique Query Threshold Evaluation:")

for threshold in [0.30, 0.35, 0.40, 0.45]:
    accepted = unique_confidences >= threshold
    count = accepted.sum()

    if count > 0:
        accuracy = accuracy_score(
            unique_test["category"][accepted],
            unique_preds[accepted]
        )

        print(
            f"Threshold: {threshold:.0%} | "
            f"Accepted: {count}/{len(unique_test)} | "
            f"Accuracy: {accuracy:.2%}"
        )
    else:
        print(f"Threshold: {threshold:.0%} | Accepted: 0")

##KFold Cross Validation

from sklearn.model_selection import StratifiedGroupKFold

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nCross-validation configured successfully!")
print("Number of folds:", cv.n_splits)

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    balanced_model,
    df["query_text"],
    df["category"],
    groups=df["query_text"],
    cv=cv,
    scoring="accuracy"
)

print("\n5-Fold Cross-Validation Results:")

for fold, score in enumerate(cv_scores, start=1):
    print(f"Fold {fold}: {score * 100:.2f}%")

print(f"\nAverage Accuracy: {cv_scores.mean() * 100:.2f}%")
print(f"Standard Deviation: {cv_scores.std() * 100:.2f}%")

##Inspect category distribution in each fold
print("\nCross-Validation Fold Distribution:")

for fold, (train_idx, val_idx) in enumerate(
    cv.split(
        df["query_text"],
        df["category"],
        groups=df["query_text"]
    ),
    start=1
):
    val_df = df.iloc[val_idx]

    print(f"\nFold {fold}:")
    print("Validation tickets:", len(val_df))
    print(
        "Unique queries:",
        val_df["query_text"].nunique()
    )
    print(val_df["category"].value_counts())

## Cross validation with deduplicated training data
from sklearn.base import clone

dedup_cv_scores = []

for fold, (train_idx, val_idx) in enumerate(
    cv.split(
        df["query_text"],
        df["category"],
        groups=df["query_text"]
    ),
    start=1
):
    fold_train = df.iloc[train_idx].drop_duplicates(
        subset=["query_text"]
    )

    fold_val = df.iloc[val_idx]

    fold_model = clone(balanced_model)

    fold_model.fit(
        fold_train["query_text"],
        fold_train["category"]
    )

    predictions = fold_model.predict(
        fold_val["query_text"]
    )

    score = accuracy_score(
        fold_val["category"],
        predictions
    )

    dedup_cv_scores.append(score)

    print(f"Fold {fold}: {score * 100:.2f}%")

##Calculate the deduplicated model's average and stability
import numpy as np

print("\nDeduplicated Cross-Validation Summary:")

print(
    f"Average Accuracy: "
    f"{np.mean(dedup_cv_scores) * 100:.2f}%"
)

print(
    f"Standard Deviation: "
    f"{np.std(dedup_cv_scores) * 100:.2f}%"
)

# Evaluate each banking category across all five folds

from sklearn.model_selection import cross_val_predict

cv_predictions = cross_val_predict(
    balanced_model,
    df["query_text"],
    df["category"],
    groups=df["query_text"],
    cv=cv
)

print("\nCross-Validation Classification Report:")

print(
    classification_report(
        df["category"],
        cv_predictions,
        zero_division=0
    )
)

labels = [
    "Account Access",
    "Fraud/Unauthorized",
    "KYC",
    "Loan"
]

cm = confusion_matrix(
    df["category"],
    cv_predictions,
    labels=labels
)

print("\nCross-Validation Confusion Matrix:")

print(
    pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )
)

error_df = df.copy()

error_df["predicted_category"] = cv_predictions

account_errors = error_df[
    (error_df["category"] == "Account Access")
    & (error_df["predicted_category"] != "Account Access")
]

print("\nMisclassified Account Access Queries:")

print(
    account_errors[
        ["query_text", "predicted_category"]
    ]
    .drop_duplicates()
    .to_string(index=False)
)

print(
    "\nUnique misclassified queries:",
    account_errors["query_text"].nunique()
)

account_queries = df[
    df["category"] == "Account Access"
]["query_text"].drop_duplicates()

print("\nAll Unique Account Access Queries:")

for number, query in enumerate(account_queries, start=1):
    print(f"{number}. {query}")

print("\nTotal unique queries:", len(account_queries))


print("\nDataset Category Summary:")

summary = df.groupby("category").agg(
    total_tickets=("query_text", "count"),
    unique_queries=("query_text", "nunique")
)

summary["duplicate_tickets"] = (
    summary["total_tickets"]
    - summary["unique_queries"]
)

print(summary)

#Inspect the unique KYC queries

kyc_queries = df[
    df["category"] == "KYC"
]["query_text"].drop_duplicates()

print("\nAll Unique KYC Queries:")

for number, query in enumerate(kyc_queries, start=1):
    print(f"{number}. {query}")

print("\nTotal unique KYC queries:", len(kyc_queries))

loan_queries = df[
    df["category"] == "Loan"
]["query_text"].drop_duplicates()

print("\nAll Unique Loan Queries:")

for number, query in enumerate(loan_queries, start=1):
    print(f"{number}. {query}")

print("\nTotal unique Loan queries:", len(loan_queries))

#inspect Fraud and unauthorized queries

fraud_queries = df[
    df["category"] == "Fraud/Unauthorized"
]["query_text"].drop_duplicates()

print("\nAll Unique Fraud Queries:")

for number, query in enumerate(fraud_queries, start=1):
    print(f"{number}. {query}")

print("\nTotal unique Fraud queries:", len(fraud_queries))

#Detect template-based duplicates

import re

def normalize_query(query):
    query = query.lower()

    # Replace ticket numbers
    query = re.sub(r"#tkt\d+", "<ticket>", query)

    # Replace dates such as 25-Apr-2023
    query = re.sub(
        r"\d{1,2}-[a-z]{3}-\d{4}",
        "<date>",
        query
    )

    # Replace currency amounts
    query = re.sub(
        r"₹[\d,]+",
        "<amount>",
        query
    )

    return query


fraud_df = df[
    df["category"] == "Fraud/Unauthorized"
].copy()

fraud_df["normalized_query"] = (
    fraud_df["query_text"].apply(normalize_query)
)

print("\nFraud Query Template Analysis:")
print(
    "Unique original queries:",
    fraud_df["query_text"].nunique()
)
print(
    "Unique normalized templates:",
    fraud_df["normalized_query"].nunique()
)

# Measure template overlap in cross-validation
df["normalized_query"] = df["query_text"].apply(
    normalize_query
)

print("\nCross-Validation Template Overlap:")

for fold, (train_idx, val_idx) in enumerate(
    cv.split(
        df["query_text"],
        df["category"],
        groups=df["query_text"]
    ),
    start=1
):
    train_templates = set(
        df.iloc[train_idx]["normalized_query"]
    )

    val_templates = set(
        df.iloc[val_idx]["normalized_query"]
    )

    overlap = train_templates.intersection(
        val_templates
    )

    print(
        f"Fold {fold}: "
        f"{len(overlap)} overlapping templates"
    )

#Run template-grouped cross-validation
template_cv_scores = cross_val_score(
    balanced_model,
    df["query_text"],
    df["category"],
    groups=df["normalized_query"],
    cv=cv,
    scoring="accuracy"
)

print("\nTemplate-Grouped Cross-Validation:")

for fold, score in enumerate(template_cv_scores, start=1):
    print(f"Fold {fold}: {score * 100:.2f}%")

print(
    f"\nAverage Accuracy: "
    f"{template_cv_scores.mean() * 100:.2f}%"
)

print(
    f"Standard Deviation: "
    f"{template_cv_scores.std() * 100:.2f}%"
)

#Check template-grouped fold distribution
print("\nTemplate-Grouped Fold Distribution:")

for fold, (train_idx, val_idx) in enumerate(
    cv.split(
        df["query_text"],
        df["category"],
        groups=df["normalized_query"]
    ),
    start=1
):
    val_df = df.iloc[val_idx]

    print(f"\nFold {fold}:")
    print("Validation tickets:", len(val_df))

    print(
        val_df["category"].value_counts()
    )

#Find which categories struggle with unseen templates
template_cv_predictions = cross_val_predict(
    balanced_model,
    df["query_text"],
    df["category"],
    groups=df["normalized_query"],
    cv=cv
)

print("\nTemplate-Grouped Classification Report:")

print(
    classification_report(
        df["category"],
        template_cv_predictions,
        zero_division=0
    )
)

#Identify the model's Account Access mistakes
template_errors = df.copy()

template_errors["prediction"] = template_cv_predictions

account_template_errors = template_errors[
    template_errors["category"] == "Account Access"
]

print("\nAccount Access Template-Grouped Predictions:")

print(
    account_template_errors[
        ["query_text", "prediction"]
    ]
    .drop_duplicates()
    .to_string(index=False)
)