import pandas as pd
from pathlib import Path
from src.intent import classify_intent

PROJECT_ROOT = Path(__file__).resolve().parent.parent

file_path = PROJECT_ROOT / "data" / "tickets" / "01_support_tickets.csv"

df = pd.read_csv(file_path)

print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nCustomer queries and categories:")
print(df[["query_text", "category"]].head(10))

df["predicted_intent"] = df["query_text"].apply(classify_intent)

print(df[["query_text", "category", "predicted_intent"]].head(10))


df["correct_prediction"] = (
    df["category"] == df["predicted_intent"]
)

total_queries = len(df)

correct_queries = df["correct_prediction"].sum()

accuracy = (correct_queries / total_queries) * 100

print("Total Queries:", total_queries)
print("Correct Predictions:", correct_queries)
print(f"Accuracy: {accuracy:.2f}%")

print("\nFull query for ticket 6:")
print(df.loc[5, "query_text"])
print("Actual category:", df.loc[5, "category"])
print("Predicted category:", df.loc[5, "predicted_intent"])

incorrect = df[df["correct_prediction"] == False]

print("\nTotal incorrect predictions:", len(incorrect))

print("\nFirst 5 incorrect predictions:")

for _, row in incorrect.head(5).iterrows():
    print("\nCustomer Query:", row["query_text"])
    print("Actual Category:", row["category"])
    print("Predicted Category:", row["predicted_intent"])

# Step 19: Investigate duplicate charge tickets
duplicate_tickets = df[
    df["query_text"].str.contains(
        "duplicate charge",
        case=False,
        na=False
    )
]

print("\nDuplicate charge tickets:")
print(
    duplicate_tickets[
        ["query_text", "category", "sub_category"]
    ].to_string(index=False)
)

df_review = df.copy()

mask = df_review["query_text"].str.contains(
    "duplicate charge",
    case=False,
    na=False
)

df_review.loc[mask, "category"] = "Transaction/Payment Issue"

print("Tickets requiring label review:", mask.sum())

print("\nIncorrect predictions by category:")

print(
    incorrect["category"].value_counts()
)

account_errors = incorrect[
    incorrect["category"] == "Account Access"
]

print("\nAccount Access classification errors:")

print(
    account_errors[
        ["query_text", "predicted_intent"]
    ].head(10).to_string(index=False)
)

print("\nUnique Account Access errors:")

print(
    account_errors["query_text"]
    .value_counts()
    .to_string()
)

print("\nActual dataset categories:")
print(df["category"].value_counts())

print("\nPredicted categories:")
print(df["predicted_intent"].value_counts())


print("\nConfusion Matrix:")

print(
    pd.crosstab(
        df["category"],
        df["predicted_intent"],
        rownames=["Actual"],
        colnames=["Predicted"]
    ).to_string()
)

kyc_errors = incorrect[
    incorrect["category"] == "KYC"
]

print("\nUnique KYC classification errors:")

print(
    kyc_errors["query_text"]
    .value_counts()
    .to_string()
)

fraud_errors = incorrect[
    incorrect["category"] == "Fraud/Unauthorized"
]

print("\nUnique Fraud classification errors:")

print(
    fraud_errors["query_text"]
    .value_counts()
    .to_string()
)

print("\nDataset diversity:")

print("Total tickets:", len(df))

print(
    "Unique customer queries:",
    df["query_text"].nunique()
)

print(
    "Repeated query records:",
    len(df) - df["query_text"].nunique()
)

unique_df = df.drop_duplicates(subset=["query_text"])

unique_correct = (
    unique_df["category"] == unique_df["predicted_intent"]
).sum()

unique_accuracy = (
    unique_correct / len(unique_df)
) * 100

print("\nUnique Query Evaluation:")
print("Total unique queries:", len(unique_df))
print("Correct predictions:", unique_correct)
print(f"Unique query accuracy: {unique_accuracy:.2f}%")


print("\nUnique queries by category:")

print(
    unique_df["category"]
    .value_counts()
    .to_string()
)


label_counts = df.groupby("query_text")["category"].nunique()

conflicting_queries = label_counts[label_counts > 1]

print("\nLabel consistency check:")
print("Queries with conflicting categories:", len(conflicting_queries))

from sklearn.model_selection import GroupShuffleSplit

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

print("\nTraining and Testing Split:")
print("Training tickets:", len(train_df))
print("Testing tickets:", len(test_df))

print("\nTraining category distribution:")
print(train_df["category"].value_counts())

print("\nTesting category distribution:")
print(test_df["category"].value_counts())