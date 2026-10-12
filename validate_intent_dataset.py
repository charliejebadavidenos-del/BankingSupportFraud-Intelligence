
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

dataset_path = (
    PROJECT_ROOT
    / "data"
    / "tickets"
    / "02_improved_support_tickets.csv"
)

df = pd.read_csv(dataset_path)

print("\nDATASET VALIDATION")
print("-" * 50)

print("Total records:", len(df))
print("Total categories:", df["category"].nunique())

# Check missing values
print("\nMissing Values:")
print(df[["query_text", "category"]].isnull().sum())

# Check exact duplicate queries
print("\nDuplicate Queries:")
print(df["query_text"].duplicated().sum())

# Check queries assigned to multiple categories
conflicts = (
    df.groupby("query_text")["category"]
    .nunique()
)

conflicting_queries = conflicts[conflicts > 1]

print("\nConflicting Category Labels:")
print(len(conflicting_queries))

if not conflicting_queries.empty:
    print(
        df[
            df["query_text"].isin(
                conflicting_queries.index
            )
        ][["query_text", "category"]]
    )

# Check category distribution
print("\nCategory Distribution:")
print(df["category"].value_counts())

print("\nValidation Completed")
