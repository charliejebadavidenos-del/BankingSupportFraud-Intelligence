
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

source_file = (
    PROJECT_ROOT
    / "data"
    / "tickets"
    / "01_support_tickets.csv"
)

output_file = (
    PROJECT_ROOT
    / "data"
    / "tickets"
    / "02_improved_support_tickets.csv"
)

df = pd.read_csv(source_file)

# Retain only the columns needed for intent training
improved_df = df[
    ["query_text", "category"]
].copy()

# Remove exact duplicate queries
improved_df = improved_df.drop_duplicates(
    subset=["query_text", "category"]
)

# Reset row numbering
improved_df = improved_df.reset_index(drop=True)

# Save a separate working dataset
improved_df.to_csv(output_file, index=False)

print("\nImproved Dataset Created Successfully!")
print("Original tickets:", len(df))
print("Unique training examples:", len(improved_df))
print("Saved to:", output_file)

print("\nCategory Distribution:")
print(improved_df["category"].value_counts())


# Step 9 - Inspect Account Access queries

account_queries = improved_df[
    improved_df["category"] == "Account Access"
]

print("\nAccount Access Training Queries:")
print("-" * 60)

for index, query in enumerate(
    account_queries["query_text"],
    start=1
):
    print(f"{index}. {query}")

##Expand the intent categories


# Step 12 - Expand banking intent categories

category_updates = {
    "My account is showing negative balance but I didn't overdraw": "Account Services",
    "My salary credit is delayed by 3 days": "Transaction/Payment Issue",
    "I need to close my account but there are pending charges": "Account Services",
    "My fixed deposit matured but amount not credited": "Deposit Issue",
    "I want to add my wife as a joint account holder": "Account Services",
    "My cheque was returned even though I had sufficient balance": "Transaction/Payment Issue",
    "Why was my NEFT transfer delayed?": "Transaction/Payment Issue"
}

improved_df["category"] = improved_df.apply(
    lambda row: category_updates.get(
        row["query_text"],
        row["category"]
    ),
    axis=1
)

# Save updated categories to the separate dataset
improved_df.to_csv(output_file, index=False)

print("\nUpdated Category Distribution:")
print(improved_df["category"].value_counts())

print("\nTotal Categories:")
print(improved_df["category"].nunique())
