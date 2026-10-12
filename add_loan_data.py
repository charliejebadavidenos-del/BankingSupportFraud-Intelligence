
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

new_queries = [
    "What is the status of my personal loan application?",
    "My home loan EMI was deducted twice this month",
    "I want to know the interest rate on my education loan",
    "How can I apply for a vehicle loan?",
    "My loan disbursement has been delayed",
    "I need a statement showing my outstanding loan balance",
    "How do I make a partial prepayment on my home loan?",
    "I want to close my personal loan before the tenure ends",
    "Why was my loan application rejected?",
    "My EMI payment is not reflected in the loan account",
    "How can I change the repayment date for my loan?",
    "I need a loan foreclosure statement",
    "The bank charged an unexpected penalty on my EMI",
    "Can I extend the repayment tenure of my education loan?",
    "My loan interest rate increased without notification",
    "How can I obtain a no-dues certificate after loan closure?",
    "My bank has not released the approved home loan amount"
]

new_data = pd.DataFrame({
    "query_text": new_queries,
    "category": "Loan"
})

updated_df = pd.concat(
    [df, new_data],
    ignore_index=True
)

updated_df = updated_df.drop_duplicates(
    subset=["query_text", "category"]
)

updated_df.to_csv(dataset_path, index=False)

print("\nLoan Dataset Updated")
print("Total examples:", len(updated_df))

print("\nLoan examples:")
print(
    (updated_df["category"] == "Loan").sum()
)

print("\nCategory Distribution:")
print(updated_df["category"].value_counts())


# Step 20 - Identify existing Loan queries

existing_loan_queries = df.loc[
    df["category"] == "Loan",
    "query_text"
]

duplicates = set(new_queries).intersection(
    set(existing_loan_queries)
)

print("\nLoan Queries Already Present:")
for query in sorted(duplicates):
    print("-", query)

print("\nNumber of Existing Matches:", len(duplicates))

