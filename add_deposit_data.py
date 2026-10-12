
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
    "My fixed deposit maturity amount has not been credited",
    "I want to renew my fixed deposit for another year",
    "The interest credited on my FD is less than expected",
    "How can I close my fixed deposit before maturity?",
    "My recurring deposit installment was not processed",
    "I need a statement for my recurring deposit account",
    "Why was a penalty charged on my premature FD withdrawal?",
    "My fixed deposit receipt is missing",
    "I want to change the maturity instructions for my FD",
    "The bank renewed my fixed deposit without my request",
    "My recurring deposit maturity payment is delayed",
    "How is interest calculated on my fixed deposit?",
    "I want to open a tax-saving fixed deposit",
    "The maturity date shown for my deposit is incorrect",
    "My FD interest payout has not arrived",
    "I need to update the nominee on my fixed deposit",
    "How can I transfer my fixed deposit to another branch?",
    "My recurring deposit account was closed unexpectedly",
    "I want to increase my recurring deposit installment",
    "Why is the interest rate on my FD different from the advertised rate?",
    "I need a duplicate fixed deposit certificate",
    "My deposit renewal request is still pending",
    "Can I withdraw part of my fixed deposit?",
    "My fixed deposit was prematurely closed without my instruction",
    "The principal amount credited after FD maturity is incorrect",
    "I want to change the interest payout frequency on my deposit",
    "My recurring deposit auto-debit failed",
    "How do I stop automatic renewal of my fixed deposit?",
    "The bank has not issued confirmation for my new fixed deposit"
]

new_data = pd.DataFrame({
    "query_text": new_queries,
    "category": "Deposit Issue"
})

updated_df = pd.concat(
    [df, new_data],
    ignore_index=True
)

updated_df = updated_df.drop_duplicates(
    subset=["query_text", "category"]
)

updated_df.to_csv(dataset_path, index=False)

print("\nDeposit Issue Dataset Updated")
print("Total examples:", len(updated_df))

print("\nDeposit Issue examples:")
print(
    (updated_df["category"] == "Deposit Issue").sum()
)

print("\nCategory Distribution:")
print(updated_df["category"].value_counts())
