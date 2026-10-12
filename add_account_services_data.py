
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
    "How do I close my savings account?",
    "I want to convert my savings account into a salary account",
    "Please explain the procedure to add a joint account holder",
    "I need to remove a joint holder from my bank account",
    "How can I update the nominee for my savings account?",
    "I want to change my registered mobile number",
    "Please update my email address linked to my bank account",
    "I need to change my residential address in bank records",
    "How can I transfer my account to another branch?",
    "I want to reactivate my dormant savings account",
    "What documents are required to close a current account?",
    "I want to convert my minor account into a regular account",
    "How can I change the operating instructions for my joint account?",
    "Please help me remove an outdated nominee",
    "I need to update my name after marriage",
    "How do I request an account closure certificate?",
    "I want to change my account type",
    "Can I add a second account holder to my existing account?",
    "How can I update my communication preferences?",
    "I need a bank account statement for the previous financial year",
    "How do I request a new passbook?",
    "I want to change my home branch",
    "Please explain how to register a nominee",
    "How do I update my signature in bank records?",
    "I want to link my savings account to another customer profile",
    "How can I change the mode of operation of my account?",
    "I need assistance transferring my account to another city"
]

new_data = pd.DataFrame({
    "query_text": new_queries,
    "category": "Account Services"
})

updated_df = pd.concat(
    [df, new_data],
    ignore_index=True
)

updated_df = updated_df.drop_duplicates(
    subset=["query_text", "category"]
)

updated_df.to_csv(dataset_path, index=False)

print("\nAccount Services Dataset Updated")
print("Total examples:", len(updated_df))

print("\nAccount Services examples:")
print(
    (updated_df["category"] == "Account Services").sum()
)

print("\nCategory Distribution:")
print(updated_df["category"].value_counts())
