
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
    "What is the daily ATM cash withdrawal limit?",
    "How much money can I withdraw from an ATM per day?",
    "Can I increase my ATM withdrawal limit?",
    "What is the maximum cash withdrawal allowed at an ATM?",
    "How do I change my ATM withdrawal limit?",
    "What are the charges for ATM withdrawals?",
    "How many free ATM withdrawals are allowed each month?",
    "Is there a limit on international ATM withdrawals?",
    "Can I reduce my daily cash withdrawal limit?",
    "Where can I check my ATM transaction limits?"
]

new_data = pd.DataFrame({
    "query_text": new_queries,
    "category": ["Account Services"] * len(new_queries)
})

existing_queries = set(
    df["query_text"].str.lower().str.strip()
)

new_data = new_data[
    ~new_data["query_text"].str.lower().str.strip().isin(existing_queries)
]

updated_df = pd.concat(
    [df, new_data],
    ignore_index=True
)

updated_df.to_csv(dataset_path, index=False)

print("\nATM SERVICE DATA ADDED")
print("-" * 45)
print("New Records:", len(new_data))
print("Total Records:", len(updated_df))
print("\nCATEGORY DISTRIBUTION")
print(updated_df["category"].value_counts())
