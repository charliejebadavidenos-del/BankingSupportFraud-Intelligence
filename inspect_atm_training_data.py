
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

atm_queries = df[
    df["query_text"].str.contains(
        r"ATM|withdrawal|withdraw",
        case=False,
        na=False,
        regex=True
    )
]

print("\nATM-RELATED TRAINING EXAMPLES")
print("-" * 65)

for _, row in atm_queries.iterrows():
    print("\nQuery:", row["query_text"])
    print("Category:", row["category"])

print("\nTOTAL ATM-RELATED EXAMPLES:", len(atm_queries))

print("\nCATEGORY DISTRIBUTION")
print(atm_queries["category"].value_counts())
