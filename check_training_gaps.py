
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

target_per_category = 30

counts = df["category"].value_counts()

print("\nTraining Data Requirements")
print("-" * 55)

for category, count in counts.items():
    required = max(0, target_per_category - count)

    print(
        f"{category}: "
        f"Existing={count}, "
        f"Additional Needed={required}"
    )

print("\nTotal Existing Examples:", len(df))
print(
    "Total Additional Examples Needed:",
    sum(max(0, target_per_category - count) for count in counts)
)
