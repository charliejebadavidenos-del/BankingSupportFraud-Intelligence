
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent

output_path = (
    project_root / "data" / "tickets"
    / "05_final_holdout.csv"
)

if output_path.exists():
    print("File already exists. No changes made.")
else:
    columns = [
        "query_text",
        "expected_intent",
        "expected_escalation"
    ]

    pd.DataFrame(columns=columns).to_csv(
        output_path,
        index=False
    )

    print("Final holdout template created successfully.")
    print("File:", output_path)
    print("Columns:", columns)
