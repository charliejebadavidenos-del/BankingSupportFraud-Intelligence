
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

output_path = (
    project_root / "data" / "tickets"
    / "03_independent_test.csv"
)

columns = [
    "query_text",
    "expected_intent",
    "expected_escalation"
]

df = pd.DataFrame(columns=columns)
df.to_csv(output_path, index=False)

print("Independent test template created")
print("File:", output_path)
print("Columns:", list(df.columns))
