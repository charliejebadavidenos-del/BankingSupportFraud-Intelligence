
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

training_path = (
    project_root / "data" / "tickets"
    / "02_improved_support_tickets.csv"
)

test_path = (
    project_root / "data" / "tickets"
    / "03_independent_test.csv"
)

training = pd.read_csv(training_path)
testing = pd.read_csv(test_path)

def normalize_query(text):
    return " ".join(str(text).lower().split())

training_queries = set(
    training["query_text"].map(normalize_query)
)

test_queries = set(
    testing["query_text"].map(normalize_query)
)

overlap = training_queries.intersection(test_queries)

print("\nDATA LEAKAGE CHECK")
print("-" * 55)
print("Training Records:", len(training))
print("Evaluation Records:", len(testing))
print("Exact Normalized Overlaps:", len(overlap))

if overlap:
    print("\nOVERLAPPING QUERIES:")
    for query in sorted(overlap):
        print("-", query)
else:
    print("\nPASS: No exact normalized query overlap detected.")
