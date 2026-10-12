
import joblib
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

model_path = project_root / "models" / "intent_classifier.joblib"
test_path = project_root / "data" / "tickets" / "03_independent_test.csv"

model = joblib.load(model_path)
df = pd.read_csv(test_path)

print("\nHOLDOUT CLASSIFICATION SCORE ANALYSIS")
print("-" * 65)

for _, row in df.iterrows():
    query = row["query_text"]
    expected = row["expected_intent"]
    predicted = model.predict([query])[0]

    if predicted != expected:
        scores = model.decision_function([query])[0]

        ranking = sorted(
            zip(model.classes_, scores),
            key=lambda item: item[1],
            reverse=True
        )

        print("\nQuery:", query)
        print("Expected:", expected)
        print("Predicted:", predicted)

        print("Top 3 category scores:")

        for category, score in ranking[:3]:
            print(f"  {category}: {score:.4f}")
