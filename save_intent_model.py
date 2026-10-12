
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent

dataset_path = (
    PROJECT_ROOT / "data" / "tickets"
    / "02_improved_support_tickets.csv"
)

model_dir = PROJECT_ROOT / "models"
model_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(dataset_path)

X = df["query_text"]
y = df["category"]

model = Pipeline([
    ("features", FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True
        ))
    ])),
    ("classifier", LinearSVC(
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X, y)

model_path = model_dir / "intent_classifier.joblib"
joblib.dump(model, model_path)

print("\nMODEL TRAINING COMPLETED")
print("Training Records:", len(df))
print("Categories:", model.classes_)
print("Model Saved At:", model_path)
