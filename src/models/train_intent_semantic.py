from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "archive" / "processed" / "apple_support_golden.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_labeled_data(data_path: Path = DATA_PATH):
    """Load the labeled golden set and drop review-needed rows."""
    if not data_path.exists():
        raise FileNotFoundError(f"Labeled CSV not found: {data_path}")

    df = pd.read_csv(data_path)
    df = df[df["intent"] != "NEEDS_REVIEW"].copy()

    if df.empty:
        raise ValueError("No labeled examples found in the golden dataset.")

    return df


def main():
    df = load_labeled_data()

    X = df["customer_text"].fillna("")
    y = df["intent"]

    print(f"Number of labeled examples: {len(df)}")

    if len(y.unique()) < 2:
        raise ValueError("At least two intent classes are required for training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Number of training examples: {len(X_train)}")
    print(f"Number of test examples: {len(X_test)}")

    print("Loading sentence-transformers model...")
    encoder = SentenceTransformer(MODEL_NAME)

    print("Generating sentence embeddings...")
    X_train_embeddings = encoder.encode(X_train.tolist(), show_progress_bar=False)
    X_test_embeddings = encoder.encode(X_test.tolist(), show_progress_bar=False)

    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train_embeddings, y_train)

    predictions = classifier.predict(X_test_embeddings)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted precision: {precision:.4f}")
    print(f"Weighted recall: {recall:.4f}")
    print(f"Weighted F1: {f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("\nIncorrect predictions:")
    incorrect_mask = y_test.reset_index(drop=True) != pd.Series(predictions, index=y_test.index).reset_index(drop=True)
    incorrect_indices = incorrect_mask[incorrect_mask].index

    if len(incorrect_indices) == 0:
        print("No incorrect predictions.")
    else:
        for idx in incorrect_indices:
            actual = y_test.iloc[idx]
            predicted = predictions[idx]
            message = X_test.iloc[idx]
            print(f"- Actual: {actual} | Predicted: {predicted} | Message: {message}")


if __name__ == "__main__":
    main()
