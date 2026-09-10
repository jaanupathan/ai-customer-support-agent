from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "archive" / "processed" / "apple_support_golden.csv"


def load_labeled_data(data_path: Path = DATA_PATH):
    """Load the labeled golden set and keep only final, non-review entries."""
    if not data_path.exists():
        raise FileNotFoundError(f"Labeled CSV not found: {data_path}")

    df = pd.read_csv(data_path)
    df = df[df["intent"] != "NEEDS_REVIEW"].copy()

    if df.empty:
        raise ValueError("No labeled examples found in the golden dataset.")

    return df


def build_model():
    """Create the simple TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def main():
    df = load_labeled_data()

    X = df["customer_text"].fillna("")
    y = df["intent"]

    labeled_count = len(df)
    print(f"Number of labeled examples: {labeled_count}")
    print("Intent distribution:")
    print(y.value_counts().sort_index())

    if len(y.unique()) < 2:
        raise ValueError("At least two intent classes are required for train/test splitting.")

    class_counts = y.value_counts()
    min_class_count = class_counts.min()

    if min_class_count >= 2:
        stratify_value = y
        print("Using stratify=y because every intent has at least 2 examples.")
    else:
        stratify_value = None
        print(
            "Not using stratify=y because at least one intent has fewer than 2 examples. "
            "This avoids the train_test_split error while keeping all labels and examples."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_value,
    )

    print(f"Number of training examples: {len(X_train)}")
    print(f"Number of test examples: {len(X_test)}")

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

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
