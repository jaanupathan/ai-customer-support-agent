from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_ROOT / "archive" / "processed" / "apple_support.faiss"
RETRIEVAL_DATA_PATH = PROJECT_ROOT / "archive" / "processed" / "apple_support_retrieval.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_index_and_data():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")
    if not RETRIEVAL_DATA_PATH.exists():
        raise FileNotFoundError(f"Retrieval data not found: {RETRIEVAL_DATA_PATH}")

    index = faiss.read_index(str(INDEX_PATH))
    data = pd.read_csv(RETRIEVAL_DATA_PATH)
    return index, data


def main():
    index, data = load_index_and_data()
    model = SentenceTransformer(MODEL_NAME)

    query = input("Enter a customer message: ")
    if not query.strip():
        print("No message entered.")
        return

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    distances, result_indices = index.search(query_embedding, 3)

    for similarity, row_idx in zip(distances[0], result_indices[0]):
        if row_idx < 0:
            continue

        row = data.iloc[row_idx]
        print(f"Similarity score: {float(similarity):.4f}")
        print(f"Historical customer message: {row['customer_text']}")
        print(f"Historical AppleSupport reply: {row['support_text']}")
        print("-" * 80)


if __name__ == "__main__":
    main()
