from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "archive" / "processed" / "apple_support_pairs.csv"
INDEX_PATH = PROJECT_ROOT / "archive" / "processed" / "apple_support.faiss"
RETRIEVAL_DATA_PATH = PROJECT_ROOT / "archive" / "processed" / "apple_support_retrieval.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_pairs(csv_path: Path = DATA_PATH):
    """Load the processed AppleSupport pairs data."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Pairs CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def build_index(df: pd.DataFrame):
    """Create a FAISS IndexFlatIP index from normalized sentence embeddings."""
    model = SentenceTransformer(MODEL_NAME)

    texts = df["customer_text"].fillna("").astype(str).tolist()

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, dimension


def main():
    df = load_pairs()
    df = df.sample(n=min(10_000, len(df)), random_state=42).reset_index(drop=True)
    index, dimension = build_index(df)

    index_dir = INDEX_PATH.parent
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))

    # Keep the retrieval data simple and focused on the text pairs.
    retrieval_df = df[["customer_tweet_id", "customer_text", "support_tweet_id", "support_text"]].copy()
    retrieval_df.to_csv(RETRIEVAL_DATA_PATH, index=False)

    print(f"Number of indexed examples: {len(df)}")
    print(f"Embedding dimension: {dimension}")
    print(f"FAISS index saved to: {INDEX_PATH}")
    print(f"Retrieval data saved to: {RETRIEVAL_DATA_PATH}")


if __name__ == "__main__":
    main()
