from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "archive" / "twcs" / "twcs.csv"
OUTPUT_PATH = PROJECT_ROOT / "archive" / "processed" / "apple_support_pairs.csv"
CHUNK_SIZE = 100000
TARGET_BRAND = "AppleSupport"


def normalize_tweet_id(value):
    """Normalize IDs so numbers and floats that represent the same ID match."""
    if pd.isna(value):
        return None

    value = str(value).strip()
    if value in ("", "nan", "NaN", "None", "none"):
        return None

    return str(int(float(value)))


def read_csv_in_chunks(data_path: Path, chunk_size: int = CHUNK_SIZE):
    """Read the large CSV in chunks to avoid loading entire dataset into memory."""
    if not data_path.exists():
        raise FileNotFoundError(f"CSV file not found: {data_path}")

    try:
        return pd.read_csv(data_path, chunksize=chunk_size)
    except Exception as exc:
        raise RuntimeError(f"Could not read the CSV file: {exc}") from exc


def build_apple_support_index(data_path: Path = DATA_PATH):
    """Create a mapping from customer tweet ID to AppleSupport reply details."""
    support_index = {}

    try:
        csv_reader = read_csv_in_chunks(data_path)
        for chunk in csv_reader:
            for _, row in chunk.iterrows():
                author_id = row.get("author_id")
                if pd.isna(author_id):
                    continue

                if str(author_id) != TARGET_BRAND:
                    continue

                support_tweet_id = normalize_tweet_id(row.get("tweet_id"))
                if support_tweet_id is None:
                    continue

                parent_tweet_id = normalize_tweet_id(row.get("in_response_to_tweet_id"))
                if parent_tweet_id is None:
                    continue

                support_index[parent_tweet_id] = {
                    "tweet_id": support_tweet_id,
                    "author_id": str(author_id),
                    "created_at": row.get("created_at"),
                    "text": row.get("text"),
                    "response_tweet_id": row.get("response_tweet_id"),
                    "in_response_to_tweet_id": parent_tweet_id,
                }
    except Exception as exc:
        raise RuntimeError(f"Error building AppleSupport index: {exc}") from exc

    return support_index


def reconstruct_pairs(data_path: Path = DATA_PATH):
    """Match customer tweets to AppleSupport replies using in_response_to_tweet_id."""
    support_index = build_apple_support_index(data_path)
    pairs = []
    seen_customer_tweets = set()
    seen_support_replies = set()

    try:
        csv_reader = read_csv_in_chunks(data_path)
        for chunk in csv_reader:
            for _, row in chunk.iterrows():
                author_id = row.get("author_id")
                if pd.isna(author_id):
                    continue

                customer_author = str(author_id)
                if customer_author == TARGET_BRAND:
                    continue

                customer_tweet_id = normalize_tweet_id(row.get("tweet_id"))
                if customer_tweet_id is None:
                    continue

                support_match = support_index.get(customer_tweet_id)
                if support_match is None:
                    continue

                support_tweet_id = support_match["tweet_id"]
                pair = {
                    "customer_tweet_id": customer_tweet_id,
                    "customer_text": row.get("text"),
                    "customer_created_at": row.get("created_at"),
                    "support_tweet_id": support_tweet_id,
                    "support_text": support_match["text"],
                    "support_created_at": support_match["created_at"],
                }

                pairs.append(pair)
                seen_customer_tweets.add(customer_tweet_id)
                seen_support_replies.add(support_tweet_id)

    except Exception as exc:
        raise RuntimeError(f"Error reconstructing pairs: {exc}") from exc

    return pairs, len(seen_customer_tweets), len(seen_support_replies)


def save_pairs_to_csv(pairs, output_path: Path = OUTPUT_PATH):
    """Save the reconstructed customer-support pairs to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not pairs:
        empty_df = pd.DataFrame(
            columns=[
                "customer_tweet_id",
                "customer_text",
                "customer_created_at",
                "support_tweet_id",
                "support_text",
                "support_created_at",
            ]
        )
        empty_df.to_csv(output_path, index=False)
        return

    pair_df = pd.DataFrame(pairs)
    pair_df.to_csv(output_path, index=False)


def print_summary(pairs):
    """Print a readable summary of reconstructed customer-support pairs."""
    unique_customers = len({pair["customer_tweet_id"] for pair in pairs})
    unique_support = len({pair["support_tweet_id"] for pair in pairs})

    print("=== AppleSupport conversation reconstruction ===")
    print(f"Number of reconstructed customer-support pairs: {len(pairs)}")
    print(f"Number of unique customer tweets: {unique_customers}")
    print(f"Number of unique AppleSupport replies: {unique_support}")

    print("\nFirst 10 reconstructed pairs:")
    for index, pair in enumerate(pairs[:10], start=1):
        print(f"\nPair {index}:")
        print(f"  customer_tweet_id: {pair['customer_tweet_id']}")
        print(f"  customer_text: {pair['customer_text']}")
        print(f"  customer_created_at: {pair['customer_created_at']}")
        print(f"  support_tweet_id: {pair['support_tweet_id']}")
        print(f"  support_text: {pair['support_text']}")
        print(f"  support_created_at: {pair['support_created_at']}")


if __name__ == "__main__":
    try:
        pairs, unique_customers, unique_support_replies = reconstruct_pairs()
        save_pairs_to_csv(pairs)
        print_summary(pairs)
        print(f"\nCSV saved to: {OUTPUT_PATH}")
    except Exception as exc:
        print(f"Failed to reconstruct AppleSupport conversations: {exc}")
        raise
