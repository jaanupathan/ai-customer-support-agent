from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "archive" / "twcs" / "twcs.csv"
CHUNK_SIZE = 100000


def read_csv_in_chunks(data_path: Path, chunk_size: int = CHUNK_SIZE):
    """Open the CSV as a generator of smaller DataFrame chunks."""
    if not data_path.exists():
        raise FileNotFoundError(f"CSV file not found: {data_path}")

    try:
        return pd.read_csv(data_path, chunksize=chunk_size)
    except Exception as exc:
        raise RuntimeError(f"Could not read the CSV file: {exc}") from exc


def summarize_dataset(data_path: Path = DATA_PATH, chunk_size: int = CHUNK_SIZE):
    """Read the dataset in chunks and compute the basic statistics we need."""
    total_tweets = 0
    inbound_tweets = 0
    outbound_tweets = 0
    column_names = None
    missing_values = {}
    unique_authors = set()
    tweet_count_by_author = {}

    try:
        csv_reader = read_csv_in_chunks(data_path, chunk_size)

        for chunk in csv_reader:
            if column_names is None:
                column_names = list(chunk.columns)

            total_tweets += len(chunk)

            # The dataset uses a boolean-like inbound column, so we convert it to text and count True/False.
            inbound_series = chunk["inbound"].astype(str).str.lower()
            inbound_tweets += int((inbound_series == "true").sum())
            outbound_tweets += int((inbound_series == "false").sum())

            # Count unique tweet authors without loading the full file into memory.
            author_series = chunk["author_id"].fillna("UNKNOWN").astype(str)
            unique_authors.update(author_series.unique().tolist())

            for author_id, count in author_series.value_counts().items():
                tweet_count_by_author[author_id] = tweet_count_by_author.get(author_id, 0) + int(count)

            # Track missing values by column across all chunks.
            chunk_missing_values = chunk.isna().sum()
            for column_name, missing_count in chunk_missing_values.items():
                missing_values[column_name] = missing_values.get(column_name, 0) + int(missing_count)

    except Exception as exc:
        raise RuntimeError(f"Error while processing the dataset: {exc}") from exc

    return {
        "total_tweets": total_tweets,
        "column_names": column_names,
        "missing_values": missing_values,
        "inbound_tweets": inbound_tweets,
        "outbound_tweets": outbound_tweets,
        "unique_author_count": len(unique_authors),
        "tweet_count_by_author": tweet_count_by_author,
    }


def print_dataset_summary(summary):
    """Print a readable summary of the dataset so it is interview-friendly."""
    print("=== Dataset exploration summary ===")
    print(f"CSV path: {DATA_PATH}")
    print(f"Total tweets: {summary['total_tweets']}")
    print(f"Column names: {summary['column_names']}")

    print("Missing values by column:")
    if summary["missing_values"]:
        for column_name, missing_count in summary["missing_values"].items():
            print(f"  - {column_name}: {missing_count}")
    else:
        print("  - No missing values found.")

    print(f"Inbound tweets: {summary['inbound_tweets']}")
    print(f"Outbound tweets: {summary['outbound_tweets']}")
    print(f"Unique authors: {summary['unique_author_count']}")

    top_authors = sorted(
        summary["tweet_count_by_author"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:15]

    print("Top authors by tweet count:")
    if top_authors:
        for author_id, tweet_count in top_authors:
            print(f"  - {author_id}: {tweet_count} tweets")
    else:
        print("  - No tweet data found.")


if __name__ == "__main__":
    try:
        dataset_summary = summarize_dataset()
        print_dataset_summary(dataset_summary)
    except Exception as exc:
        print(f"Failed to explore dataset: {exc}")
        raise
