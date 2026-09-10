from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "archive" / "twcs" / "twcs.csv"
CHUNK_SIZE = 100000
TARGET_BRAND = "AppleSupport"


def read_csv_in_chunks(data_path: Path, chunk_size: int = CHUNK_SIZE):
    """Return the CSV as a sequence of manageable DataFrame chunks."""
    if not data_path.exists():
        raise FileNotFoundError(f"CSV file not found: {data_path}")

    try:
        return pd.read_csv(data_path, chunksize=chunk_size)
    except Exception as exc:
        raise RuntimeError(f"Could not read the CSV file: {exc}") from exc


def analyze_brand(data_path: Path = DATA_PATH, brand_name: str = TARGET_BRAND):
    """Analyze one brand account in a memory-safe way using chunked processing."""
    total_brand_tweets = 0
    brand_replies = 0
    brand_non_replies = 0
    tweets_with_response_tweet_id = 0
    tweets_with_in_response_to_tweet_id = 0
    apple_support_sample = []
    customer_sample = []
    customer_response_tweet_ids = set()

    try:
        csv_reader = read_csv_in_chunks(data_path)

        # First pass: collect AppleSupport tweets and some quick counts.
        for chunk in csv_reader:
            apple_rows = chunk[chunk["author_id"].astype(str) == brand_name]

            if apple_rows.empty:
                continue

            total_brand_tweets += len(apple_rows)

            # Count reply vs non-reply support tweets.
            for _, row in apple_rows.iterrows():
                response_tweet_id = row.get("response_tweet_id")
                in_response_to = row.get("in_response_to_tweet_id")

                if pd.notna(response_tweet_id):
                    tweets_with_response_tweet_id += 1
                if pd.notna(in_response_to):
                    tweets_with_in_response_to_tweet_id += 1

                if pd.notna(in_response_to):
                    brand_replies += 1
                else:
                    brand_non_replies += 1

                # Save a small sample of AppleSupport support tweets.
                if len(apple_support_sample) < 20:
                    apple_support_sample.append(
                        {
                            "tweet_id": row.get("tweet_id"),
                            "text": row.get("text"),
                            "created_at": row.get("created_at"),
                            "response_tweet_id": row.get("response_tweet_id"),
                            "in_response_to_tweet_id": row.get("in_response_to_tweet_id"),
                        }
                    )

                # Keep track of any AppleSupport tweet IDs that were used as a response.
                if pd.notna(response_tweet_id):
                    customer_response_tweet_ids.add(str(response_tweet_id))

        # Second pass: collect customer tweets that point to an AppleSupport response.
        csv_reader = read_csv_in_chunks(data_path)
        for chunk in csv_reader:
            for _, row in chunk.iterrows():
                if pd.isna(row.get("response_tweet_id")):
                    continue

                response_tweet_id = str(row["response_tweet_id"])
                if response_tweet_id not in customer_response_tweet_ids:
                    continue

                # This customer tweet is a response to an AppleSupport tweet.
                if len(customer_sample) < 20:
                    customer_sample.append(
                        {
                            "tweet_id": row.get("tweet_id"),
                            "author_id": row.get("author_id"),
                            "text": row.get("text"),
                            "created_at": row.get("created_at"),
                            "response_tweet_id": row.get("response_tweet_id"),
                            "in_response_to_tweet_id": row.get("in_response_to_tweet_id"),
                        }
                    )

    except Exception as exc:
        raise RuntimeError(f"Error while analyzing AppleSupport: {exc}") from exc

    return {
        "brand_name": brand_name,
        "total_brand_tweets": total_brand_tweets,
        "brand_replies": brand_replies,
        "brand_non_replies": brand_non_replies,
        "tweets_with_response_tweet_id": tweets_with_response_tweet_id,
        "tweets_with_in_response_to_tweet_id": tweets_with_in_response_to_tweet_id,
        "apple_support_sample": apple_support_sample,
        "customer_sample": customer_sample,
    }


def print_analysis(results):
    """Print AppleSupport statistics and a small sample of tweets."""
    print("=== AppleSupport analysis ===")
    print(f"Brand account: {results['brand_name']}")
    print(f"Total AppleSupport tweets: {results['total_brand_tweets']}")
    print(f"AppleSupport tweets that are replies: {results['brand_replies']}")
    print(f"AppleSupport tweets that are not replies: {results['brand_non_replies']}")
    print(f"Tweets with response_tweet_id: {results['tweets_with_response_tweet_id']}")
    print(f"Tweets with in_response_to_tweet_id: {results['tweets_with_in_response_to_tweet_id']}")

    print("\nSample AppleSupport tweets (first 20):")
    for index, tweet in enumerate(results["apple_support_sample"], start=1):
        print(f"\n{index}. tweet_id={tweet['tweet_id']}")
        print(f"   created_at={tweet['created_at']}")
        print(f"   response_tweet_id={tweet['response_tweet_id']}")
        print(f"   in_response_to_tweet_id={tweet['in_response_to_tweet_id']}")
        print(f"   text={tweet['text']}")

    print("\nSample customer tweets that point to AppleSupport replies (up to 20):")
    if results["customer_sample"]:
        for index, tweet in enumerate(results["customer_sample"], start=1):
            print(f"\n{index}. tweet_id={tweet['tweet_id']}")
            print(f"   author_id={tweet['author_id']}")
            print(f"   created_at={tweet['created_at']}")
            print(f"   response_tweet_id={tweet['response_tweet_id']}")
            print(f"   in_response_to_tweet_id={tweet['in_response_to_tweet_id']}")
            print(f"   text={tweet['text']}")
    else:
        print("No matching customer tweets were found.")


if __name__ == "__main__":
    try:
        brand_analysis = analyze_brand()
        print_analysis(brand_analysis)
    except Exception as exc:
        print(f"Failed to analyze AppleSupport: {exc}")
        raise
