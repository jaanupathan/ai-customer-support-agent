from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "archive" / "twcs" / "twcs.csv"
CHUNK_SIZE = 100000

SUPPORT_KEYWORDS = (
    "support",
    "help",
    "care",
    "cares",
    "assist",
    "service",
    "customer",
    "team",
    "contact",
    "reply",
)


def normalize_author_id(author_value):
    """Convert author values into a simple string so we can compare names consistently."""
    if pd.isna(author_value):
        return "UNKNOWN"
    return str(author_value).strip()


def looks_like_support_account(author_value):
    """Heuristic to find likely customer-support brand handles."""
    author_id = normalize_author_id(author_value).lower()
    if author_id == "unknown":
        return False

    return any(keyword in author_id for keyword in SUPPORT_KEYWORDS)


def find_candidate_brands(data_path: Path = DATA_PATH, chunk_size: int = CHUNK_SIZE):
    """Scan the CSV in chunks and identify support accounts and their customer interactions."""

    author_stats = {}

    try:
        # ---------------------------------------------------------
        # PASS 1: Count tweets by author
        # ---------------------------------------------------------
        csv_reader = pd.read_csv(data_path, chunksize=chunk_size)

        for chunk in csv_reader:
            for author_value in chunk["author_id"]:
                author_name = normalize_author_id(author_value)

                stats = author_stats.setdefault(
                    author_name,
                    {
                        "tweets": 0,
                        "inbound": 0,
                        "outbound": 0,
                    },
                )

                stats["tweets"] += 1

        # ---------------------------------------------------------
        # Identify support-like accounts
        # ---------------------------------------------------------
        support_accounts = {
            author_name
            for author_name in author_stats
            if looks_like_support_account(author_name)
        }

        # ---------------------------------------------------------
        # PASS 2: Measure customer/support interactions
        # ---------------------------------------------------------
        interaction_stats = {
            author_name: {
                "support_replies": 0,
                "customer_tweets_replied_to": 0,
            }
            for author_name in support_accounts
        }

        csv_reader = pd.read_csv(data_path, chunksize=chunk_size)

        for chunk in csv_reader:

            # Tweets written by support accounts
            support_tweets = chunk[
                chunk["author_id"].astype(str).isin(support_accounts)
            ]

            for _, row in support_tweets.iterrows():

                author_name = normalize_author_id(row["author_id"])

                # A support tweet with a parent tweet means
                # this is a reply to another tweet.
                parent_tweet_id = row.get("in_response_to_tweet_id")

                if pd.notna(parent_tweet_id):
                    interaction_stats[author_name]["support_replies"] += 1

            # Customer tweets that receive a response from a support account
            for _, row in chunk.iterrows():

                response_tweet_id = row.get("response_tweet_id")

                if pd.isna(response_tweet_id):
                    continue

                response_tweet_id = str(response_tweet_id)

                # The response_tweet_id points to the tweet that
                # answered this customer tweet.
                #
                # We will count it as a customer interaction when
                # the response belongs to one of our support accounts.
                #
                # This requires the response tweet to be found in
                # the dataset, so we collect these IDs below.
        
        # ---------------------------------------------------------
        # PASS 3: Build support tweet ID -> support account mapping
        # ---------------------------------------------------------
        support_tweet_ids = {}

        csv_reader = pd.read_csv(data_path, chunksize=chunk_size)

        for chunk in csv_reader:

            support_rows = chunk[
                chunk["author_id"].astype(str).isin(support_accounts)
            ]

            for _, row in support_rows.iterrows():
                tweet_id = row.get("tweet_id")

                if pd.notna(tweet_id):
                    support_tweet_ids[str(tweet_id)] = normalize_author_id(
                        row["author_id"]
                    )

        # ---------------------------------------------------------
        # PASS 4: Count customer tweets answered by support accounts
        # ---------------------------------------------------------
        csv_reader = pd.read_csv(data_path, chunksize=chunk_size)

        for chunk in csv_reader:

            for _, row in chunk.iterrows():

                response_tweet_id = row.get("response_tweet_id")

                if pd.isna(response_tweet_id):
                    continue

                support_account = support_tweet_ids.get(
                    str(response_tweet_id)
                )

                if support_account is not None:

                    # The current tweet is a customer tweet
                    # that received a support response.
                    interaction_stats[support_account][
                        "customer_tweets_replied_to"
                    ] += 1

    except Exception as exc:
        raise RuntimeError(
            f"Error while finding support-like accounts: {exc}"
        ) from exc

    # -------------------------------------------------------------
    # Build final candidate list
    # -------------------------------------------------------------
    candidate_brands = []

    for author_name in support_accounts:

        stats = author_stats[author_name]
        interactions = interaction_stats[author_name]

        candidate_brands.append(
            {
                "author_id": author_name,
                "tweet_count": stats["tweets"],
                "inbound": stats["inbound"],
                "outbound": stats["outbound"],
                "support_replies": interactions["support_replies"],
                "customer_tweets_replied_to": interactions[
                    "customer_tweets_replied_to"
                ],
            }
        )

    # Rank by actual customer interaction volume,
    # not simply by number of support tweets.
    candidate_brands.sort(
        key=lambda item: item["customer_tweets_replied_to"],
        reverse=True,
    )

    return author_stats, candidate_brands

def print_brand_summary(author_stats, candidate_brands):
    """Print a readable candidate brand summary."""

    print("=== Likely support/brand account discovery ===")
    print(f"Unique authors seen: {len(author_stats)}")
    print(f"Likely support-like accounts found: {len(candidate_brands)}")

    if not candidate_brands:
        print("No brand candidates matched the support-like keyword heuristic.")
        return

    print("Top candidate brands/accounts:")

    for brand in candidate_brands[:20]:
        print(
            f"  - {brand['author_id']}: "
            f"support_tweets={brand['tweet_count']}, "
            f"support_replies={brand['support_replies']}, "
            f"customer_tweets_replied_to={brand['customer_tweets_replied_to']}"
        )


if __name__ == "__main__":
    try:
        author_stats, candidate_brands = find_candidate_brands()
        print_brand_summary(author_stats, candidate_brands)
    except Exception as exc:
        print(f"Failed to identify likely customer-support accounts: {exc}")
        raise
