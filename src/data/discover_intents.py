from pathlib import Path
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "archive" / "processed" / "apple_support_pairs.csv"


def load_pairs(csv_path: Path = INPUT_CSV):
    """Load the processed customer-support pairs CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def clean_customer_text(text):
    """Clean customer text for analysis without changing the source CSV."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def print_basic_counts(df):
    """Print the main dataset counts."""
    print("=== Dataset summary ===")
    print(f"Total number of pairs: {len(df)}")
    print(f"Number of unique customer messages: {df['customer_tweet_id'].nunique()}")
    print(f"Number of unique support replies: {df['support_tweet_id'].nunique()}")


def get_top_tfidf_terms(df, top_n=100):
    """Compute top unigrams and bigrams from customer text using TF-IDF."""
    cleaned_texts = df["customer_text"].fillna("").apply(clean_customer_text)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_features=5000,
    )

    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    ranked_terms = sorted(zip(feature_names, scores), key=lambda item: item[1], reverse=True)

    return ranked_terms[:top_n]


def print_top_terms(top_terms):
    """Print the top 100 TF-IDF terms/phrases."""
    print("\n=== Top TF-IDF terms and phrases ===")
    for rank, (term, score) in enumerate(top_terms, start=1):
        print(f"{rank:>3}. {term:<25} score={score:.6f}")


def print_representative_messages(df, sample_size=50):
    """Print a deterministic sample of customer messages."""
    sample_df = df[["customer_text"]].dropna().copy()
    sample_df = sample_df.sample(n=min(sample_size, len(sample_df)), random_state=42)

    print(f"\n=== Representative customer messages ({len(sample_df)}) ===")
    for index, row in enumerate(sample_df.itertuples(index=False), start=1):
        text = str(row.customer_text)
        if len(text) > 180:
            text = text[:177] + "..."
        print(f"{index:>2}. {text}")


def build_candidate_intents():
    """Return a small set of candidate intent themes to inspect in the data."""
    return [
        {
            "intent_name": "device_setup",
            "description": "Problems setting up, updating, or configuring an Apple device.",
            "keywords": ["setup", "update", "install", "ios", "device", "new phone"],
        },
        {
            "intent_name": "battery_and_performance",
            "description": "Issues about battery drain, lag, overheating, or poor performance.",
            "keywords": ["battery", "drain", "slow", "lag", "performance", "overheating"],
        },
        {
            "intent_name": "keyboard_and_typing",
            "description": "Typing, keyboard, autocorrect, or text input issues.",
            "keywords": ["keyboard", "typing", "text", "autocorrect", "spell", "input"],
        },
        {
            "intent_name": "mail_and_messages",
            "description": "Problems with mail, message apps, notifications, or communication features.",
            "keywords": ["mail", "message", "messages", "texting", "imessage", "notification"],
        },
        {
            "intent_name": "login_and_account_access",
            "description": "Login, account, password, or authentication problems.",
            "keywords": ["login", "password", "account", "code", "verify", "sign in"],
        },
        {
            "intent_name": "network_and_connectivity",
            "description": "Connectivity, Wi-Fi, cellular, or signal issues.",
            "keywords": ["wifi", "signal", "network", "cellular", "connection", "offline"],
        },
        {
            "intent_name": "app_crashes_and_errors",
            "description": "Crashes, freezes, errors, or app malfunction problems.",
            "keywords": ["crash", "freeze", "error", "bug", "app", "stuck"],
        },
        {
            "intent_name": "apple_pay_and_billing",
            "description": "Billing, payment, app store, or purchase-related concerns.",
            "keywords": ["payment", "charge", "billing", "purchase", "apple pay", "refund"],
        },
        {
            "intent_name": "hardware_issue",
            "description": "Potential device hardware or physical device issues.",
            "keywords": ["screen", "speaker", "camera", "button", "hardware", "damaged"],
        },
        {
            "intent_name": "data_and_storage",
            "description": "Backup, storage, missing files, or data-related trouble.",
            "keywords": ["storage", "backup", "photos", "data", "memory", "lost"],
        },
    ]


def describe_theme(df, theme):
    """Find three representative customer messages matching a theme by keyword."""
    cleaned = df["customer_text"].fillna("").apply(clean_customer_text)
    keywords = [word.lower() for word in theme["keywords"]]
    matches = []

    for text in cleaned:
        text_lower = str(text).lower()
        if any(keyword in text_lower for keyword in keywords):
            matches.append(text)

    # keep unique examples in order, then cap at 3
    seen = set()
    examples = []
    for text in matches:
        if text in seen:
            continue
        seen.add(text)
        examples.append(text)
        if len(examples) == 3:
            break

    return examples


def print_candidate_intents(df):
    """Print candidate intent themes and examples."""
    print("\n=== Candidate intent themes ===")
    candidate_themes = build_candidate_intents()

    for theme in candidate_themes:
        print(f"\nIntent: {theme['intent_name']}")
        print(f"Description: {theme['description']}")
        print(f"Keywords: {', '.join(theme['keywords'])}")
        examples = describe_theme(df, theme)
        if examples:
            print("Representative customer messages:")
            for idx, example in enumerate(examples, start=1):
                print(f"  {idx}. {example}")
        else:
            print("Representative customer messages: none found from the keyword signal")


def main():
    df = load_pairs()
    print_basic_counts(df)

    top_terms = get_top_tfidf_terms(df)
    print_top_terms(top_terms)

    print_representative_messages(df)
    print_candidate_intents(df)


if __name__ == "__main__":
    main()
