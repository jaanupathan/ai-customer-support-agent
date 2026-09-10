from pathlib import Path
import re

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "archive" / "processed" / "apple_support_pairs.csv"
OUTPUT_CSV = PROJECT_ROOT / "archive" / "processed" / "apple_support_golden_candidates.csv"

THEMES = {
    "device_setup": {
        "keywords": [
            "setup", "set up", "install", "device", "phone", "iphone", "new phone",
            "activation", "restore", "sim", "configuration", "configure"
        ],
        "description": "Device setup, activation, restore, and configuration problems",
    },
    "ios_update": {
        "keywords": [
            "ios", "update", "upgrade", "system update", "software update", "latest update",
            "11 1", "11.1", "bug after update", "after update"
        ],
        "description": "Issues after iOS version changes or software updates",
    },
    "battery_and_performance": {
        "keywords": [
            "battery", "drain", "slow", "lag", "performance", "overheating", "heat",
            "crashes", "freeze", "stuck", "hanging"
        ],
        "description": "Battery drain, performance degradation, lag, and freezes",
    },
    "keyboard_and_typing": {
        "keywords": [
            "keyboard", "typing", "type", "autocorrect", "text", "letter", "caps", "input",
            "spell", "emoji", "keypad"
        ],
        "description": "Typing, keyboard, text input, autocorrect, and keyboard glitches",
    },
    "mail_and_messages": {
        "keywords": [
            "mail", "email", "messages", "message", "texting", "imessage", "sms",
            "notification", "notifications", "inbox"
        ],
        "description": "Mail, messaging, notifications, and texting issues",
    },
    "account_and_access": {
        "keywords": [
            "login", "log in", "password", "account", "icloud", "apple id", "verification",
            "code", "two factor", "sign in", "access"
        ],
        "description": "Account access, login, authentication, and verification problems",
    },
    "network_and_connectivity": {
        "keywords": [
            "wifi", "wi fi", "signal", "network", "cellular", "connection", "offline",
            "connect", "lost signal", "no service"
        ],
        "description": "Network, connectivity, signal strength, and connection issues",
    },
    "app_issues": {
        "keywords": [
            "app", "apps", "error", "bug", "crash", "not working", "freezing", "stuck",
            "glitch", "malfunction"
        ],
        "description": "Application bugs, crashes, and app-specific issues",
    },
    "payments_and_purchases": {
        "keywords": [
            "payment", "payments", "charge", "charged", "billing", "purchase", "purchases",
            "refund", "apple pay", "itunes", "store"
        ],
        "description": "Payment, billing, purchases, and refund problems",
    },
    "hardware_and_device": {
        "keywords": [
            "screen", "speaker", "camera", "button", "hardware", "damaged", "charger",
            "headphone", "port", "battery replacement", "broken"
        ],
        "description": "Hardware or physical device defects and accessories problems",
    },
}


def clean_customer_text(text):
    """Clean customer text for sampling and review, without modifying the source CSV."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_low_value_message(text):
    """Reject generic or obviously low-value messages before sampling."""
    cleaned = clean_customer_text(text)
    if not cleaned:
        return True

    generic_phrases = [
        "thanks",
        "thank you",
        "hello",
        "hi",
        "hey",
        "good morning",
        "good evening",
        "sorry",
        "help me",
        "please help",
        "urgent",
        "not working",
        "bad",
        "problem",
    ]

    if cleaned in {"", "thanks", "thank you", "hello", "hi", "hey"}:
        return True

    if any(phrase in cleaned for phrase in generic_phrases) and len(cleaned.split()) <= 4:
        return True

    return False


def score_theme_match(text, keywords):
    """Return a simple word-match score for a theme."""
    cleaned = clean_customer_text(text)
    if not cleaned:
        return 0

    score = 0
    for keyword in keywords:
        if keyword in cleaned:
            score += 1
    return score


def load_pairs(csv_path: Path = INPUT_CSV):
    """Load the processed AppleSupport pairs CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def build_candidate_sample(df, target_per_theme=20):
    """Build a deterministic sample of candidate messages for human review."""
    all_rows = []

    for theme_name, config in THEMES.items():
        keyword_list = config["keywords"]
        theme_candidates = []

        for _, row in df.iterrows():
            text = row.get("customer_text", "")
            if pd.isna(text):
                continue

            if is_low_value_message(text):
                continue

            if score_theme_match(text, keyword_list) == 0:
                continue

            # Keep only the first occurrence of duplicates.
            customer_tweet_id = row.get("customer_tweet_id")
            if pd.isna(customer_tweet_id):
                continue

            duplicate_key = str(customer_tweet_id)
            theme_candidates.append({
                "customer_tweet_id": duplicate_key,
                "customer_text": str(text),
                "candidate_theme": theme_name,
            })

        # Deterministic sampling; keep diversity in the best effort sense.
        sampled = pd.DataFrame(theme_candidates).drop_duplicates(subset=["customer_text"]).sample(
            n=min(target_per_theme, len(theme_candidates)),
            random_state=42,
        )

        all_rows.extend(
            {
                "customer_tweet_id": row["customer_tweet_id"],
                "customer_text": row["customer_text"],
                "intent": "NEEDS_REVIEW",
                "candidate_theme": row["candidate_theme"],
            }
            for _, row in sampled.iterrows()
        )

    result_df = pd.DataFrame(all_rows, columns=[
        "customer_tweet_id",
        "customer_text",
        "intent",
        "candidate_theme",
    ])

    result_df = result_df.dropna(subset=["customer_text"]).reset_index(drop=True)
    return result_df


def print_summary(result_df):
    """Print the final golden candidate summary."""
    print("=== Golden candidate sample ===")
    print(f"Total candidates: {len(result_df)}")
    print("Candidates per candidate theme:")

    counts = result_df["candidate_theme"].value_counts().sort_index()
    for theme_name, count in counts.items():
        print(f"  - {theme_name}: {count}")

    print("\nExamples for each theme:")
    for theme_name in THEMES:
        theme_rows = result_df[result_df["candidate_theme"] == theme_name].head(5)
        print(f"\n--- {theme_name} ---")
        if theme_rows.empty:
            print("  (no examples found)")
            continue

        for index, row in enumerate(theme_rows.itertuples(index=False), start=1):
            text = str(row.customer_text)
            if len(text) > 160:
                text = text[:157] + "..."
            print(f"  {index}. {text}")


def main():
    df = load_pairs()
    candidate_df = build_candidate_sample(df)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    candidate_df.to_csv(OUTPUT_CSV, index=False)
    print_summary(candidate_df)
    print(f"\nSaved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
