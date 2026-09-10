from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "archive" / "processed" / "apple_support_golden_candidates.csv"
OUTPUT_CSV = PROJECT_ROOT / "archive" / "processed" / "apple_support_golden.csv"

INTENT_MENU = {
    1: "device_setup",
    2: "ios_update",
    3: "battery_and_performance",
    4: "keyboard_and_typing",
    5: "mail_and_messages",
    6: "account_and_access",
    7: "network_and_connectivity",
    8: "app_issues",
    9: "payments_and_purchases",
    10: "hardware_and_device",
}


def load_candidates(csv_path: Path = INPUT_CSV):
    """Load the candidate set and keep the required columns in order."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = [
        "customer_tweet_id",
        "customer_text",
        "candidate_theme",
        "intent",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    return df[required_columns].copy()


def load_progress(output_path: Path = OUTPUT_CSV, input_df: pd.DataFrame = None):
    """Load existing labeled output if present; otherwise start from the candidate CSV."""
    if output_path.exists():
        saved_df = pd.read_csv(output_path)
        if set(saved_df.columns) == {"customer_tweet_id", "customer_text", "candidate_theme", "intent"}:
            # Keep only rows from the source set to avoid stray entries.
            if input_df is not None:
                merged = input_df.merge(
                    saved_df[["customer_tweet_id", "candidate_theme", "intent"]],
                    on=["customer_tweet_id", "candidate_theme"],
                    how="left",
                )
                merged["intent"] = merged["intent_y"].fillna(merged["intent_x"])
                merged = merged[["customer_tweet_id", "customer_text", "candidate_theme", "intent"]]
                return merged
            return saved_df

    return input_df.copy()


def show_intent_menu():
    """Print the intent options shown to the human labeler."""
    print("\nIntent menu:")
    for key, value in INTENT_MENU.items():
        print(f"{key} = {value}")
    print("s = skip / set back to NEEDS_REVIEW")
    print("q = quit and save progress")
    print("r = review / relabel an existing example")


def review_existing_examples(df):
    """Let the human review and relabel already-labeled examples."""
    labeled_rows = df[df["intent"] != "NEEDS_REVIEW"].copy()

    if labeled_rows.empty:
        print("No labeled examples available to review.")
        return

    print("\n=== Review existing labeled examples ===")
    for idx, row in labeled_rows.reset_index(drop=True).iterrows():
        print(f"{idx + 1}. customer_tweet_id={row['customer_tweet_id']}, intent={row['intent']}")

    while True:
        try:
            selection = input("Select an example number to review (or press Enter to cancel): ").strip()
        except EOFError:
            print("Input cancelled.")
            return

        if selection == "":
            return

        if not selection.isdigit():
            print("Please enter a valid example number.")
            continue

        choice_index = int(selection) - 1
        if choice_index < 0 or choice_index >= len(labeled_rows):
            print("Example number out of range.")
            continue

        selected_row = labeled_rows.iloc[choice_index]
        print(f"\nSelected example:")
        print(f"customer_tweet_id: {selected_row['customer_tweet_id']}")
        print(f"customer_text: {selected_row['customer_text']}")
        print(f"current intent: {selected_row['intent']}")

        while True:
            print("\nNew intent selection:")
            show_intent_menu()
            new_value = input("Enter your choice: ").strip()

            if new_value.lower() == "q":
                save_dataframe(df, OUTPUT_CSV)
                print_progress_summary(df)
                print("\nProgress saved. Exiting.")
                raise SystemExit

            if new_value.lower() == "s":
                matched_index = df.index[df["customer_tweet_id"] == selected_row["customer_tweet_id"]].tolist()
                if matched_index:
                    df.at[matched_index[0], "intent"] = "NEEDS_REVIEW"
                    save_dataframe(df, OUTPUT_CSV)
                    print("Example reset to NEEDS_REVIEW.")
                    return
                print("Could not reset the selected example.")
                return

            if new_value.isdigit():
                choice = int(new_value)
                if choice in INTENT_MENU:
                    matched_index = df.index[df["customer_tweet_id"] == selected_row["customer_tweet_id"]].tolist()
                    if matched_index:
                        df.at[matched_index[0], "intent"] = INTENT_MENU[choice]
                        save_dataframe(df, OUTPUT_CSV)
                        print(f"Updated intent to: {INTENT_MENU[choice]}")
                        return
                    print("Could not update the selected example.")
                    return

            print("Invalid input. Please choose a number from 1-10, 's', or 'q'.")


def save_dataframe(df, output_path: Path = OUTPUT_CSV):
    """Save progress after every labeled example."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def print_progress_summary(df):
    """Print the current labeling progress summary."""
    total_examples = len(df)
    labeled = int((df["intent"] != "NEEDS_REVIEW").sum())
    remaining = total_examples - labeled
    count_by_intent = df["intent"].value_counts().sort_index()

    print("\n=== Golden labeling progress ===")
    print(f"Total examples: {total_examples}")
    print(f"Labeled: {labeled}")
    print(f"Remaining: {remaining}")
    print("Intent counts:")
    if count_by_intent.empty:
        print("  No labels yet.")
    else:
        for intent_name, count in count_by_intent.items():
            print(f"  - {intent_name}: {count}")


def main():
    source_df = load_candidates()
    working_df = load_progress(output_path=OUTPUT_CSV, input_df=source_df)

    if "intent" not in working_df.columns:
        working_df["intent"] = "NEEDS_REVIEW"

    # Ensure columns are exactly in the required order.
    working_df = working_df[["customer_tweet_id", "customer_text", "candidate_theme", "intent"]].copy()

    print_progress_summary(working_df)

    for idx in range(len(working_df)):
        row = working_df.iloc[idx]

        if row["intent"] != "NEEDS_REVIEW":
            continue

        print(f"\nExample {idx + 1} / {len(working_df)}")
        print(f"Customer message:\n{row['customer_text']}")
        print(f"Candidate sampling theme: {row['candidate_theme']}")
        show_intent_menu()

        while True:
            user_input = input("Enter your choice: ").strip()

            if user_input.lower() == "r":
                review_existing_examples(working_df)
                print_progress_summary(working_df)
                break

            if user_input.lower() == "q":
                save_dataframe(working_df, OUTPUT_CSV)
                print_progress_summary(working_df)
                print("\nProgress saved. Exiting.")
                return

            if user_input.lower() == "s":
                working_df.at[idx, "intent"] = "NEEDS_REVIEW"
                save_dataframe(working_df, OUTPUT_CSV)
                break

            if user_input.isdigit():
                choice = int(user_input)
                if choice in INTENT_MENU:
                    working_df.at[idx, "intent"] = INTENT_MENU[choice]
                    save_dataframe(working_df, OUTPUT_CSV)
                    break

            print("Invalid input. Please choose a number from 1-10, 's', 'r', or 'q'.")

    save_dataframe(working_df, OUTPUT_CSV)
    print_progress_summary(working_df)
    print("\nAll examples are labeled or skipped. Finished.")


if __name__ == "__main__":
    main()
