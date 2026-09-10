import sys
from pathlib import Path

import ollama
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.search_faiss import MODEL_NAME, load_index_and_data

LLM_MODEL = "llama3.2:1b"


def retrieve_cases(customer_message: str, top_k: int = 3) -> pd.DataFrame:
    """Retrieve historical AppleSupport cases relevant to the customer message."""
    index, data = load_index_and_data()
    print("Loading retrieval model...")
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(
        [customer_message],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    print("Searching historical cases...")
    _, result_indices = index.search(query_embedding, top_k)
    valid_indices = [index_value for index_value in result_indices[0] if index_value >= 0]
    return data.iloc[valid_indices].copy()


def build_prompt(customer_message: str, cases: pd.DataFrame) -> str:
    examples = []
    for number, (_, case) in enumerate(cases.iterrows(), start=1):
        examples.append(
            f"Example {number}\n"
            f"Customer: {case['customer_text']}\n"
            f"AppleSupport reply: {case['support_text']}"
        )

    return (
        "Write a concise, professional customer-support draft for the customer message below.\n"
        "Use the historical AppleSupport examples as grounding and context.\n"
        "Do not invent policies, refunds, guarantees, or actions that are not supported by the examples.\n"
        "When the examples do not establish a specific resolution, acknowledge the issue and suggest only a cautious next step.\n"
        "Return only the drafted reply.\n\n"
        f"Customer message:\n{customer_message}\n\n"
        "Historical AppleSupport examples:\n"
        + "\n\n".join(examples)
    )


def generate_reply(customer_message: str, cases: pd.DataFrame) -> str:
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": build_prompt(customer_message, cases)}],
    )
    return response["message"]["content"].strip()


def main():
    customer_message = input("Enter a customer message: ").strip()
    if not customer_message:
        return

    cases = retrieve_cases(customer_message, top_k=3)
    for number, (_, case) in enumerate(cases.iterrows(), start=1):
        print(f"\nCase {number}")
        print(f"Customer: {case['customer_text']}")
        print(f"AppleSupport reply: {case['support_text']}")

    print("\nGenerating reply with Ollama...")
    reply = generate_reply(customer_message, cases)
    print(reply)


if __name__ == "__main__":
    main()
