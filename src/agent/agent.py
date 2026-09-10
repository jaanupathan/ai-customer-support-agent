import re
import subprocess
import sys
from pathlib import Path

import ollama

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LLM_MODEL = "llama3.2:1b"
TOP_K = 3
RETRIEVAL_THRESHOLD = 0.60
RETRIEVAL_SCRIPT = PROJECT_ROOT / "src" / "retrieval" / "search_faiss.py"

INTENT_KEYWORDS = {
    "device_setup": {"setup", "set up", "activate", "activation", "transfer", "new iphone", "new ipad"},
    "ios_update": {"ios update", "update ios", "update iphone", "update ipad", "upgrade ios"},
    "battery_and_performance": {"battery", "charging", "slow", "overheating", "performance", "lag"},
    "keyboard_and_typing": {"keyboard", "typing", "autocorrect", "keys", "type"},
    "mail_and_messages": {"mail", "email", "message", "imessage", "text", "sms"},
    "account_and_access": {"account", "login", "log in", "sign in", "password", "apple id", "locked out"},
    "network_and_connectivity": {"wifi", "wi-fi", "internet", "bluetooth", "cellular", "connection", "connect"},
    "app_issues": {"app", "application", "crash", "crashing", " download"},
    "payments_and_purchases": {"payment", "purchase", "app store", "subscription", "billing", "charged"},
    "hardware_and_device": {"screen", "display", "camera", "speaker", "broken", "damaged", "device"},
}


# This prototype avoids retraining the baseline classifier at every process start.
def classify_intent(customer_message: str):
    message = customer_message.lower()
    scores = {
        intent: sum(1 for keyword in keywords if keyword in message)
        for intent, keywords in INTENT_KEYWORDS.items()
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_intent, best_score = ranked[0]

    if best_score == 0:
        return "NEEDS_REVIEW", 0.0

    second_score = ranked[1][1]
    if best_score == second_score:
        return "NEEDS_REVIEW", 0.0

    matched_keywords = len(INTENT_KEYWORDS[best_intent])
    confidence = min(0.99, best_score / max(1, matched_keywords))
    return best_intent, confidence


def retrieve_cases(customer_message: str):
    try:
        result = subprocess.run(
            [sys.executable, str(RETRIEVAL_SCRIPT)],
            input=f"{customer_message}\n",
            text=True,
            capture_output=True,
            cwd=PROJECT_ROOT,
            check=False,
        )
    except OSError as error:
        return [], f"Could not start retrieval subprocess: {error}"

    if result.returncode != 0:
        error = result.stderr.strip() or "The retrieval subprocess failed."
        return [], f"Retrieval failed: {error}"

    cases = []
    blocks = result.stdout.split("-" * 80)
    for block in blocks:
        match = re.search(
            r"Similarity score:\s*([0-9.-]+)\s*"
            r"Historical customer message:\s*(.*?)\s*"
            r"Historical AppleSupport reply:\s*(.*)",
            block,
            re.DOTALL,
        )
        if match:
            cases.append(
                {
                    "similarity": float(match.group(1)),
                    "customer_text": match.group(2).strip(),
                    "support_text": match.group(3).strip(),
                }
            )

    if len(cases) != TOP_K:
        return cases, f"Retrieval returned {len(cases)} of {TOP_K} expected cases."
    return cases, ""


def build_prompt(customer_message: str, cases) -> str:
    examples = []
    for number, case in enumerate(cases, start=1):
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


def generate_reply(customer_message: str, cases) -> str:
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": build_prompt(customer_message, cases)}],
    )
    return response["message"]["content"].strip()


def escalation_reason(customer_message: str, intent: str, cases, retrieval_error: str) -> str:
    message = customer_message.lower()
    if retrieval_error:
        return retrieval_error
    if intent == "NEEDS_REVIEW":
        return "The customer message does not map clearly to one intent."
    if not cases or cases[0]["similarity"] < RETRIEVAL_THRESHOLD:
        return "The best historical match has retrieval similarity below 0.60."

    intervention_terms = {
        "refund", "reimburse", "reverse charge", "change password", "reset password",
        "unlock account", "security breach", "stolen", "cancel subscription",
        "payment intervention", "charge dispute",
    }
    if any(term in message for term in intervention_terms):
        return "The request asks for an account, security, payment, refund, or other intervention the prototype cannot safely complete."
    return ""


def main():
    customer_message = input("Enter a customer message: ").strip()
    if not customer_message:
        return

    intent, confidence = classify_intent(customer_message)
    cases, retrieval_error = retrieve_cases(customer_message)
    reply = generate_reply(customer_message, cases)
    reason = escalation_reason(customer_message, intent, cases, retrieval_error)
    decision = "ESCALATE" if reason else "AUTO-HANDLE"

    print(f"\nIntent: {intent}")
    print(f"Intent confidence: {confidence:.2f}")
    print("Top retrieved cases:")
    for number, case in enumerate(cases, start=1):
        print(f"{number}. Similarity: {case['similarity']:.4f}")
        print(f"   Customer: {case['customer_text']}")
        print(f"   AppleSupport reply: {case['support_text']}")
    print(f"\nDraft reply: {reply}")
    print(f"Decision: {decision}")
    print(f"Escalation reason: {reason or 'None'}")


if __name__ == "__main__":
    main()
