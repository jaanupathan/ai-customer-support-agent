import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.agent.agent import (
    classify_intent,
    escalation_reason,
    generate_reply,
    retrieve_cases,
)


st.set_page_config(page_title="AppleSupport AI Customer Support Agent")
st.title("AppleSupport AI Customer Support Agent")
st.caption("A local, retrieval-grounded prototype for drafting AppleSupport replies.")

customer_message = st.text_area("Enter customer message")

if st.button("Analyze Message"):
    if not customer_message.strip():
        st.warning("Please enter a customer message.")
    else:
        with st.spinner("Analyzing message..."):
            intent, confidence = classify_intent(customer_message)
            cases, retrieval_error = retrieve_cases(customer_message)
            reply = generate_reply(customer_message, cases)
            reason = escalation_reason(customer_message, intent, cases, retrieval_error)
            decision = "ESCALATE" if reason else "AUTO-HANDLE"

        st.subheader("Analysis")
        metric_columns = st.columns(2)
        metric_columns[0].metric("Intent", intent)
        metric_columns[1].metric("Confidence", f"{confidence:.2f}")

        st.subheader("Top 3 historical cases")
        for number, case in enumerate(cases, start=1):
            with st.container(border=True):
                st.write(f"**Case {number}** | Similarity: `{case['similarity']:.4f}`")
                st.write(f"Customer: {case['customer_text']}")
                st.write(f"AppleSupport reply: {case['support_text']}")

        st.subheader("Generated reply")
        st.info(reply)

        st.subheader("Routing")
        st.write(f"**Decision:** `{decision}`")
        st.write(f"**Escalation reason:** {reason or 'None'}")
