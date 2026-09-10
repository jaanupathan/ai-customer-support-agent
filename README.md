# AppleSupport AI Customer Support Agent

An interview-oriented customer-support prototype that combines intent classification, semantic retrieval, and locally generated support replies. It uses reconstructed AppleSupport conversations from the Twitter Customer Support dataset (TWCS) as historical grounding.

No OpenAI API or paid API is required.

## 1. Problem Statement

Customer-support messages are short, noisy, and often ambiguous. A useful support assistant should identify the likely issue category, find similar historical conversations, draft a cautious response, and recognize when a human should take over.

This project demonstrates that workflow for AppleSupport messages. It is a prototype for analysis and support drafting, not an automated customer account system.

## 2. What the System Actually Does

For a customer message, the system:

1. Predicts one of 10 support intents using a lightweight keyword classifier in the runtime agent.
2. Runs semantic retrieval against historical AppleSupport cases.
3. Returns the top three cases and their cosine-style inner-product similarity scores.
4. Builds a prompt containing the customer message and retrieved historical examples.
5. Calls local Ollama with `llama3.2:1b` to draft a reply.
6. Decides `AUTO-HANDLE` or `ESCALATE` using explicit prototype rules.

The runtime agent launches retrieval in a subprocess. This keeps the Sentence Transformer memory out of the process that calls Ollama, which is important on memory-constrained Windows machines.

## 3. Architecture

```text
Customer message
       |
       v
Intent classifier
       |
       +------------------------------+
       |                              |
       v                              v
Python subprocess                Parent process
search_faiss.py                  keeps intent and orchestration
       |
       +-- Load FAISS index and retrieval CSV
       +-- Load all-MiniLM-L6-v2
       +-- Encode query and search top 3
       +-- Print similarity and historical text
       |
       v
Parent parses retrieval stdout
       |
       v
Grounded prompt with three examples
       |
       v
Ollama llama3.2:1b
       |
       v
Draft reply + AUTO-HANDLE / ESCALATE decision
```

The Streamlit application in `src/app.py` is a presentation layer. It reuses functions from `src/agent/agent.py` and does not duplicate classification, retrieval, generation, or escalation logic.

## 4. Dataset and AppleSupport Brand Selection

The source data is the Twitter Customer Support dataset in `archive/twcs/twcs.csv`. The reconstruction pipeline selects rows whose `author_id` is `AppleSupport` as support messages.

The current project contains 106,623 reconstructed customer-support pairs. These are historical records, not live Twitter data and not a real-time API connection.

## 5. Conversation Reconstruction

`src/data/reconstruct_conversations.py` reads the source CSV in chunks. It builds an index of AppleSupport replies keyed by `in_response_to_tweet_id`, then matches customer tweets to those replies by tweet ID.

The resulting pairs contain customer text, AppleSupport reply text, IDs, and timestamps. They are saved to:

```text
archive/processed/apple_support_pairs.csv
```

## 6. Intent Taxonomy

The project uses 10 support intents:

1. `device_setup`
2. `ios_update`
3. `battery_and_performance`
4. `keyboard_and_typing`
5. `mail_and_messages`
6. `account_and_access`
7. `network_and_connectivity`
8. `app_issues`
9. `payments_and_purchases`
10. `hardware_and_device`

The labeling workflow also uses `NEEDS_REVIEW` for examples that are ambiguous or do not fit safely into one intent. It is excluded from the supervised training scripts.

## 7. Golden Dataset

The manually labeled golden set is stored at:

```text
archive/processed/apple_support_golden.csv
```

It currently contains 177 manually labeled examples. The set is useful for prototyping and comparison, but it is small relative to the 106,623 reconstructed pairs. The labels also contain ambiguous and noisy examples, so results should not be presented as production-level classification performance.

Candidate selection and manual labeling are supported by:

```text
python src/data/create_golden_sample.py
python src/data/label_golden_set.py
```

## 8. TF-IDF + Logistic Regression Baseline

`src/models/train_intent_baseline.py` trains a scikit-learn pipeline consisting of:

- Word and bigram TF-IDF features
- Logistic Regression with `max_iter=1000`
- A deterministic `random_state=42`
- An 80/20 train/test split, stratified when the class counts allow it

The script reports accuracy, weighted precision, weighted recall, weighted F1, a classification report, and incorrect predictions.

Run it with:

```text
python src/models/train_intent_baseline.py
```

The baseline is an evaluation script, not a persisted production model. The runtime prototype does not retrain it on every request. Instead, the runtime agent uses a simple taxonomy-backed fallback classifier so the end-to-end demo remains lightweight and does not require model serialization.

## 9. Sentence Transformer Semantic Classifier

`src/models/train_intent_semantic.py` evaluates a second approach:

1. Encode labeled customer messages with `sentence-transformers/all-MiniLM-L6-v2`.
2. Train Logistic Regression on the resulting sentence embeddings.
3. Evaluate on the same style of held-out split.

Run it with:

```text
python src/models/train_intent_semantic.py
```

This is a semantic classification experiment separate from the runtime retrieval path. It also reports accuracy, weighted precision, weighted recall, weighted F1, and incorrect predictions.

## 10. FAISS Semantic Retrieval

`src/retrieval/build_faiss_index.py` creates the retrieval artifacts:

- It loads `apple_support_pairs.csv`.
- It deterministically samples at most 10,000 rows with `random_state=42`.
- It encodes customer messages with `all-MiniLM-L6-v2`.
- It normalizes embeddings.
- It builds a FAISS `IndexFlatIP` index.
- It saves the index and matching retrieval rows.

Outputs:

```text
archive/processed/apple_support.faiss
archive/processed/apple_support_retrieval.csv
```

The retrieval index contains 10,000 examples in the current project. `src/retrieval/search_faiss.py` searches the top three examples and prints their similarity scores and historical customer/support text.

## 11. Local Ollama Reply Generation

The grounded reply is generated locally with:

```text
llama3.2:1b
```

The prompt includes the customer message and the three retrieved historical AppleSupport cases. It asks for a concise, professional reply and instructs the model not to invent unsupported policies, refunds, guarantees, or actions.

Ollama is called from the parent agent process after the retrieval subprocess exits. This avoids keeping the Sentence Transformer model in memory alongside Ollama.

## 12. AUTO-HANDLE / ESCALATE Decision

The prototype returns `ESCALATE` when:

- Intent classification returns `NEEDS_REVIEW`.
- The best retrieval similarity is below `0.60`.
- The message requests a refund, payment intervention, account or security intervention, password or access intervention, or another action the prototype cannot safely perform.

Otherwise it returns `AUTO-HANDLE`.

Every escalation includes a human-readable reason. The decision is a conservative routing signal, not an authorization to take a real customer action.

## 13. Streamlit Demo

`src/app.py` provides a simple interface with:

- A customer message text area
- An `Analyze Message` button
- Predicted intent and confidence
- The top three historical cases and similarity scores
- Historical customer messages and AppleSupport replies
- The generated draft reply
- The routing decision and escalation reason

## 14. Project Structure

```text
archive/
  twcs/
    twcs.csv                         Source TWCS dataset
  processed/
    apple_support_pairs.csv         Reconstructed pairs
    apple_support_golden.csv        Manually labeled golden set
    apple_support.faiss             FAISS retrieval index
    apple_support_retrieval.csv     Rows aligned with the FAISS index

src/
  agent/
    agent.py                         End-to-end runtime agent
    generate_reply.py                Retrieval plus Ollama draft utility
  app.py                             Streamlit interface
  data/
    reconstruct_conversations.py    Build AppleSupport pairs
    create_golden_sample.py         Select labeling candidates
    label_golden_set.py              Manual labeling workflow
    ...
  models/
    train_intent_baseline.py        TF-IDF + Logistic Regression evaluation
    train_intent_semantic.py        Embedding + Logistic Regression evaluation
  retrieval/
    build_faiss_index.py            Build the sampled FAISS index
    search_faiss.py                 Search the FAISS index

requirements.txt                     Python dependencies
```

## 15. Installation

From the project root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install Ollama separately from the official Ollama distribution, then download the local model:

```powershell
ollama pull llama3.2:1b
```

If Ollama is not already running as a background service, start it in another terminal:

```powershell
ollama serve
```

The project does not require an OpenAI key or any paid API credential.

## 16. How to Run

Run the end-to-end command-line agent from the project root:

```powershell
python src/agent/agent.py
```

Run the Streamlit demo:

```powershell
streamlit run src/app.py
```

To rebuild the retrieval artifacts after changing the source pairs:

```powershell
python src/retrieval/build_faiss_index.py
```

To test retrieval directly:

```powershell
python src/retrieval/search_faiss.py
```

## 17. Example Input/Output

Example input:

```text
My iPhone battery is draining very quickly after the latest update.
```

Representative output shape:

```text
Intent: battery_and_performance
Intent confidence: 0.14
Top retrieved cases:
1. Similarity: 0.82
   Customer: ...historical customer message...
   AppleSupport reply: ...historical support reply...
2. Similarity: 0.79
   Customer: ...historical customer message...
   AppleSupport reply: ...historical support reply...
3. Similarity: 0.75
   Customer: ...historical customer message...
   AppleSupport reply: ...historical support reply...

Draft reply: ...locally generated grounded draft...
Decision: AUTO-HANDLE
Escalation reason: None
```

The exact retrieved cases, similarity scores, and draft vary with the input and local model behavior. The example is an output shape, not a claimed fixed result.

## 18. Evaluation Results

The project has two supervised evaluation scripts: the TF-IDF + Logistic Regression baseline and the Sentence Transformer embeddings + Logistic Regression experiment. Both use the current manually labeled golden data and report standard held-out metrics from the command line.

The current repository does not contain a saved evaluation report with authoritative numeric values. Therefore, this README does not invent or freeze accuracy, precision, recall, or F1 numbers. Run both commands to reproduce the current measurements in the local environment:

```powershell
python src/models/train_intent_baseline.py
python src/models/train_intent_semantic.py
```

There is no committed retrieval relevance benchmark yet. The 10,000-example FAISS index is an engineering choice to reduce memory use, not a claim that it represents all possible historical support cases.

## 19. Limitations

- The runtime intent classifier is a lightweight keyword fallback, not a persisted trained classifier.
- The golden set has only 177 manually labeled examples.
- The labels contain ambiguity and noise.
- The supervised metrics are exploratory and should not be treated as production-level performance.
- The repository does not currently record authoritative accuracy, precision, recall, or F1 results; rerun the evaluation scripts to measure the current data and environment.
- Retrieval uses a deterministic 10,000-row sample of 106,623 reconstructed pairs.
- Retrieval relevance has not been evaluated with a labeled Recall@3, MRR, or similar benchmark.
- Similarity threshold `0.60` is a prototype routing heuristic and has not been calibrated on a large validation set.
- Ollama output is not guaranteed to be correct, complete, or policy-compliant.
- The system drafts replies but cannot access accounts, change passwords, issue refunds, process payments, or perform other real customer actions.
- Historical Twitter conversations may be incomplete, informal, duplicated, or time-specific.
- There is no authentication, database, monitoring, rate limiting, or production deployment infrastructure.
- There is no implemented or measured LLM judge agreement study.

## What is misleading about my headline number?

The headline number of 106,623 reconstructed customer-support pairs sounds like a large supervised dataset, but it is primarily an unlabeled retrieval corpus. Only 177 examples are currently manually labeled for intent classification, and some of those labels are ambiguous or noisy.

The retrieval index is also built from a deterministic sample of 10,000 rows, not all 106,623 rows. As a result, the large pair count should not be presented as classification evidence or as proof of production-scale coverage.

## If I had one more week

I would prioritize:

1. Expand and rebalance the manually labeled evaluation set across all 10 intents.
2. Create a fixed, reviewed test set that is never used for tuning.
3. Persist and version the trained classifier rather than using the runtime keyword fallback.
4. Calibrate the retrieval threshold using labeled retrieval relevance judgments.
5. Add tests for subprocess parsing, escalation rules, and prompt construction.
6. Benchmark memory, latency, and failure behavior on representative Windows machines.
7. Add structured logging and basic monitoring without storing sensitive message content by default.
8. Review generated drafts against a written support policy before considering broader automation.

## Future Improvements

- Use a larger and more consistently labeled intent dataset.
- Compare calibrated linear, embedding, and hybrid classifiers.
- Improve retrieval with metadata filters, deduplication, and a larger memory-safe index.
- Add retrieval relevance evaluation such as Recall@3 and MRR.
- Add a policy or safety layer that checks generated drafts before display.
- Use a queue or service boundary for model calls if deployment constraints require it.
- Add retry, timeout, and health checks for Ollama and retrieval.
- Add unit, integration, and Streamlit smoke tests.
- Track model and dataset versions for reproducibility.
- Introduce human review feedback loops for escalated and accepted drafts.

## Interview Summary

This project demonstrates an end-to-end ML application workflow: data filtering and conversation reconstruction, manual labeling, baseline and semantic experiments, vector retrieval, local LLM grounding, conservative escalation, and a small user interface. Its strongest claim is a clear, reproducible prototype architecture. Its results should be interpreted in the context of a small noisy labeled set and a sampled retrieval index.
