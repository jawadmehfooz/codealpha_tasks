# CodeAlpha FAQ Chatbot

A chatbot that answers frequently asked questions about **CodeAlpha**, the
**AI internship**, and **general AI topics**, using **TF-IDF vectorization**
and **cosine similarity** (scikit-learn).

## Project Purpose

This is Project 2 of 3 for the **CodeAlpha Artificial Intelligence Internship
(Oct–Nov 2026)**. The chatbot matches a user's question against a knowledge
base of FAQ pairs in `faqs.json`, finds the most similar question, and returns
its answer. If no FAQ is similar enough, it politely says it doesn't know and
suggests related questions to try.

## How It Works

1. **Preprocessing** — each question is lowercased, tokenized with a regex,
   and common stopwords are removed (pure Python; no NLTK download needed).
2. **Vectorization** — the FAQ questions are converted to TF-IDF vectors with
   `sklearn.feature_extraction.text.TfidfVectorizer`.
3. **Matching** — the user's question is vectorized the same way and compared
   to every FAQ question with cosine similarity. The best match wins.
4. **Fallback** — if the best similarity score is below **0.3**, the bot
   replies that it doesn't know and suggests the 2–3 most similar FAQ
   questions.

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

### Command-line chatbot (interactive)

```bash
python chatbot.py
```

Type your question and press Enter. Type `quit` or `exit` to stop.

### Command-line chatbot (single question)

```bash
python chatbot.py "What is machine learning?"
```

### Streamlit web UI

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

## Demo

The demo shows the chatbot answering questions from its FAQ knowledge base:

- **Exact question** — *"What is machine learning?"* returns the FAQ answer
  about machine learning.
- **Paraphrased question** — *"How long does the internship last?"* still
  matches the FAQ about the internship duration.
- **Off-topic question** — *"What is the weather on Mars?"* triggers the
  fallback reply with suggested questions.

The Streamlit UI shows the same behaviour in a chat interface.

## Files

| File | Description |
| --- | --- |
| `faqs.json` | Knowledge base — 18 question/answer pairs |
| `chatbot.py` | Matching logic (TF-IDF + cosine similarity) and CLI |
| `app.py` | Streamlit chat UI (reuses `chatbot.py`) |
| `requirements.txt` | Dependencies: scikit-learn, streamlit |

## Author

**Jawad Mehfooz** — CodeAlpha Artificial Intelligence Internship, Oct–Nov 2026.
