"""CodeAlpha FAQ Chatbot — TF-IDF + cosine similarity (scikit-learn).

Usage:
    python chatbot.py                  # interactive loop
    python chatbot.py "your question"  # single-question mode
"""

import json
import re
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQ_PATH = Path(__file__).with_name("faqs.json")
SIMILARITY_THRESHOLD = 0.3

# Small built-in stopword list (pure Python; no NLTK download needed).
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "am", "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "its", "they", "them", "their", "this", "that", "these", "those",
    "and", "or", "but", "if", "then", "else", "when", "while", "of", "at",
    "by", "for", "with", "about", "into", "through", "after", "over",
    "between", "out", "against", "during", "without", "before", "under",
    "around", "among", "to", "from", "up", "down", "in", "on", "off",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "can", "could", "will", "would", "shall", "should", "may", "might",
    "must", "what", "which", "who", "whom", "how", "why", "where",
    "please", "tell", "give", "know", "need",
}


def preprocess(text: str) -> str:
    """Lowercase, tokenize with regex, drop stopwords, rejoin."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


def load_faqs(path: Path = FAQ_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class FAQChatbot:
    def __init__(self, faqs: list[dict]):
        self.faqs = faqs
        self.questions = [faq["question"] for faq in faqs]
        self.vectorizer = TfidfVectorizer(preprocessor=preprocess)
        self.question_vectors = self.vectorizer.fit_transform(self.questions)

    def get_response(self, user_input: str) -> str:
        """Return the best-matching FAQ answer, or a fallback with suggestions."""
        if not user_input.strip():
            return "Please ask me a question — for example, 'What is CodeAlpha?'"

        user_vector = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, self.question_vectors)[0]
        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])

        if best_score < SIMILARITY_THRESHOLD:
            # Suggest the 2-3 most similar FAQ questions, even if below threshold.
            top_indices = similarities.argsort()[::-1][:3]
            suggestions = "\n".join(
                f"  - {self.questions[i]}" for i in top_indices
            )
            return (
                "I'm sorry, I don't know the answer to that question. "
                "I can only answer questions about CodeAlpha, the AI "
                "internship, and general AI topics.\n"
                "Here are some questions you could try:\n"
                f"{suggestions}"
            )

        return self.faqs[best_idx]["answer"]


def main() -> None:
    faqs = load_faqs()
    bot = FAQChatbot(faqs)

    # Single-question mode: python chatbot.py "question here"
    if len(sys.argv) > 1:
        print(bot.get_response(" ".join(sys.argv[1:])))
        return

    # Interactive loop
    print("FAQ Chatbot — ask me about CodeAlpha, the AI internship, or AI.")
    print("Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break
        if user_input.lower() in {"quit", "exit"}:
            print("Bot: Goodbye!")
            break
        if not user_input:
            continue
        print(f"Bot: {bot.get_response(user_input)}\n")


if __name__ == "__main__":
    main()
