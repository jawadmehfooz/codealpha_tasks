# CodeAlpha Language Translator 🌍

A simple Streamlit web app that translates text between languages using
Google Translate (via the `deep-translator` library — no API key needed) and
reads the translation aloud with Google Text-to-Speech (`gTTS`).

Built as **Project 1 of 3** for the **CodeAlpha Artificial Intelligence
Internship (Oct–Nov 2026)**.

## Features

- Translate text between 20 languages, including English, Urdu, Hindi,
  Arabic, French, Spanish, German and Chinese
- Source and target language dropdowns
- Copy the translated text (built-in copy button on the result block)
- Text-to-speech: listen to the translation in the target language and
  download it as an MP3
- Resilient translation: `deep-translator` is tried first; if Google
  rate-limits that endpoint, the app automatically falls back to Google's
  free JSON translation endpoint (standard library only, no extra dependency)

## Demo

1. Type or paste text into the input box.
2. Pick a source and a target language.
3. Click **Translate** — the translation appears below.
4. Press play on the audio player to hear it, or download the MP3.

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Test

```bash
python test_translate.py
```

Translates "Hello, how are you?" from English to Urdu, verifies the result
is non-empty Urdu script, and generates a short `test_tts.mp3` to prove
text-to-speech works.

## Tech Stack

- Python 3.12
- [Streamlit](https://streamlit.io/) — web UI
- [deep-translator](https://github.com/nidhaloff/deep-translator) — Google Translate wrapper
- [gTTS](https://github.com/pndurette/gTTS) — Google Text-to-Speech

## Author

**Jawad Mehfooz** — CodeAlpha Artificial Intelligence Internship, Oct–Nov 2026
