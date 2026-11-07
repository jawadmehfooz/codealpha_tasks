"""CodeAlpha Language Translator — Streamlit app.

Translates text between languages using deep-translator's GoogleTranslator
(no API key required) and reads the translation aloud with gTTS.

Fallback: if deep-translator's endpoint is rate-limited (HTTP 429), the
translation is retried against Google's free JSON endpoint
(translate.googleapis.com/translate_a/single) using only the standard library.
"""

import io
import json
import urllib.parse
import urllib.request

import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS

# Language name -> ISO 639-1 code (deep-translator / Google Translate)
LANGUAGES = {
    "English": "en",
    "Urdu": "ur",
    "Hindi": "hi",
    "Arabic": "ar",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Chinese": "zh-CN",
    "Turkish": "tr",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Persian": "fa",
    "Portuguese": "pt",
    "Italian": "it",
    "Dutch": "nl",
    "Bengali": "bn",
    "Punjabi": "pa",
    "Indonesian": "id",
    "Malay": "ms",
}


def _translate_via_google_api(text: str, source: str, target: str) -> str:
    """Fallback translation via Google's free JSON endpoint (no API key).

    Used only when deep-translator's endpoint is unreachable or rate-limited.
    """
    params = urllib.parse.urlencode(
        {"client": "gtx", "sl": source, "tl": target, "dt": "t", "q": text}
    )
    url = f"https://translate.googleapis.com/translate_a/single?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    translated = "".join(seg[0] for seg in data[0] if seg and seg[0])
    if not translated:
        raise ValueError("Google Translate JSON endpoint returned an empty result")
    return translated


def translate_text(text: str, source: str, target: str) -> str:
    """Translate ``text`` from ``source`` language code to ``target`` language code.

    Primary: deep-translator's GoogleTranslator. If that fails (e.g. Google
    rate-limits its scraping endpoint), fall back to Google's free JSON endpoint.
    """
    try:
        return GoogleTranslator(source=source, target=target).translate(text)
    except Exception:
        return _translate_via_google_api(text, source, target)


def text_to_speech(text: str, lang: str) -> bytes:
    """Generate MP3 audio bytes for ``text`` spoken in language ``lang``."""
    buffer = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buffer)
    buffer.seek(0)
    return buffer.read()


def main() -> None:
    st.set_page_config(page_title="Language Translator", page_icon="🌍")
    st.title("🌍 Language Translator")
    st.write(
        "Translate text between languages instantly — powered by Google Translate "
        "(via deep-translator) and text-to-speech with gTTS."
    )

    text = st.text_area(
        "Enter text to translate",
        placeholder="Type something here, e.g. Hello, how are you?",
        height=120,
    )

    col_src, col_arrow, col_tgt = st.columns([5, 1, 5])
    with col_src:
        source_lang = st.selectbox("Source language", list(LANGUAGES.keys()), index=0)
    with col_arrow:
        st.markdown("<div style='text-align:center; font-size:28px; margin-top:18px;'>➡️</div>", unsafe_allow_html=True)
    with col_tgt:
        target_lang = st.selectbox("Target language", list(LANGUAGES.keys()), index=1)

    if st.button("Translate", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Please enter some text to translate.")
            return
        if source_lang == target_lang:
            st.warning("Source and target languages are the same.")
            return

        with st.spinner("Translating..."):
            try:
                translated = translate_text(
                    text.strip(), LANGUAGES[source_lang], LANGUAGES[target_lang]
                )
            except Exception as exc:  # network / Google Translate errors
                st.error(f"Translation failed: {exc}")
                return

        st.subheader("Translation")
        st.success(translated)

        # Copy button: st.code renders the text with a built-in copy icon.
        st.code(translated, language=None)

        # Text-to-speech for the translated text.
        with st.spinner("Generating audio..."):
            try:
                audio_bytes = text_to_speech(translated, LANGUAGES[target_lang])
            except Exception as exc:
                st.error(f"Text-to-speech failed: {exc}")
                return

        st.subheader("🔊 Listen")
        st.audio(audio_bytes, format="audio/mp3")
        st.download_button(
            label="⬇️ Download audio (MP3)",
            data=audio_bytes,
            file_name="translation.mp3",
            mime="audio/mp3",
        )


if __name__ == "__main__":
    main()
