import os
import sys

# Allow importing translate_text / text_to_speech from app.py when run directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import text_to_speech, translate_text  # noqa: E402

TEST_TEXT = "Hello, how are you?"
TARGET = "Urdu"


def _is_urdu_script(text: str) -> bool:
    """True if the text contains at least one Urdu/Arabic-script character."""
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


def main() -> None:
    translated = translate_text(TEST_TEXT, source="en", target="ur")
    print(f"Input : {TEST_TEXT}")
    print(f"Target: {TARGET}")
    print(f"Output: {translated}")
    assert translated and translated.strip(), "Translation came back empty!"
    assert translated.strip() != TEST_TEXT, "Translation is unchanged English text!"
    assert _is_urdu_script(translated), "Translation does not contain Urdu script!"
    print("Translation check passed: non-empty Urdu text received.")

    audio = text_to_speech(translated, lang="ur")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_tts.mp3")
    with open(out_path, "wb") as f:
        f.write(audio)
    assert os.path.getsize(out_path) > 0, "TTS MP3 is empty!"
    print(f"TTS check passed: wrote {out_path} ({os.path.getsize(out_path)} bytes).")


if __name__ == "__main__":
    main()
