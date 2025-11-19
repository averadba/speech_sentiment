import streamlit as st
import speech_recognition as sr
from textblob import TextBlob


def transcribe_audio(file, language: str = "en-US") -> str:
    """
    Convert an uploaded WAV audio file to text using Google Speech Recognition.
    """
    recognizer = sr.Recognizer()

    # Streamlit's UploadedFile behaves like a file object
    with sr.AudioFile(file) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data, language=language)
        return text
    except sr.UnknownValueError:
        # Speech could not be understood
        return ""
    except sr.RequestError as e:
        # API was unreachable or unresponsive
        raise RuntimeError(f"Speech recognition service error: {e}")


def sentiment_label(text: str) -> str:
    """
    Return a human-friendly sentiment label based on TextBlob polarity.
    """
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        return f"Positive 😀 (polarity={polarity:.2f})"
    elif polarity < 0:
        return f"Negative 😕 (polarity={polarity:.2f})"
    else:
        return f"Neutral 😐 (polarity={polarity:.2f})"


def main():
    st.title("Tone and Sentiment Analyzer")

    st.write(
        """
        Upload a short **WAV** audio clip with speech, and this app will:
        1. Transcribe what was said.
        2. Analyze the **sentiment** of the text (positive / neutral / negative).
        """
    )

    # Optional language selector
    language = st.selectbox(
        "Language of the speech in the audio:",
        options=["en-US", "es-US", "es-ES"],
        index=0,
    )

    uploaded_file = st.file_uploader(
        "Upload an audio file (WAV format)",
        type=["wav"],
        help="For reliability on Streamlit Cloud, only WAV is supported.",
    )

    if st.button("Analyze"):
        if uploaded_file is None:
            st.warning("Please upload a WAV audio file first.")
            return

        with st.spinner("Transcribing audio..."):
            try:
                text = transcribe_audio(uploaded_file, language=language)
            except RuntimeError as e:
                st.error(str(e))
                return

        if not text.strip():
            st.error(
                "I couldn't understand the audio. "
                "Try a clearer recording, or make sure the language setting matches the audio."
            )
            return

        st.subheader("Transcribed Text")
        st.write(text)

        st.subheader("Sentiment")
        label = sentiment_label(text)
        st.success(label)


if __name__ == "__main__":
    main()
