import io
import wave

import numpy as np
import pandas as pd
import speech_recognition as sr
import streamlit as st
from pydub import AudioSegment
from textblob import TextBlob
from streamlit_webrtc import (
    AudioProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
import av


# ========== Core utilities ==========

def convert_to_wav_using_pydub(file) -> io.BytesIO:
    """
    Converts an uploaded audio file (MP3 or WAV) to WAV format in memory.
    Returns a BytesIO object ready for SpeechRecognition.
    """
    audio = AudioSegment.from_file(file)
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io


def transcribe_audio_from_file(file, language: str = "en-US") -> str:
    """
    Convert MP3/WAV audio to text using Google Speech Recognition.
    """
    recognizer = sr.Recognizer()

    wav_file = convert_to_wav_using_pydub(file)

    with sr.AudioFile(wav_file) as source:
        audio_data = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio_data, language=language)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        raise RuntimeError(f"Speech recognition service error: {e}")


def analyze_sentiment(text: str):
    """
    Returns polarity, subjectivity and a human-friendly label.
    """
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity  # [-1, 1]
    subjectivity = analysis.sentiment.subjectivity  # [0, 1]

    if polarity > 0:
        label = "Positive 😀"
    elif polarity < 0:
        label = "Negative 😕"
    else:
        label = "Neutral 😐"

    return polarity, subjectivity, label


def sentiment_probabilities_from_polarity(p: float):
    """
    Map polarity [-1, 1] to pseudo-probabilities for Neg / Neu / Pos
    just for visualization purposes.
    """
    pos = max(p, 0.0)
    neg = max(-p, 0.0)
    # keep neutral so they sum to 1
    neu = 1.0 - (pos + neg)
    neu = max(min(neu, 1.0), 0.0)

    total = pos + neg + neu
    if total == 0:
        return 0.33, 0.34, 0.33
    return neg / total, neu / total, pos / total


def show_sentiment_visualizations(text: str):
    """
    Show text, numeric stats, and a bar chart for sentiment.
    """
    polarity, subjectivity, label = analyze_sentiment(text)

    st.subheader("Sentiment Summary")
    st.write(f"**Label:** {label}")
    st.write(f"- Polarity: `{polarity:.3f}`  (−1 = very negative, +1 = very positive)")
    st.write(f"- Subjectivity: `{subjectivity:.3f}`  (0 = very objective, 1 = very subjective)")

    neg_prob, neu_prob, pos_prob = sentiment_probabilities_from_polarity(polarity)

    df = pd.DataFrame(
        {
            "Sentiment": ["Negative", "Neutral", "Positive"],
            "Score": [neg_prob, neu_prob, pos_prob],
        }
    ).set_index("Sentiment")

    st.subheader("Sentiment Visualization")
    st.bar_chart(df)

    return polarity, subjectivity, label


# ========== Microphone recording via streamlit-webrtc ==========

class MicAudioProcessor(AudioProcessorBase):
    """
    Collects audio frames from the user's microphone.
    We'll store raw mono int16 PCM data in a BytesIO buffer
    and later wrap it as a WAV file.
    """

    def __init__(self) -> None:
        self._frames = []
        self.sample_rate = 48000  # default; will be updated from frames if possible

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert to numpy array: shape (channels, samples)
        pcm = frame.to_ndarray()
        self.sample_rate = frame.sample_rate

        # Convert to mono (average channels)
        if pcm.ndim == 2:
            pcm_mono = pcm.mean(axis=0).astype(np.int16)
        else:
            pcm_mono = pcm.astype(np.int16)

        self._frames.append(pcm_mono.tobytes())

        # We don't modify the outgoing audio; just pass it through.
        return frame

    def get_wav_bytes_io(self) -> io.BytesIO:
        """
        Build a WAV file in memory from collected PCM frames.
        """
        if not self._frames:
            return None

        wav_io = io.BytesIO()

        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(self._frames))

        wav_io.seek(0)
        return wav_io

    def reset(self):
        self._frames = []


def transcribe_from_mic_processor(proc: MicAudioProcessor, language: str = "en-US") -> str:
    """
    Use SpeechRecognition on the audio captured by MicAudioProcessor.
    """
    wav_io = proc.get_wav_bytes_io()
    if wav_io is None:
        return ""

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_io) as source:
        audio_data = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio_data, language=language)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        raise RuntimeError(f"Speech recognition service error: {e}")


# ========== Streamlit UI ==========

def main():
    st.title("🎤 Tone and Sentiment Analyzer")

    st.write(
        """
        Analyze the **sentiment** of what you say or upload:

        - 📤 Upload an audio file (**MP3** or **WAV**), or  
        - 🎙️ Record directly from your **microphone** (browser-based).  

        I’ll transcribe the speech and show you the sentiment with visualizations.
        """
    )

    language = st.selectbox(
        "Language of the speech:",
        ["en-US", "es-US", "es-ES"],
        index=0,
    )

    tab_upload, tab_mic = st.tabs(["📁 Upload Audio", "🎙️ Microphone"])

    # ---- Upload tab ----
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload audio (MP3 or WAV)",
            type=["mp3", "wav"],
            help="Short clips (a few seconds) work best.",
        )

        if st.button("Analyze Uploaded Audio"):
            if uploaded_file is None:
                st.warning("Please upload an MP3 or WAV file first.")
            else:
                with st.spinner("Processing and transcribing audio..."):
                    try:
                        text = transcribe_audio_from_file(uploaded_file, language=language)
                    except RuntimeError as e:
                        st.error(str(e))
                        text = ""

                if not text.strip():
                    st.error(
                        "I couldn't understand the audio. "
                        "Try a clearer recording, or verify the language setting."
                    )
                else:
                    st.subheader("Transcribed Text")
                    st.write(text)
                    show_sentiment_visualizations(text)

    # ---- Microphone tab ----
    with tab_mic:
        st.write(
            "Click **Start** in the widget below, speak for a few seconds, "
            "then click **Stop**. When you're ready, click **Transcribe & Analyze**."
        )

        ctx = webrtc_streamer(
            key="mic",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=256,
            media_stream_constraints={"audio": True, "video": False},
            audio_processor_factory=MicAudioProcessor,
        )

        if "mic_transcribed_text" not in st.session_state:
            st.session_state["mic_transcribed_text"] = ""

        if ctx.audio_processor is not None:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Transcribe & Analyze (Mic)"):
                    with st.spinner("Transcribing microphone audio..."):
                        proc: MicAudioProcessor = ctx.audio_processor
                        try:
                            text = transcribe_from_mic_processor(proc, language=language)
                        except RuntimeError as e:
                            st.error(str(e))
                            text = ""

                        if not text.strip():
                            st.error(
                                "I couldn't understand the microphone audio. "
                                "Try again and speak clearly near the mic."
                            )
                        else:
                            st.session_state["mic_transcribed_text"] = text
                            # Reset frames so next recording is fresh
                            proc.reset()

            with col2:
                if st.button("Reset Mic Buffer"):
                    proc = ctx.audio_processor
                    if proc is not None:
                        proc.reset()
                    st.session_state["mic_transcribed_text"] = ""
                    st.info("Microphone buffer reset.")

        if st.session_state.get("mic_transcribed_text"):
            st.subheader("Transcribed Text (Mic)")
            st.write(st.session_state["mic_transcribed_text"])
            show_sentiment_visualizations(st.session_state["mic_transcribed_text"])


if __name__ == "__main__":
    main()
