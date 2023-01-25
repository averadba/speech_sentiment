from pydub import AudioSegment
from pydub_silence import split_on_silence
import sounddevice as sd
import soundfile as sf
import textblob
import streamlit as st

def speech_to_text():
    fs = 44100  # Sample rate
    seconds = 3  # Duration of recording

    # Recording the audio
    myrecording = sd.rec(int(fs * seconds), samplerate=fs, channels=2)
    sd.wait()
    sf.write('audio.wav', myrecording, fs)

    # Loading the audio
    audio = AudioSegment.from_wav("audio.wav")
    # Splitting the audio on silence
    chunks = split_on_silence(audio, min_silence_len=1000, silence_thresh=-16)
    # Concatenating the chunks and converting to text
    text = " ".join([chunk.get_flac_data() for chunk in chunks])
    return text

def sentiment_analysis(text):
    analysis = textblob.TextBlob(text)
    sentiment = analysis.sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment == 0:
        return "Neutral"
    else:
        return "Negative"

def main():
    st.title("Tone and Sentiment Analyzer")
    st.write("Speak into the microphone and press the 'Analyze' button to determine the tone and sentiment of your conversation.")
    if st.button("Analyze"):
        text = speech_to_text()
        sentiment = sentiment_analysis(text)
        st.success("Sentiment: {}".format(sentiment))

if __name__ == "__main__":
    main()
