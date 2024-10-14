import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import requests
from pydub import AudioSegment
from io import BytesIO
import yt_dlp as youtube_dl
from google.cloud import translate_v2 as translate
import time

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key="")

# IBM Watson credentials
apikey = ""
url = ""

# Initialize IBM Watson Speech to Text with SSL verification disabled
authenticator = IAMAuthenticator(apikey)
speech_to_text = SpeechToTextV1(authenticator=authenticator)
speech_to_text.set_service_url(url)
speech_to_text.set_disable_ssl_verification(True)

# Initialize Google Translate API
translate_client = translate.Client()

def get_youtube_transcript(video_id):
    """Retrieve the transcript for a YouTube video if available."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            # Fall back to the first available transcript if English is not available
            transcript = transcript_list.find_manually_created_transcript(['en']) or transcript_list.find_generated_transcript(['en'])

        # Return the transcript as plain text
        formatter = TextFormatter()
        return formatter.format_transcript(transcript.fetch())
    except Exception as e:
        st.write(f"No transcript available: {e}")
        return None

def translate_transcript(transcript, target_language="en"):
    """Translate the transcript to the target language."""
    result = translate_client.translate(transcript, target_language=target_language)
    return result['translatedText']

def download_audio_from_url(audio_url):
    """Download audio from a given URL."""
    response = requests.get(audio_url)
    audio = AudioSegment.from_file(BytesIO(response.content))
    return audio

def download_audio_from_youtube(video_url):
    """Download audio from a YouTube video URL."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': '/tmp/%(id)s.%(ext)s',
        'quiet': True,
        'postprocessor_args': ['-acodec', 'mp3'],
        'ffmpeg_location': r'C:\\Users\\jaspi\\OneDrive\\Desktop\\ffmpeg\\ffmpeg-7.0.1-essentials_build\\bin'  # Update this with your ffmpeg bin path
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=True)
        audio_file_path = ydl.prepare_filename(info_dict)
        audio_file_path = audio_file_path.replace('.webm', '.mp3').replace('.m4a', '.mp3')

    audio = AudioSegment.from_file(audio_file_path, format="mp3")
    return audio

def transcribe_audio(audio):
    """Transcribe audio data using IBM Watson Speech-to-Text API."""
    audio_bytes = BytesIO()
    audio.export(audio_bytes, format="wav")
    audio_bytes.seek(0)
    
    response = speech_to_text.recognize(
        audio=audio_bytes,
        content_type='audio/wav'
    ).get_result()

    transcription = ""
    for result in response['results']:
        transcription += result['alternatives'][0]['transcript'] + " "
    
    return transcription

# Function to get responses from Gemini model based on the transcript
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question, context):
    prompt = f"{context}\n\nQuestion: {question}"
    response = chat.send_message(prompt, stream=True)
    return response

# Initialize Streamlit app
st.set_page_config(page_title="YouTube Video Summarizer and Q&A")
st.header("Video Quizzer")

# Initialize session state for chat history and transcript if they don't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'transcript' not in st.session_state:
    st.session_state['transcript'] = ""

# User input for YouTube URL
audio_url = st.text_input("Enter video URL:", key="audio_url")

# Display the YouTube video if a valid URL is provided
if audio_url:
    if "youtube.com" in audio_url or "youtu.be" in audio_url:
        st.video(audio_url)

if st.button("Get Transcript"):
    if audio_url:
        with st.spinner("Downloading and transcribing..."):
            try:
                start_time = time.time()
                if "youtube.com" in audio_url or "youtu.be" in audio_url:
                    video_id = audio_url.split("v=")[-1]
                    transcript = get_youtube_transcript(video_id)
                    if transcript:
                        # Translate transcript if it's not in English
                        if transcript and 'en' not in transcript:
                            transcript = translate_transcript(transcript)
                        st.session_state['transcript'] = transcript
                        st.write("Transcription (YouTube):")
                        st.write(transcript)
                    else:
                        audio = download_audio_from_youtube(audio_url)
                        transcription = transcribe_audio(audio)
                        st.session_state['transcript'] = transcription
                        st.write("Transcription (IBM Watson):")
                        st.write(transcription)
                else:
                    audio = download_audio_from_url(audio_url)
                    transcription = transcribe_audio(audio)
                    st.session_state['transcript'] = transcription
                    st.write("Transcription (IBM Watson):")
                    st.write(transcription)
                end_time = time.time()
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a valid URL.")

# User input for additional questions
input_question = st.text_input("Ask a question based on the transcript:", key="input_question")

if st.button("Submit Question") and input_question:
    transcript = st.session_state.get('transcript', "")

    if transcript:
        response = get_gemini_response(input_question, transcript)
        for chunk in response:
            st.write(chunk.text)
    else:
        st.error("Please extract the transcript from a YouTube video first.")

# Display the extracted transcript
if st.session_state['transcript']:
    st.write("Transcript:", st.session_state['transcript'])
