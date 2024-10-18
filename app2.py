import streamlit as st
import tempfile
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
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()
# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("API_KEY"))

# IBM Watson credentials
apikey = os.getenv("apikey")
url = os.getenv("url")
if not apikey or not url:
    raise ValueError("IBM Watson API key or URL is missing. Please check your environment variables.")

# Initialize IBM Watson Speech to Text
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
        'ffmpeg_location': r'C:\\Users\\jaspi\\OneDrive\\Desktop\\ffmpeg\\ffmpeg-7.0.1-essentials_build\\bin'
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
llm = genai.GenerativeModel("gemini-1.5-pro")
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question, context):
    prompt = f"{context}\n\nQuestion: {question}"
    response= chat.send_message(prompt, stream="true")
    return response


# Initialize Streamlit app
st.set_page_config(page_title="Video & Document Q&A System")
st.header("Video & Document Summarizer and Q&A")

# User input for YouTube URL or file upload
audio_url = st.text_input("Enter YouTube video URL:", key="audio_url")
uploaded_file = st.file_uploader("Upload a document (pdf, docx, txt):", type=["pdf", "docx", "txt"])

# Transcript extraction and display
if audio_url:
    if "youtube.com" in audio_url or "youtu.be" in audio_url:
        st.video(audio_url)

    if st.button("Get Transcript"):
        if audio_url:
            with st.spinner("Downloading and transcribing..."):
                try:
                    start_time = time.time()
                    video_id = audio_url.split("v=")[-1]
                    transcript = get_youtube_transcript(video_id)
                    if transcript:
                        if 'en' not in transcript:
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
                    end_time = time.time()
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.error("Please enter a valid URL.")

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension == "pdf":
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Pass the temporary file path to PyPDFLoader
            loader = PyPDFLoader(temp_file_path)

    elif file_extension == "txt":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            loader = TextLoader(temp_file_path)

    elif file_extension == "docx":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            loader = DOCXLoader(temp_file_path)

    
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)
    
    vectorstore = Chroma.from_documents(docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use 200 words maximum and keep the answer concise."
        "\n\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    
    query = st.text_input("Ask a question based on the document:", key="query")
    
    if query:
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input": query})
        st.write(response["answer"])

# Display the extracted transcript for YouTube
if 'transcript' in st.session_state and st.session_state['transcript']:
    st.write("Transcript:", st.session_state['transcript'])

# User input for YouTube questions
input_question = st.text_input("Ask a question based on the YouTube transcript:", key="input_question")

if st.button("Submit YouTube Question") and input_question:
    transcript = st.session_state.get('transcript', "")
    if transcript:
        response = get_gemini_response(input_question, transcript)
        for chunk in response:
            st.write(chunk.text)  #use chunk to get response in a stream otherwise it will get the whole response and then give the answer and you might need to wait
    else:
        st.error("Please extract the transcript from a YouTube video first.")
