# Video & Document Summarizer and Q&A System

This project is a Streamlit-based application that allows users to summarize YouTube videos and documents, extract transcripts, and perform Q&A based on the extracted content. The system integrates several APIs and models, including Google Gemini (PaLM), IBM Watson Speech-to-Text, and Google Translate. Additionally, it handles multimedia content like YouTube videos and supports document types like PDFs, DOCX, and plain text files.

## Project Overview

The system extracts text content from YouTube videos and documents, then allows users to ask questions based on the extracted transcripts or document content. This is done through a **Retrieval-Augmented Generation (RAG)** system that retrieves relevant document chunks and generates answers using the **Google Gemini** API.

## Features

- **YouTube Transcript Extraction**: Fetches transcripts from YouTube videos using the YouTube Transcript API. If no transcript is available, the audio is downloaded and transcribed using IBM Watson Speech-to-Text.
- **Document Upload & Summarization**: Supports PDF, DOCX, and plain text file uploads. Documents are split into manageable chunks for efficient summarization and Q&A tasks.
- **Multilingual Support**: Automatically translates transcripts to English if required using Google Cloud Translation API.
- **Question-Answering System**: Uses Google Gemini (PaLM) to generate answers based on retrieved document content or video transcripts.
- **Error Handling**: The system handles cases where no transcript is available, and falls back to audio-based transcription. If an error occurs, it is handled gracefully with error messages displayed to the user.

## APIs and Libraries Used

- **Google Gemini (PaLM)**: For question-answering tasks and generating responses.
- **IBM Watson Speech-to-Text**: For transcribing audio from YouTube videos when transcripts are not available.
- **YouTube Transcript API**: To fetch available transcripts from YouTube videos.
- **Google Cloud Translation API**: For translating non-English transcripts.
- **PyPDF2**, **DOCXLoader**, **TextLoader**: For handling PDF, DOCX, and TXT files.

## File Structure

- `app.py`: The main Streamlit application file where all components are integrated.
- `requirements.txt`: List of all required Python libraries for the project.
- `README.md`: This README file.
- `.env`: Environment variables, including API keys for IBM Watson and Google services.

## Setup & Running the Project

1. **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Configure environment variables**:
    Create a `.env` file in the project root with the following content:
    ```plaintext
    API_KEY=your_google_gemini_api_key
    apikey=your_ibm_watson_api_key
    url=your_ibm_watson_url
    ```

3. **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

4. **Using the App**:
   - Enter a YouTube URL to fetch or transcribe video content.
   - Upload a document (PDF, DOCX, or TXT) to extract content for Q&A.
   - Ask questions based on the YouTube transcript or uploaded document.

## Error Handling and Future Deployment

- The project is handling an error related to embedding functions, and the website will be deployed once this issue is resolved.
- Until the error is fixed, the app runs locally and is fully functional.

## Author

- **Jaspinder Singh Maan**
