# youtube-video-summarization-and-langchain-rag-for-Q-A-
Streamlit app that summarizes Arabic &amp; English YouTube videos with timestampes and provides LangChain RAG question-answering over the transcript
# YouTube Summarizer + RAG Q&A

**One-file Streamlit app** that:  
- fetches YouTube transcripts using `YouTubeTranscriptApi.fetch` (api.fetch),  
- summarizes the video (English / Arabic) with HuggingFace summarization pipelines,  
- chunks the transcript into documents (with timestamps),  
- builds a LangChain RAG (Retrieval-Augmented Generation) system using FAISS and a HuggingFace text2text model,  
- answers user questions and shows source timestamps.

This repository focuses on a minimal Streamlit UI and keeps the original pipeline you provided intact.
and also you can find a google colab without streamlit in the files

## Features
- Supports English and Arabic videos.
- Uses `YouTubeTranscriptApi.fetch` for transcripts.
- Local summarization pipelines (`facebook/bart-large-cnn` for EN, `csebuetnlp/mT5_multilingual_XLSum` for AR).
- RAG built with FAISS + HuggingFace generation models (`google/flan-t5-base` for EN; `UBC-NLP/AraT5-base-title-generation` for AR).
- Shows provenance: timestamps and source chunks.
- Single-file Streamlit app: `streamlit_youtube_summarizer_app.py`.

## Quick start

1. Clone the repo:
```bash
git clone https://github.com/<your-username>/youtube-summarizer-rag.git
cd youtube-summarizer-rag
