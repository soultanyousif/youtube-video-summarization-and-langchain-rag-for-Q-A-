import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from urllib.parse import urlparse, parse_qs
import yt_dlp
import whisper
from langdetect import detect
from langchain_core.documents import Document
from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import math
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="YT Summarizer RAG", layout="wide")

# Minimal black & white styling
st.markdown(
    """
    <style>
    /* page */
    .main * {color: #000000; background-color: #ffffff}
    .stButton>button {background: #000000; color: #ffffff}
    .stTextInput>div>div>input {background: #ffffff; color: #000000}
    .stTextArea>div>div>textarea {background: #ffffff; color: #000000}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown p {color: #000000}
    /* remove Streamlit default accent colors to keep B/W */
    .css-1v3fvcr, .css-1v3fvcr button {background: #ffffff}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- Utility functions (from user's code) ----------------------

def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    video_ids = qs.get('v')
    if not video_ids:
        raise ValueError(f"No video id found in URL: {url}")
    return video_ids[0]


@st.cache_data(show_spinner=False)
def extract_transcript(url):
    """
    Extract transcript from YouTube video using YouTubeTranscriptApi.fetch (user requirement).
    Returns full_text and segments (with start, duration, text).
    """
    video_id = extract_video_id(url)
    api = YouTubeTranscriptApi()

    try:
        # Try English, fallback to Arabic
        try:
            fetched = api.fetch(video_id, languages=['en'])
        except Exception:
            fetched = api.fetch(video_id, languages=['ar'])

        full_text = "\n".join(snippet['text'] if isinstance(snippet, dict) else snippet.text for snippet in fetched)
        segments = [
            {"start": (snippet['start'] if isinstance(snippet, dict) else snippet.start),
             "duration": (snippet['duration'] if isinstance(snippet, dict) else snippet.duration),
             "text": (snippet['text'] if isinstance(snippet, dict) else snippet.text)}
            for snippet in fetched
        ]
        return full_text, segments

    except (TranscriptsDisabled, NoTranscriptFound):
        raise RuntimeError("Transcript not available for this video.")


def detect_lang(text):
    try:
        lang = detect(text)
        return "ar" if lang.startswith("ar") else "en"
    except Exception:
        return "en"


def chunk_transcript(segments, chunk_size=500, chunk_overlap=50):
    """
    Chunks transcript segments into LangChain Document objects with metadata.
    """
    documents = []
    current_chunk = ""
    current_start_time = 0

    for i, seg in enumerate(segments):
        if not current_chunk:
            current_start_time = seg['start']

        segment_text = seg['text'].strip()
        if current_chunk:
            current_chunk += " " + segment_text
        else:
            current_chunk = segment_text

        if len(current_chunk) >= chunk_size:
            documents.append(
                Document(
                    page_content=current_chunk,
                    metadata={'start': current_start_time}
                )
            )

            # build overlap chunk
            overlap_text = ""
            temp_len = 0
            for j in range(i, -1, -1):
                overlap_segment_text = segments[j]['text'].strip()
                if overlap_text:
                    overlap_text = overlap_segment_text + " " + overlap_text
                else:
                    overlap_text = overlap_segment_text

                temp_len += len(overlap_segment_text) + 1
                if temp_len >= chunk_overlap:
                    current_chunk = overlap_text
                    current_start_time = segments[j]['start']
                    break
            else:
                current_chunk = ""

    if current_chunk:
        documents.append(
            Document(
                page_content=current_chunk,
                metadata={'start': current_start_time}
            )
        )

    return documents


# Create summarizers as cached resources (so Streamlit won't reload them repeatedly)
@st.cache_resource(show_spinner=False)
def get_summarizers():
    en_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    ar_summarizer = pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum")
    return en_summarizer, ar_summarizer


def summarise_text(text, lang="en"):
    text = text.strip()
    if not text:
        return ""

    words = text.split()
    text = " ".join(words[:900]) if len(words) > 900 else text

    en_summarizer, ar_summarizer = get_summarizers()

    if lang == "en":
        summary = en_summarizer(text, max_length=200, min_length=50, do_sample=False)
    else:
        summary = ar_summarizer(text, max_length=200, min_length=50, do_sample=False)

    return summary[0]['summary_text']


@st.cache_resource(show_spinner=False)
def get_rag_tools(lang="en"):
    # embeddings
    if lang == "en":
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    else:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    return embeddings


def build_rag(docs, lang="en"):
    """
    Build a LangChain RetrievalQA (RAG) object from a list of Documents.
    """
    if not docs:
        raise ValueError("No documents provided for RAG!")

    embeddings = get_rag_tools(lang)
    db = FAISS.from_documents(docs, embeddings)

    if lang == "en":
        hf_pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=512,
        )
    else:
        hf_pipe = pipeline(
            "text2text-generation",
            model="UBC-NLP/AraT5-base-title-generation",
            max_length=512,
        )

    llm = HuggingFacePipeline(pipeline=hf_pipe)

    if lang == "en":
        prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Helpful Answer:"""
    else:
        prompt_template = """أنت مساعد ذكي.
استخدم النص التالي للإجابة على السؤال باللغة العربية.

النص: {context}
السؤال: {question}

الإجابة:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )
    return qa


# Timestamp formatting

def format_time(seconds):
    minutes = math.floor(seconds / 60)
    seconds = math.floor(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


# ---------------------- Streamlit UI ----------------------

st.title("YouTube Summarizer + RAG Q&A (Arabic & English) by soultan yousif ")
st.write("paste a YouTube link, get a summary with timestampes and ask questions.")

col1, col2 = st.columns([3, 1])

with col1:
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    fetch_button = st.button("Fetch & Summarize")

with col2:
    st.markdown("**Options**")
    show_transcript = st.checkbox("Show transcript segments", value=False)
    k_sources = st.slider("# of retrieved docs (k)", 1, 6, 3)

if fetch_button and url:
    with st.spinner("Fetching transcript and building summary..."):
        try:
            full_text, segments = extract_transcript(url)
        except Exception as e:
            st.error(f"Could not fetch transcript: {e}")
            st.stop()

        lang = detect_lang(full_text)
        summary = summarise_text(full_text, lang)

        st.subheader("Summary")
        st.write(summary)

        docs = chunk_transcript(segments)
        st.write(f"Transcript chunked into {len(docs)} documents.")

        # store docs & lang in session_state for Q&A
        st.session_state['docs'] = docs
        st.session_state['lang'] = lang

        if show_transcript:
            st.subheader("Transcript segments (first 20 shown)")
            for seg in segments[:20]:
                t = format_time(seg['start'])
                st.markdown(f"**[{t}]** {seg['text']}")


st.markdown("---")

st.subheader("Ask a question about the video")
question = st.text_input("Your question")
ask_button = st.button("Ask")

if ask_button:
    if 'docs' not in st.session_state:
        st.error("You must fetch & summarize a video first.")
    elif not question.strip():
        st.error("Write a question first.")
    else:
        with st.spinner("Building RAG and fetching answer..."):
            try:
                qa = build_rag(st.session_state['docs'], st.session_state['lang'])
                result = qa({"query": question})
            except Exception as e:
                st.error(f"Error building or running RAG: {e}")
                st.stop()

        st.subheader("Answer")
        st.write(result.get('result') or result.get('answer') or "No answer returned")

        st.subheader("Sources")
        for doc in result.get('source_documents', []):
            start_time = doc.metadata.get('start', 0)
            st.markdown(f"**{format_time(start_time)}** — {doc.page_content[:400]}...")


st.markdown("---")
st.caption("This app uses YouTubeTranscriptApi for transcript retrieval and local HuggingFace models for summarization & langchain RAG. Make sure required models are installed and you have enough RAM/CPU.")
