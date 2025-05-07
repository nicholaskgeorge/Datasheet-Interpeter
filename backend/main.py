from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
from PyPDF2 import PdfReader  # fallback if needed
from langchain.document_loaders import PyPDFLoader
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering
import os
import torch
from huggingface_hub import login, InferenceApi
import requests
import traceback  # for detailed error logs
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from pdf2image import convert_from_path
from PIL import Image
import re  # for question pattern matching

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the uploaded PDF path
pdf_path = None

# System prompt for controlling response style
SYSTEM_PROMPT = "You are a datasheet assistant. Answer concisely and only use information from the datasheet, including appropriate units when applicable. When possible cite where the information came from."

# Load Hugging Face token from env
HF_TOKEN = os.getenv("HF_HUB_TOKEN", "").strip()
## Remove potential 'Bearer ' prefix if present
if HF_TOKEN.lower().startswith("bearer "):
    HF_TOKEN = HF_TOKEN.split(" ", 1)[1]

inference_llama = None
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("[startup] HF login successful")
    except Exception as e:
        print(f"[startup] HF login failed: {e}")
        print(traceback.format_exc())
    try:
        inference_llama = InferenceApi(
            repo_id="meta-llama/Llama-2-7b-chat-hf",
            token=HF_TOKEN
        )
        print("[startup] inference_llama client created for slug meta-llama/Llama-2-7b-chat-hf")
    except Exception as e:
        inference_llama = None
        print(f"[startup] inference_llama init failed: {e}")
        print(traceback.format_exc())
else:
    print("[startup] no HF_TOKEN: inference_llama None")

## Initialize Flan-T5-Small pipeline locally for fast inference
small_pipeline = None
try:
    small_pipeline = pipeline(
        'text2text-generation',
        model='google/flan-t5-small',
        device_map='auto'
    )
    print("[startup] small_pipeline created for google/flan-t5-small")
except Exception as e:
    print(f"[startup] small_pipeline init failed: {e}")
    small_pipeline = None

# Initialize HuggingFace question-answering pipeline
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

## Initialize LLaMA pipeline with detailed logging
llama_pipeline = None
tokenizer_llama = None
try:
    tokenizer_llama = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', use_fast=True)
    print("[startup] AutoTokenizer loaded successfully")
    model_llama = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-2-7b-chat-hf',
        device_map='auto',
        torch_dtype=torch.float16
    )
    llama_pipeline = pipeline('text-generation', model=model_llama, tokenizer=tokenizer_llama)
    print("[startup] LLaMA pipeline created successfully")
except Exception as e:
    print(f"[startup] LLaMA init failed (FP16): {e}")
    print(traceback.format_exc())
    llama_pipeline = None

# Initialize Visual QA pipeline (BLIP/Vilt)
vqa_pipeline = None
try:
    vqa_pipeline = pipeline(
        "visual-question-answering",
        model="dandelin/vilt-b32-finetuned-vqa",
        device_map="auto"
    )
    print("[startup] vqa_pipeline created for dandelin/vilt-b32-finetuned-vqa")
except Exception as e:
    vqa_pipeline = None
    print(f"[startup] vqa_pipeline init failed: {e}")

## Initialize LayoutLMv3 pipeline for document QA
layout_processor = None
layout_pipeline = None
try:
    layout_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
    layout_model = LayoutLMv3ForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-base")
    layout_pipeline = pipeline(
        "document-question-answering",
        model=layout_model,
        processor=layout_processor,
        device_map="auto"
    )
    print("[startup] layoutlmv3 pipeline created for microsoft/layoutlmv3-base")
except Exception as e:
    layout_pipeline = None
    print(f"[startup] layoutlmv3 init failed: {e}")

# Initialize RAG components
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
qdrant_client = QdrantClient(url="localhost", prefer_grpc=True)
vectordb = None

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        pdf_path = tmp.name
    return {"status": "success", "pdf_path": pdf_path}

@app.post("/index_pdf/")
async def index_pdf():
    global vectordb
    if not pdf_path:
        return JSONResponse({"error": "No PDF uploaded."}, status_code=400)
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    vectordb = Qdrant.from_documents(
        split_docs,
        embeddings,
        url="http://localhost:6333",
        prefer_grpc=True,
        collection_name="datasheet"
    )
    return {"status": "indexed", "n_chunks": len(split_docs)}

@app.post("/ask")
@app.post("/ask/")
async def ask_question(
    question: str = Form(...),
    model: str = Form("distilbert")
):
    # Debug: log incoming request
    print(f"[ask_question] Received question={question}, model={model}")
    print(f"[ask_question] llama_pipeline available: {llama_pipeline is not None}, inference_llama available: {inference_llama is not None}")
    if not pdf_path:
        return JSONResponse({"answer": "No PDF uploaded."}, status_code=400)
    # Try LayoutLMv3 for unified document QA
    if layout_pipeline:
        try:
            pages = convert_from_path(pdf_path)
            best_answer, best_score = "", -1.0
            for page in pages:
                res = layout_pipeline({"image": page, "question": question})
                if res and res[0].get("score", 0) > best_score:
                    best_score = res[0]["score"]
                    best_answer = res[0].get("answer", "")
            return JSONResponse({"answer": best_answer or "No answer found."})
        except Exception as e:
            print(f"[ask_question] layoutlmv3 error: {e}")
    # Simple fallback for device name queries
    if re.search(r"name of.*device|device name|model number|part number", question, re.I):
        try:
            reader = PdfReader(pdf_path)
            text0 = reader.pages[0].extract_text() or ""
            for line in text0.splitlines():
                if re.search(r"[A-Za-z]+\d+", line):
                    return JSONResponse({"answer": line.strip()})
        except Exception as e:
            print(f"[ask_question] device name fallback failed: {e}")
    # RAG context retrieval
    if vectordb:
        retrieved_docs = vectordb.similarity_search(question, k=5)
        print(f"[ask_question] Retrieved {len(retrieved_docs)} chunks from vector store")
        full_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    else:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(f"[ask_question] LangChain loader returned {len(docs)} docs")
        full_text = ""
        for idx, doc in enumerate(docs):
            page_content = doc.page_content or ""
            print(f"[ask_question] doc {idx+1}/{len(docs)} snippet: {page_content[:200]!r}")
            full_text += page_content
    print(f"[ask_question] context length={len(full_text)}")
    print(f"[ask_question] context snippet: {full_text[:500]!r}")
    # Unified QA: combine text (RAG) and visual QA, auto-select by confidence
    try:
        res_t = qa_pipeline(question=question, context=full_text)
        text_answer = res_t.get("answer", "")
        text_score = res_t.get("score", 0.0)
    except Exception:
        text_answer, text_score = "", 0.0
    best_visual_answer, best_visual_score = "", -1.0
    if vqa_pipeline:
        try:
            pages = convert_from_path(pdf_path)
            for page in pages:
                res_v = vqa_pipeline({"image": page, "question": question})
                if res_v and res_v[0].get("score", 0) > best_visual_score:
                    best_visual_score = res_v[0]["score"]
                    best_visual_answer = res_v[0].get("answer", "")
        except Exception:
            pass
    answer = best_visual_answer if best_visual_score > text_score else text_answer
    # Fallback if answer is empty
    if not answer.strip():
        answer = "No answer found."
    # Debug log
    print(f"[ask_question] model={model}, question={question}, answer={answer}")
    return JSONResponse({"answer": answer})
