from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os
import torch
from huggingface_hub import login, InferenceApi
import requests
import traceback  # for detailed error logs
from pdf2image import convert_from_path
from PIL import Image
import re  # for question pattern matching

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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

## Initialize HuggingFace question-answering pipeline
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

## RAG vectorstore placeholder
vectorstore = None
def index_pdf(path: str):
    global vectorstore
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    # Use a text-only embedding model to avoid requiring bounding boxes
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"[startup] Indexed {len(chunks)} chunks from {path}")

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        pdf_path = tmp.name
    print(f"[upload_pdf] Saved PDF to {pdf_path}, starting RAG indexing")
    index_pdf(pdf_path)
    return {"status": "success", "pdf_path": pdf_path}

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
    # Fallback full document text via LangChain loader
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"[ask_question] LangChain loader returned {len(docs)} docs")
    full_text = "\n\n".join(doc.page_content or "" for doc in docs)
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
    if model == "distilbert":
        pass
    elif model == "small":
        try:
            outputs = small_pipeline(question, max_new_tokens=128)
            answer = outputs[0].get('generated_text', '').strip()
        except Exception as e:
            answer = f"Error during small pipeline generation: {e}"
    elif model == "rag":
        # LayoutLM-based Retrieval-Augmented Generation
        if vectorstore is None:
            answer = "No RAG index available. Upload a PDF first."
        else:
            docs = vectorstore.similarity_search(question, k=5)
            print(f"[ask_question] Retrieved {len(docs)} chunks for RAG")
            context = "\n\n".join(doc.page_content or "" for doc in docs)
            prompt = f"{SYSTEM_PROMPT}\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
            print(f"[ask_question] RAG prompt snippet: {prompt[:500]!r}")
            try:
                outputs = small_pipeline(prompt, max_new_tokens=128)
                answer = outputs[0].get('generated_text', '').strip()
            except Exception as e:
                answer = f"Error during RAG generation: {e}"
    elif model == "llama":
        if inference_llama:
            try:
                answer = inference_llama(
                    inputs={"text": question, "max_new_tokens": 128},
                    params={"return_full_text": False}
                )["generated_text"]
            except Exception as e:
                answer = f"Error during LLaMA generation: {e}"
        elif llama_pipeline:
            try:
                outputs = llama_pipeline(question, max_new_tokens=128)
                answer = outputs[0].get('generated_text', '').strip()
            except Exception as e:
                answer = f"Error during LLaMA generation: {e}"
        else:
            answer = "LLaMA pipeline not available."
    return JSONResponse({"answer": answer})
