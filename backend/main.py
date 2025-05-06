from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
from PyPDF2 import PdfReader
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os
import torch
from huggingface_hub import login, InferenceApi
import requests
import traceback  # for detailed error logs

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

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        pdf_path = tmp.name
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
    # Extract text from PDF
    reader = PdfReader(pdf_path)
    # Extract full text from PDF
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""
    # Branch based on model selection
    if model == "distilbert":
        prompt_question = f"{SYSTEM_PROMPT}\n{question}"
        try:
            result = qa_pipeline(question=prompt_question, context=full_text)
            answer = result.get("answer", "Could not find an answer.")
        except Exception as e:
            answer = f"Error during QA: {str(e)}"
    elif model == "llama":
        # Truncate context to avoid large payloads
        max_chars = 2000
        llama_context = full_text if len(full_text) <= max_chars else full_text[:max_chars]
        prompt = f"{SYSTEM_PROMPT}\nContext:\n{llama_context}\n\nQuestion: {question}\nAnswer:"
        if not llama_pipeline:
            answer = "LLaMA pipeline not available."
        else:
            try:
                outputs = llama_pipeline(prompt, max_new_tokens=128)
                gen_text = outputs[0].get("generated_text", "")
                answer = gen_text.split("Answer:")[-1].strip()
            except Exception as e:
                answer = f"Error during LLaMA generation: {e}"
    elif model == "visual":
        answer = "Visual QA is not yet implemented."
    else:
        answer = "Unknown model."
    # Fallback if answer is empty
    if not answer.strip():
        answer = "No answer found."
    # Debug log
    print(f"[ask_question] model={model}, question={question}, answer={answer}")
    return JSONResponse({"answer": answer})
