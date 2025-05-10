import os
import json
import torch
import sys  # for exit on error
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
from pypdf.errors import PdfReadError

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
# Path to folder containing PDFs
pdf_folder_path = "/path/to/pdf_folder"
# Output dataset path
dataset_path = "all_qa.txt"
# HF model name (Llama 8B variant)
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "meta-llama/Meta-Llama-3-8B")
# Using Meta Llama 3 (8B params) for QA generation

# Splitting & Q params
CHUNK_SIZE  = 1000
OVERLAP     = 200
Q_TOTAL     = 50  # total Q&A pairs desired
CHUNK_Q     = 5   # Q&A to generate per chunk

# Device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ─── LOAD TOKENIZER & MODEL ─────────────────────────────────────────────────────
print(f"Loading model {LLAMA_MODEL} with 4-bit NF4 quantization…")
tokenizer = AutoTokenizer.from_pretrained(
    LLAMA_MODEL,
    use_fast=True,
    use_auth_token=os.getenv("HUGGING_FACE_HUB_TOKEN"),
)
# load quantized model onto GPU with minimal VRAM usage
if "Meta-Llama" in LLAMA_MODEL:
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL,
        device_map="auto",
        load_in_8bit=True,
        torch_dtype=torch.float16,
        use_auth_token=os.getenv("HUGGING_FACE_HUB_TOKEN"),
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL,
        device_map="auto",
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        use_auth_token=os.getenv("HUGGING_FACE_HUB_TOKEN"),
    )
llama_pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer,
    return_full_text=False  # only return generated tokens, not the prompt
)
# Debug: test simple prompt
print("=== DEBUG SINGLE PROMPT ===")
test_out = llama_pipe("What is the purpose of the LM555 timer?", max_new_tokens=64, do_sample=False)
print("DEBUG OUT:", test_out)

def is_valid_chunk(text: str) -> bool:
    """
    Skip chunks that are mostly tables or TOC based on dot sequences or all-caps lines.
    """
    lines = text.splitlines()
    toc_lines = sum(1 for l in lines if re.match(r".*\.\.\.+", l) or l.isupper())
    return toc_lines < len(lines) * 0.5

def sanitize_text(text: str) -> str:
    """
    Remove non-printable and non-ASCII characters from text.
    """
    # Keep ASCII printable characters (space to tilde) and newlines
    return re.sub(r'[^\x20-\x7E\n]', '', text)

# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main(pdf_path, out_path):
    # 0) prepare dataset list
    dataset_records = []
    # 1) load & split PDF
    print("Loading and splitting PDF…")
    loader = PyPDFLoader(pdf_path)
    try:
        docs = loader.load()
    except PdfReadError as e:
        print(f"❗ Error reading {pdf_path}: {e}")
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(chunks)} chunks from PDF.")

    # 2) generate across chunks
    qas_all = []
    seen_questions = set()
    for idx, chunk in enumerate(chunks):
        if len(qas_all) >= Q_TOTAL:
            break
        text = chunk.page_content
        print(f"\n--- Chunk {idx+1}/{len(chunks)} preview (first 200 chars) ---")
        print(text[:200] + "...\n")
        if not is_valid_chunk(text):
            print(f"❗ Skipping chunk {idx+1}/{len(chunks)} due to invalid content")
            continue
        # remove legal/trademark footers, URLs, and bullet lists
        lines = text.split("\n")
        text = "\n".join(
            [l for l in lines if not any(pat in l.lower() for pat in [
                "all rights reserved", "trademark", "www.", "http://", "https://"
            ])]
        )
        # remove bullet list lines
        text_lines = text.split("\n")
        text = "\n".join([l for l in text_lines if not l.strip().startswith("•")])
        # strip non-standard symbols
        text = sanitize_text(text)
        remaining = Q_TOTAL - len(qas_all)
        per_chunk = min(CHUNK_Q, remaining)

        prompt = (
            "You are an expert datasheet QA generator.\n"
            "Below is an excerpt of a datasheet delimited by triple backticks:\n"
            f"```{text}```\n\n"
            f"Generate {per_chunk} technical unique question-answer pairs based on the excerpt above.\n"
            "Format each pair as:\n"
            "Q: <question>\n"
            "A: <answer>\n\n"
            "Do NOT output JSON or any other surrounding text.\n"
        )
        print(f"\n{'='*10} CHUNK {idx+1}/{len(chunks)} PROMPT {'='*10}")
        print(prompt)

        outputs = llama_pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        raw = outputs[0]["generated_text"]
        print("\nRAW MODEL OUTPUT:\n", raw)

        # parse Q/A pairs in Q: A: format
        lines_out = [l.strip() for l in raw.strip().splitlines() if l.strip()]
        qas = []
        i = 0
        while i < len(lines_out):
            if lines_out[i].startswith("Q:"):
                q = lines_out[i][2:].strip()
                a = ""
                if i+1 < len(lines_out) and lines_out[i+1].startswith("A:"):
                    a = lines_out[i+1][2:].strip()
                qas.append({"question": q, "answer": a})
                i += 2
            else:
                i += 1
        # filter and dedupe
        unique_qas = []
        for qa in qas:
            qstr = qa["question"]
            if not qstr.endswith('?') or len(qstr.split()) < 5:
                continue
            if qstr.lower() in seen_questions:
                continue
            seen_questions.add(qstr.lower())
            unique_qas.append(qa)
            if len(unique_qas) >= per_chunk:
                break
        # store QA records
        for qa in unique_qas:
            dataset_records.append({
                "chunk_id": idx,
                "context": text,
                "question": qa["question"],
                "answer": qa["answer"]
            })
        qas_all.extend(unique_qas)

    # Return Q/A list for this PDF
    if not dataset_records:
        print(f"❗ No Q&A generated for {pdf_path}")
        return []
    return dataset_records

if __name__ == "__main__":
    source = pdf_folder_path
    out_path = dataset_path

    if os.path.isdir(source):
        pdf_list = [os.path.join(source, f) for f in os.listdir(source) if f.lower().endswith('.pdf')]
    else:
        pdf_list = [source]
    all_records = []
    for pdf in pdf_list:
        print(f"Processing {pdf}...")
        recs = main(pdf, out_path)
        all_records.extend(recs)
    # Save all Q/A with context to one text file
    with open(out_path, 'w') as out:
        for rec in all_records:
            out.write(f"Context: {rec['context']}\nQ: {rec['question']}\nA: {rec['answer']}\n\n")
    print(f"Saved {len(all_records)} Q/A pairs with context to {out_path}")
