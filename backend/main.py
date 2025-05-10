from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber, io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "ok"}

# Global RetrievalQA chain
qa_chain = None

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()
    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()
    def __call__(self, text):
        return self.embed_query(text)

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global qa_chain
    content = await file.read()
    try:
        pdf = pdfplumber.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid PDF: {e}")
    text = ""
    for page in pdf.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    pdf.close()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedder)
    hf_pipe = hf_pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_length=256,
        temperature=0
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return {"status": "success", "chunks": len(texts)}

@app.post("/ask/")
async def ask(question: str = Form(...), model: str = Form(None)):
    global qa_chain
    if qa_chain is None:
        raise HTTPException(status_code=400, detail="No PDF uploaded")
    result = qa_chain(question)
    if isinstance(result, dict):
        answer = result.get("result", "")
        source_docs = result.get("source_documents", [])
        sources = [{"content": d.page_content} for d in source_docs]
    else:
        answer = result
        sources = []
    return {"answer": answer, "sources": sources}