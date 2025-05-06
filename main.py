import argparse
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from typing import List

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=True).tolist()
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], show_progress_bar=False)[0].tolist()
    def __call__(self, text: str) -> List[float]:
        """Allow the embedder instance to be called directly for query embeddings."""
        return self.embed_query(text)

def load_pdf(path: str) -> str:
    """Extract all text from a PDF file."""
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                texts.append(page_text)
    return "\n".join(texts)


def main():
    parser = argparse.ArgumentParser(description="Basic PDF Q&A with RAG")
    parser.add_argument("pdf_path", help="Path to the datasheet PDF")
    args = parser.parse_args()

    print(f"Loading PDF from {args.pdf_path}...")
    raw_text = load_pdf(args.pdf_path)

    print("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(raw_text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    print("Indexing embeddings in FAISS...")
    embed_model = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embed_model)

    print("Loading LLM pipeline (Flan-T5)...")
    hf_pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_length=256,
        temperature=0
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )

    print("Ready for questions (type 'exit' to quit)")
    while True:
        query = input("Q: ")
        if query.lower().startswith("exit"):
            break
        answer = qa.run(query)
        print(f"A: {answer}\n")


if __name__ == "__main__":
    main()
