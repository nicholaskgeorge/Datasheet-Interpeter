"""
CLI script to test PDF QA pipeline from the command line.
Usage:
  python test_qa.py /path/to/file.pdf "Your question here"
"""
import sys
from PyPDF2 import PdfReader
from transformers import pipeline

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_qa.py <pdf_path> <question>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    question = sys.argv[2]
    # Read PDF
    reader = PdfReader(pdf_path)
    context = ""
    for page in reader.pages:
        context += page.extract_text() or ""
    # Initialize QA pipeline
    qa = pipeline('question-answering')
    # Run QA
    result = qa(question=question, context=context)
    print(f"Answer: {result.get('answer')}")

if __name__ == '__main__':
    main()
