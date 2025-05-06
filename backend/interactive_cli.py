#!/usr/bin/env python3
"""
Interactive CLI for PDF QA pipeline.
Usage: python interactive_cli.py <pdf_path>
"""
import sys
from PyPDF2 import PdfReader
from transformers import pipeline

def load_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def main():
    if len(sys.argv) < 2:
        print("Usage: python interactive_cli.py <pdf_path>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    print(f"Loading PDF: {pdf_path}")
    context = load_pdf_text(pdf_path)
    qa = pipeline('question-answering')
    print("PDF loaded. Enter your question below (type 'exit' to quit):")
    while True:
        question = input(">> ").strip()
        if not question or question.lower() in ('exit', 'quit'):
            print("Exiting CLI. Goodbye!")
            break
        try:
            result = qa(question=question, context=context)
            answer = result.get('answer', 'No answer found.')
        except Exception as e:
            answer = f"Error: {str(e)}"
        print(f"Answer: {answer}\n")

if __name__ == '__main__':
    main()
