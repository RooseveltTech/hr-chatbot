from PyPDF2 import PdfReader
import pdfplumber
import re 
from rank_bm25 import BM25Okapi
import os
from pdf2image import convert_from_path
import pytesseract

def extract_pdf_ocr(path):
    text = ""
    try:
        pages = convert_from_path(path)
        for i, page_image in enumerate(pages):
            page_text = pytesseract.image_to_string(page_image)
            if page_text.strip():
                text += page_text + "\n"
    except Exception as e:
        print(f"OCR failed for {path}: {e}")
    return text

def extract_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# def extract_text(path):
#     full_text = ""

#     with pdfplumber.open(path) as pdf:
#         for i, page in enumerate(pdf.pages):
#             try:
#                 page_text = page.extract_text()
#                 if not page_text:
#                     continue
#                 # Remove very short noisy pages
#                 if len(page_text.strip()) < 30:
#                     continue
#                 full_text += page_text + "\n"
#             except Exception as e:
#                 print(f"Warning: Skipping page {i+1} in {os.path.basename(path)} due to: {e}")
#                 continue

#     return full_text

# def split_text(text, chunk_size=500):
#     return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
# def split_text(text, chunk_size=500, overlap=100):
#     words = text.split()
#     chunks = []

#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = " ".join(words[i:i + chunk_size])

#         # Skip garbage chunks
#         if len(chunk.strip()) < 50:
#             continue

#         chunks.append(chunk)

#     return chunks
def split_text(text, chunk_size=400, overlap=80):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        if len(chunk.strip()) > 50:  # skip tiny garbage chunks
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks

# def clean_text(text):
#     # Remove excessive newlines
#     text = re.sub(r'\n+', '\n', text)

#     # Remove isolated single-character newlines
#     text = re.sub(r'\n\s*\n', '\n', text)

#     # Fix word breaks caused by line splits
#     text = text.replace('\n', ' ')

#     # Remove multiple spaces
#     text = re.sub(r'\s+', ' ', text)

#     return text.strip()
# --------------------------------
# CLEANING PIPELINE
# --------------------------------
def clean_text(text):
    if not text:
        return ""

    # Remove repeated headers like "Employee Handbook LIBERTY ASSURED"
    text = remove_repeated_lines(text)

    # Fix broken word splits caused by PDF line wrapping
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Remove page numbers alone
    text = re.sub(r"\b\d+\b(?=\s)", "", text)

    return text.strip()

# --------------------------------
# REMOVE REPEATED HEADERS / FOOTERS
# --------------------------------
def remove_repeated_lines(text):
    lines = text.split("\n")

    line_counts = {}
    for line in lines:
        stripped = line.strip()
        if stripped:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1

    # Remove lines repeated more than 3 times (likely headers)
    cleaned = [
        line for line in lines
        if line_counts.get(line.strip(), 0) <= 3
    ]

    return "\n".join(cleaned)
    
def build_bm25(metadata):
    """
    Build BM25 index from metadata chunks.
    """
    corpus = [item["text"] for item in metadata]
    tokenized_corpus = [doc.split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)

