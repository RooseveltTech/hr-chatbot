# from django.core.management.base import BaseCommand
# from main.models import Book
# from main.document_processor import clean_text, extract_text, split_text
# from main.embedding_service import get_embedding
# from main.vector_store import rebuild_index, save_index, save_metadata
# import faiss
# import numpy as np

# class Command(BaseCommand):
#     help = "Rebuild FAISS index from scratch"

#     def handle(self, *args, **kwargs):
#         books = Book.objects.all()

#         all_chunks = []
#         all_embeddings = []

#         for book in books:
#             text = extract_text(book.file.path)
#             text = clean_text(text)
#             chunks = split_text(text)

#             for chunk in chunks:
#                 embedding = get_embedding(chunk)
#                 all_embeddings.append(embedding)
#                 all_chunks.append({
#                     "book_id": book.id,
#                     "text": chunk
#                 })

#         if not all_embeddings:
#             self.stdout.write("No books found.")
#             return

#         dimension = len(all_embeddings[0])
#         index = faiss.IndexFlatL2(dimension)
#         index.add(np.array(all_embeddings).astype("float32"))

#         save_index(index)
#         save_metadata(all_chunks)

#         self.stdout.write("Index rebuilt successfully.")


import os
import numpy as np
import faiss
from django.core.management.base import BaseCommand
from django.conf import settings

from main.models import Book
from main.document_processor import clean_text, extract_pdf_ocr, extract_text, split_text
from main.embedding_service import get_embedding
from main.vector_store import (
    save_index,
    save_metadata,
)

VECTOR_DIR = os.path.join(settings.BASE_DIR, "vector_store")
INDEX_PATH = os.path.join(VECTOR_DIR, "faiss.index")
META_PATH = os.path.join(VECTOR_DIR, "metadata.pkl")


class Command(BaseCommand):
    help = "Rebuild FAISS index from scratch (all books, clean text, overlapping chunks)"

    def handle(self, *args, **kwargs):

        self.stdout.write(self.style.WARNING("Starting full index rebuild..."))

        books = Book.objects.all()
        if not books.exists():
            self.stdout.write(self.style.ERROR("No books found in database."))
            return

        all_embeddings = []
        all_metadata = []

        for book in books:
            self.stdout.write(self.style.SUCCESS(f"Processing book: {book.id}"))

            try:
                # 1️⃣ Extract + Clean
                text = extract_text(book.file.path)
                print("RAW TEXT LENGTH:", len(text))
                text = clean_text(text)
                if not text.strip():
                    # fallback to OCR
                    self.stdout.write(f"Falling back to OCR for book {book.id}")
                    text = extract_pdf_ocr(book.file.path)
                    text = clean_text(text)

                print("CLEANED TEXT LENGTH:", len(text))

                if not text or len(text.strip()) < 50:
                    print("TEXT EMPTY OR TOO SHORT")
                    # self.stdout.write(self.style.WARNING(
                    #     f"Skipping book {book.id} (no valid text)"
                    # ))
                    # continue
                    # text = extract_pdf_ocr(book.file.path)
                    if not text.strip():
                        self.stdout.write(self.style.WARNING(f"Skipping book {book.id} — no text after OCR"))
                        continue

                # 2️⃣ Split into overlapping chunks
                chunks = split_text(text, chunk_size=500, overlap=100)
                print("CHUNKS CREATED:", len(chunks))

                # 3️⃣ Embed each chunk
                for chunk in chunks:
                    embedding = get_embedding(chunk)

                    if embedding is None or len(embedding) == 0:
                        continue

                    all_embeddings.append(embedding)
                    all_metadata.append({
                        "book_id": book.id,
                        "book_title": getattr(book, "title", ""),
                        "text": chunk
                    })

            except Exception as e:
                self.stdout.write(self.style.ERROR(
                    f"Error processing book {book.id}: {str(e)}"
                ))

        if not all_embeddings:
            self.stdout.write(self.style.ERROR("No embeddings generated."))
            return

        # 4️⃣ Build FAISS index
        embeddings_array = np.array(all_embeddings, dtype="float32")
        dimension = embeddings_array.shape[1]

        self.stdout.write(self.style.SUCCESS(
            f"Creating FAISS index with dimension {dimension}"
        ))

        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)

        # 5️⃣ Ensure vector_store directory exists
        os.makedirs(VECTOR_DIR, exist_ok=True)

        # 6️⃣ Save index and metadata
        save_index(index)
        save_metadata(all_metadata)

        self.stdout.write(self.style.SUCCESS(
            f"Rebuild complete. Total vectors: {index.ntotal}"
        ))