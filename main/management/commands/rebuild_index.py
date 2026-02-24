from django.core.management.base import BaseCommand
from main.models import Book
from main.document_processor import extract_text, split_text
from main.embedding_service import get_embedding
from main.vector_store import rebuild_index, save_index, save_metadata
import faiss
import numpy as np

class Command(BaseCommand):
    help = "Rebuild FAISS index from scratch"

    def handle(self, *args, **kwargs):
        books = Book.objects.all()

        all_chunks = []
        all_embeddings = []

        for book in books:
            text = extract_text(book.file.path)
            chunks = split_text(text)

            for chunk in chunks:
                embedding = get_embedding(chunk)
                all_embeddings.append(embedding)
                all_chunks.append({
                    "book_id": book.id,
                    "text": chunk
                })

        if not all_embeddings:
            self.stdout.write("No books found.")
            return

        dimension = len(all_embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(all_embeddings).astype("float32"))

        save_index(index)
        save_metadata(all_chunks)

        self.stdout.write("Index rebuilt successfully.")


# class Command(BaseCommand):
#     help = "Rebuild the FAISS vector store for all books with overlapping chunks"

#     def add_arguments(self, parser):
#         parser.add_argument(
#             "--chunk_size",
#             type=int,
#             default=300,
#             help="Number of words per chunk (default: 300)"
#         )
#         parser.add_argument(
#             "--overlap",
#             type=int,
#             default=50,
#             help="Number of overlapping words between consecutive chunks (default: 50)"
#         )

#     def handle(self, *args, **options):
#         chunk_size = options["chunk_size"]
#         overlap = options["overlap"]

#         all_books = Book.objects.all()
#         if not all_books.exists():
#             self.stdout.write(self.style.WARNING("No books found to rebuild index."))
#             return

#         self.stdout.write(
#             self.style.SUCCESS(f"Rebuilding FAISS index for {all_books.count()} books...")
#         )

#         # Rebuild the index using the improved pipeline
#         rebuild_index(
#             all_books=all_books,
#             extract_text_func=extract_text,
#             chunk_size=chunk_size,
#             overlap=overlap
#         )

#         self.stdout.write(self.style.SUCCESS("FAISS index successfully rebuilt!"))