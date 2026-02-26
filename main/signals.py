from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from .models import Book
from .document_processor import extract_text, split_text
from .embedding_service import get_embedding
from .vector_store import add_embeddings, delete_book_embeddings, save_index

@receiver(post_save, sender=Book)
def index_book(sender, instance, created, **kwargs):
    if created:
        text = extract_text(instance.file.path)
        print("File path:", instance.file.path)
        print("Extracted text length:", len(text))
        chunks = split_text(text)

        embeddings = [get_embedding(chunk) for chunk in chunks]
        print("Embedding length:", len(embeddings))

        add_embeddings(instance.id, embeddings, chunks)


@receiver(post_delete, sender=Book)
def remove_book_from_index(sender, instance, **kwargs):
    delete_book_embeddings(instance.id)