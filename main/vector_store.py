import os
import pickle
import numpy as np
import faiss
from django.conf import settings

# Paths
VECTOR_DIR = os.path.join(settings.BASE_DIR, "vector_store")
INDEX_PATH = os.path.join(VECTOR_DIR, "faiss.index")
META_PATH = os.path.join(VECTOR_DIR, "metadata.pkl")


def ensure_dir():
    os.makedirs(VECTOR_DIR, exist_ok=True)


def load_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return None


def load_metadata():
    if os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            return pickle.load(f)
    return []


def save_index(index):
    ensure_dir()
    faiss.write_index(index, INDEX_PATH)


def save_metadata(metadata):
    ensure_dir()
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)


# ----------------------------
# ADD EMBEDDINGS (Incremental)
# ----------------------------
def add_embeddings(book_id, embeddings, chunks):
    """
    Add new embeddings + chunks to existing FAISS index
    """
    ensure_dir()

    # Convert embeddings to np.float32 and 2D array
    embeddings = np.array(embeddings, dtype="float32")

    # Load index & metadata
    index = load_index()
    metadata = load_metadata()

    # If no index exists, create new
    if index is None:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
    else:
        # Check dimension match
        if embeddings.shape[1] != index.d:
            raise ValueError(
                f"Dimension mismatch: embeddings {embeddings.shape[1]}, index {index.d}"
            )

    # Add embeddings
    index.add(embeddings)

    # Add metadata
    for chunk in chunks:
        metadata.append({"book_id": book_id, "text": chunk})

    # Save
    save_index(index)
    save_metadata(metadata)


# ----------------------------
# SEARCH
# ----------------------------
def search(query_embedding, k=5):
    """
    Search FAISS index for top-k relevant chunks
    """
    if not os.path.exists(INDEX_PATH):
        return None

    index = faiss.read_index(INDEX_PATH)
    metadata = load_metadata()

    # Convert query to correct type & shape
    query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)

    # Dimension check
    if query_embedding.shape[1] != index.d:
        raise ValueError(
            f"Dimension mismatch: query {query_embedding.shape[1]}, index {index.d}"
        )

    # Perform search
    D, I = index.search(query_embedding, k)

    # Simple threshold filter (optional)
    if D[0][0] > 1.2:
        return None

    results = [metadata[idx]["text"] for idx in I[0]]
    return results


# ----------------------------
# DELETE EMBEDDINGS BY BOOK
# ----------------------------
def delete_book_embeddings(book_id, get_embedding_func):
    """
    Remove all embeddings of a specific book by rebuilding the index
    get_embedding_func = function to generate embedding from text
    """
    metadata = load_metadata()

    # Filter out book
    filtered = [item for item in metadata if item["book_id"] != book_id]

    # If nothing remains, remove files
    if not filtered:
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        if os.path.exists(META_PATH):
            os.remove(META_PATH)
        return

    # Rebuild embeddings
    texts = [item["text"] for item in filtered]
    embeddings = [get_embedding_func(text) for text in texts]
    embeddings = np.array(embeddings, dtype="float32")

    # Create new index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save
    save_index(index)
    save_metadata(filtered)


# ----------------------------
# REBUILD INDEX (Optional management command)
# ----------------------------
def rebuild_index(all_books, extract_text_func, split_text_func, get_embedding_func):
    """
    Rebuild the entire FAISS index from scratch
    """
    all_chunks = []
    all_embeddings = []

    for book in all_books:
        text = extract_text_func(book.file.path)
        chunks = split_text_func(text)

        for chunk in chunks:
            embedding = get_embedding_func(chunk)
            all_embeddings.append(embedding)
            all_chunks.append({"book_id": book.id, "text": chunk})

    if not all_embeddings:
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        if os.path.exists(META_PATH):
            os.remove(META_PATH)
        return

    embeddings = np.array(all_embeddings, dtype="float32")
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    save_index(index)
    save_metadata(all_chunks)