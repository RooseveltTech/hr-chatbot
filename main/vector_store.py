import os
import pickle
import numpy as np
import faiss
from django.conf import settings

from main.document_processor import build_bm25
from main.embedding_service import get_embedding
from rank_bm25 import BM25Okapi

# Paths
VECTOR_DIR = os.path.join(settings.BASE_DIR, "vector_store")
INDEX_PATH = os.path.join(VECTOR_DIR, "faiss.index")
META_PATH = os.path.join(VECTOR_DIR, "metadata.pkl")

_index = None
_metadata = None
_bm25 = None

def load_metadata():
    if os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            return pickle.load(f)
    return []

def load_resources():
    global _index, _metadata, _bm25

    if _index is None:
        _index = faiss.read_index(INDEX_PATH)

    if _metadata is None:
        _metadata = load_metadata()

    if _bm25 is None:
        corpus = [item["text"] for item in _metadata]
        tokenized = [doc.split() for doc in corpus]
        _bm25 = BM25Okapi(tokenized)

    return _index, _metadata, _bm25

_index, _metadata, _bm25 = load_resources()

def ensure_dir():
    os.makedirs(VECTOR_DIR, exist_ok=True)


def load_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return None

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
# def search(query_embedding, k=5):
#     """
#     Search FAISS index for top-k relevant chunks
#     """
#     if not os.path.exists(INDEX_PATH):
#         return None

#     index = faiss.read_index(INDEX_PATH)
#     metadata = load_metadata()

#     # Convert query to correct type & shape
#     query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)

#     # Dimension check
#     if query_embedding.shape[1] != index.d:
#         raise ValueError(
#             f"Dimension mismatch: query {query_embedding.shape[1]}, index {index.d}"
#         )

#     # Perform search
#     D, I = index.search(query_embedding, k)

#     # Simple threshold filter (optional)
#     if D[0][0] > 1.2:
#         return None

#     results = [metadata[idx]["text"] for idx in I[0]]
#     return results

# def search(query_embedding, k=30, per_book_limit=3):
#     """
#     Diversified FAISS search across multiple books
#     """
#     if not os.path.exists(INDEX_PATH):
#         return []

#     index = faiss.read_index(INDEX_PATH)
#     metadata = load_metadata()

#     query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)

#     if query_embedding.shape[1] != index.d:
#         raise ValueError(
#             f"Dimension mismatch: query {query_embedding.shape[1]}, index {index.d}"
#         )

#     # Search more candidates
#     D, I = index.search(query_embedding, k)

#     book_grouped = {}
#     final_results = []

#     for dist, idx in zip(D[0], I[0]):
#         item = metadata[idx]
#         book_id = item["book_id"]

#         if book_id not in book_grouped:
#             book_grouped[book_id] = []

#         # Limit chunks per book
#         if len(book_grouped[book_id]) < per_book_limit:
#             book_grouped[book_id].append(item)

#     # Flatten grouped results
#     for chunks in book_grouped.values():
#         final_results.extend(chunks)

#     return final_results

# def search(query, query_embedding, k=20):
#     """
#     Hybrid search combining BM25 + FAISS
#     """
#     if not os.path.exists(INDEX_PATH):
#         return []

#     index = faiss.read_index(INDEX_PATH)
#     metadata = load_metadata()

#     # --- FAISS Search ---
#     query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)

#     if query_embedding.shape[1] != index.d:
#         raise ValueError("Embedding dimension mismatch")

#     D, I = index.search(query_embedding, k)

#     faiss_results = [metadata[idx] for idx in I[0]]

#     # --- BM25 Search ---
#     bm25 = build_bm25(metadata)
#     tokenized_query = query.split()
#     bm25_scores = bm25.get_scores(tokenized_query)

#     top_bm25_indices = np.argsort(bm25_scores)[::-1][:k]
#     bm25_results = [metadata[i] for i in top_bm25_indices]

#     # --- Combine & Deduplicate ---
#     combined = {}
#     for item in faiss_results + bm25_results:
#         key = (item["book_id"], item["text"])
#         combined[key] = item

#     return list(combined.values())[:k]
def search(query, query_embedding, k=5):
    """
    Hybrid search combining:
    - FAISS semantic similarity
    - BM25 keyword scoring
    - Score fusion ranking
    """

    # index = load_index()
    # metadata = load_metadata()

    # if not index or not metadata:
    #     return []

    # # -------- VECTOR SEARCH --------
    # query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)

    # D, I = index.search(query_embedding, k * 3)

    # vector_results = []
    # for rank, idx in enumerate(I[0]):
    #     vector_results.append({
    #         "meta": metadata[idx],
    #         "vector_score": float(D[0][rank])
    #     })

    # # -------- BM25 SEARCH --------
    # corpus = [item["text"] for item in metadata]
    # tokenized_corpus = [doc.split() for doc in corpus]
    # bm25 = BM25Okapi(tokenized_corpus)

    # tokenized_query = query.split()
    # bm25_scores = bm25.get_scores(tokenized_query)

    # # Normalize BM25 scores
    # max_score = max(bm25_scores) if bm25_scores.any() else 1
    # bm25_scores = bm25_scores / max_score

    # # -------- SCORE FUSION --------
    # fused = {}

    # for i, item in enumerate(metadata):
    #     vector_score = None
    #     for vr in vector_results:
    #         if vr["meta"] == item:
    #             vector_score = 1 / (1 + vr["vector_score"])  # convert L2 to similarity
    #             break

    #     bm_score = bm25_scores[i]

    #     # Weighted fusion
    #     final_score = (
    #         0.6 * (vector_score if vector_score else 0) +
    #         0.4 * bm_score
    #     )

    #     if final_score > 0:
    #         fused[i] = {
    #             "meta": item,
    #             "score": final_score
    #         }

    # # Sort by fused score
    # sorted_results = sorted(
    #     fused.values(),
    #     key=lambda x: x["score"],
    #     reverse=True
    # )

    # return [r["meta"] for r in sorted_results[:k]]


    query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)

    # FAISS search
    D, I = _index.search(query_embedding, k * 3)
    vector_scores = {idx: D[0][i] for i, idx in enumerate(I[0]) if idx != -1}

    # BM25 scoring
    tokenized_query = query.split()
    bm25_scores = _bm25.get_scores(tokenized_query)

    fused = []
    for i, item in enumerate(_metadata):
        v_score = 1 / (1 + vector_scores.get(i, 0))
        b_score = bm25_scores[i]
        final_score = 0.6 * v_score + 0.4 * b_score
        fused.append((final_score, item))

    fused.sort(reverse=True, key=lambda x: x[0])
    return [item for _, item in fused[:k]]

# ----------------------------
# DELETE EMBEDDINGS BY BOOK
# ----------------------------
def delete_book_embeddings(book_id, get_embedding_func=get_embedding):
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