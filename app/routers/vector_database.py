from typing import List, Dict, Any, Optional, Literal, Tuple
from math import sqrt
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

"""
routers/vector_database.py

Educational FastAPI router that demonstrates the core ideas behind
a Vector Database used in GenAI systems.

What is a Vector Database?
--------------------------
- A store for (id, vector, metadata) tuples.
- You send it embeddings (numeric vectors) for your texts/documents.
- Later, you send a query -> it returns the most similar vectors (Top-K search).
- Used heavily in:
    - RAG (Retrieval-Augmented Generation)
    - Semantic search
    - Recommendations
    - Similarity search

This module is **mocked**:
- In-memory storage (Python dicts/lists) instead of a real DB.
- Brute-force cosine similarity instead of fancy indexes (HNSW, IVF, PQ, etc.).
- Simple "embedding" generation inside this file (no external model calls).

The goal:
- Show the *shape* of a vector DB API.
- Show how collections, inserts, and search work end-to-end.
- Make it plug-and-play with your existing embeddings/tokenization lab.
"""

router = APIRouter(prefix="/vector-db", tags=["vector-db"])

# ---------------------------------------------------------------------------
# Mock Embedding Model (same *idea* as embeddings.py, but self-contained)
# ---------------------------------------------------------------------------

EmbeddingModelType = Literal["char_26", "word_hash_32"]


def _l2_norm(vec: List[float]) -> float:
    return sqrt(sum(x * x for x in vec)) if vec else 0.0


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = _l2_norm(vec)
    if norm == 0.0:
        return vec[:]
    return [x / norm for x in vec]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = _l2_norm(a)
    norm_b = _l2_norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _deterministic_hash(text: str) -> int:
    """
    Deterministic polynomial rolling hash (32-bit).
    Used by the word_hash_32 mock embedding model.
    """
    h = 0
    for ch in text:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return h


def _char_26_embed(text: str, normalize: bool) -> List[float]:
    """
    char_26 model:
    - 26-dim vector (a-z counts), optionally L2-normalized.
    """
    vec = [0.0] * 26
    for ch in text.lower():
        if "a" <= ch <= "z":
            vec[ord(ch) - ord("a")] += 1.0
    if normalize:
        vec = _l2_normalize(vec)
    return vec


def _word_hash_32_embed(text: str, normalize: bool) -> List[float]:
    """
    word_hash_32 model:
    - 32-dim vector, word-level hashing into buckets.
    """
    dim = 32
    vec = [0.0] * dim
    for w in text.lower().split():
        if not w.strip():
            continue
        h = _deterministic_hash(w)
        vec[h % dim] += 1.0
    if normalize:
        vec = _l2_normalize(vec)
    return vec


def _embed_text(
    text: str,
    model: EmbeddingModelType,
    normalize: bool,
) -> Tuple[List[float], int]:
    """
    Mock embedding dispatcher used by this router.

    Real world:
    - This would call your /embeddings service OR a vendor API.
    - Here it's just simple rule-based vectors.
    """
    if model == "char_26":
        vec = _char_26_embed(text, normalize=normalize)
        return vec, 26
    elif model == "word_hash_32":
        vec = _word_hash_32_embed(text, normalize=normalize)
        return vec, 32
    else:
        # Should be blocked by Literal, but guard anyway.
        return [], 0


# ---------------------------------------------------------------------------
# In-memory Vector DB structures
# ---------------------------------------------------------------------------

class StoredVector(BaseModel):
    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorCollection(BaseModel):
    """
    Represents a collection inside the vector DB.

    Real vector DBs store:
    - Collection configuration (dimension, distance metric, etc.)
    - Index structures
    - Shards/replicas

    Here we store:
    - name
    - embedding model type
    - embedding dimension
    - normalization flag
    - a simple list of vectors
    """
    name: str
    description: Optional[str] = None
    embedding_model: EmbeddingModelType
    dim: int
    normalize: bool = True
    documents: List[StoredVector] = Field(default_factory=list)


# Global in-memory "database"
VECTOR_DB: Dict[str, VectorCollection] = {}


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class CreateCollectionRequest(BaseModel):
    name: str = Field(..., example="docs_rag")
    description: Optional[str] = Field(
        None,
        example="Collection for RAG on product docs.",
    )
    embedding_model: EmbeddingModelType = Field(
        "char_26",
        description="Mock embedding model used for this collection.",
    )
    normalize: bool = Field(
        True,
        description="Normalize embeddings before storing (recommended).",
    )


class CreateCollectionResponse(BaseModel):
    collection: VectorCollection
    notes: List[str]


class ListCollectionsResponse(BaseModel):
    collections: List[VectorCollection]
    notes: List[str]


class AddDocumentsRequest(BaseModel):
    collection: str = Field(..., example="docs_rag")
    texts: List[str] = Field(
        ...,
        min_items=1,
        description="Documents to embed and store.",
        example=["Embeddings connect text to vector space."],
    )
    ids: Optional[List[str]] = Field(
        None,
        description="Optional custom IDs. If omitted, random UUIDs are used.",
    )
    metadatas: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Optional per-document metadata. Must match length of texts if provided.",
    )


class AddDocumentsResponse(BaseModel):
    collection: str
    added: int
    ids: List[str]
    notes: List[str]


class SearchFilter(BaseModel):
    """
    Minimal metadata filter mock.

    Real vector DBs support:
    - Boolean queries
    - Range queries
    - Nested filters

    Here we do:
    - Simple equality match for key-value pairs.
    """
    equals: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value pairs to match exactly in document metadata.",
    )


class SearchRequest(BaseModel):
    collection: str = Field(..., example="docs_rag")
    query: str = Field(..., example="What are embeddings?")
    top_k: int = Field(
        3,
        ge=1,
        description="Number of most similar documents to return.",
    )
    include_embeddings: bool = Field(
        False,
        description="Include embeddings in the response (for inspection).",
    )
    filter: Optional[SearchFilter] = None


class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class SearchResponse(BaseModel):
    collection: str
    query: str
    model: EmbeddingModelType
    dim: int
    results: List[SearchResult]
    notes: List[str]


class ResetCollectionsResponse(BaseModel):
    cleared_collections: int
    notes: List[str]


# ---------------------------------------------------------------------------
# Helper functions for collection and filtering
# ---------------------------------------------------------------------------

def _get_collection_or_404(name: str) -> VectorCollection:
    coll = VECTOR_DB.get(name)
    if coll is None:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")
    return coll


def _metadata_matches_filter(
    metadata: Dict[str, Any],
    filt: Optional[SearchFilter],
) -> bool:
    if filt is None:
        return True
    for k, v in filt.equals.items():
        if metadata.get(k) != v:
            return False
    return True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/create_collection", response_model=CreateCollectionResponse)
def create_collection(payload: CreateCollectionRequest) -> CreateCollectionResponse:
    """
    Create a new vector collection.

    Real world:
    - This is equivalent to creating an index or collection in Pinecone/Chroma/etc.
    - You'd configure dimension, distance metric, replicas, etc.

    Mock behavior:
    - We don't ask for dimension; it's implied by the chosen embedding model.
    - We store everything in memory in a global dict.
    """
    if payload.name in VECTOR_DB:
        raise HTTPException(
            status_code=400,
            detail=f"Collection '{payload.name}' already exists.",
        )

    # Probe dimension with an empty text (safe because models are deterministic).
    _, dim = _embed_text("", payload.embedding_model, normalize=payload.normalize)

    collection = VectorCollection(
        name=payload.name,
        description=payload.description,
        embedding_model=payload.embedding_model,
        dim=dim,
        normalize=payload.normalize,
        documents=[],
    )
    VECTOR_DB[payload.name] = collection

    notes = [
        f"Created collection '{payload.name}' with model '{payload.embedding_model}' "
        f"and dimension {dim}.",
        "In a real vector DB, this is where you'd allocate an index and storage.",
    ]

    return CreateCollectionResponse(
        collection=collection,
        notes=notes,
    )


@router.get("/collections", response_model=ListCollectionsResponse)
def list_collections() -> ListCollectionsResponse:
    """
    List all collections in the in-memory vector DB.
    """
    collections = list(VECTOR_DB.values())
    notes = [
        f"Currently {len(collections)} collection(s) in the in-memory vector DB.",
        "In a production system this would query the DB's metadata or admin API.",
    ]
    return ListCollectionsResponse(
        collections=collections,
        notes=notes,
    )


@router.post("/add", response_model=AddDocumentsResponse)
def add_documents(payload: AddDocumentsRequest) -> AddDocumentsResponse:
    """
    Add documents to a collection.

    Flow:
    - Look up collection config (model, dim, normalization)
    - For each text:
        - Compute mock embedding
        - Validate dimension
        - Store (id, embedding, text, metadata)
    """
    collection = _get_collection_or_404(payload.collection)

    if payload.ids is not None and len(payload.ids) != len(payload.texts):
        raise HTTPException(
            status_code=400,
            detail="Length of 'ids' must match length of 'texts'.",
        )

    if payload.metadatas is not None and len(payload.metadatas) != len(payload.texts):
        raise HTTPException(
            status_code=400,
            detail="Length of 'metadatas' must match length of 'texts'.",
        )

    assigned_ids: List[str] = []

    for idx, text in enumerate(payload.texts):
        doc_id = payload.ids[idx] if payload.ids is not None else str(uuid4())
        metadata = (
            payload.metadatas[idx]
            if payload.metadatas is not None
            else {}
        )

        emb, dim = _embed_text(
            text=text,
            model=collection.embedding_model,
            normalize=collection.normalize,
        )

        if dim != collection.dim:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Embedding dimension {dim} does not match collection "
                    f"dimension {collection.dim}."
                ),
            )

        stored = StoredVector(
            id=doc_id,
            text=text,
            embedding=emb,
            metadata=metadata,
        )
        collection.documents.append(stored)
        assigned_ids.append(doc_id)

    notes = [
        f"Added {len(assigned_ids)} document(s) to collection '{collection.name}'.",
        "In a real vector DB, this is where you'd perform an upsert into the index.",
    ]

    return AddDocumentsResponse(
        collection=collection.name,
        added=len(assigned_ids),
        ids=assigned_ids,
        notes=notes,
    )


@router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest) -> SearchResponse:
    """
    Semantic search over a collection.

    Real world:
    - This is exactly what a vector DB does:
        1. Embed query
        2. Search vectors using a distance metric (cosine, dot, L2)
        3. Return Top-K nearest neighbors

    Mock behavior:
    - Uses the same mock embedding model as the collection.
    - Scans all documents linearly (O(n)).
    """
    collection = _get_collection_or_404(payload.collection)

    # Compute query embedding
    query_embedding, dim = _embed_text(
        text=payload.query,
        model=collection.embedding_model,
        normalize=collection.normalize,
    )
    if dim != collection.dim:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Query embedding dimension {dim} does not match collection "
                f"dimension {collection.dim}."
            ),
        )

    # Compute similarity for all documents (with metadata filtering)
    scored_docs: List[SearchResult] = []
    for doc in collection.documents:
        if not _metadata_matches_filter(doc.metadata, payload.filter):
            continue

        score = _cosine_similarity(query_embedding, doc.embedding)
        scored_docs.append(
            SearchResult(
                id=doc.id,
                text=doc.text,
                score=score,
                metadata=doc.metadata,
                embedding=doc.embedding if payload.include_embeddings else None,
            )
        )

    # Sort by score (highest first) and trim to top_k
    scored_docs.sort(key=lambda r: r.score, reverse=True)
    top_results = scored_docs[: payload.top_k]

    notes = [
        f"Searched collection '{collection.name}' using model "
        f"'{collection.embedding_model}' with dimension {collection.dim}.",
        f"Returned top {len(top_results)} result(s) out of {len(scored_docs)} "
        "candidate(s) after metadata filtering.",
        "In a real vector DB, this step would use an approximate nearest neighbor "
        "index instead of scanning all vectors.",
    ]

    return SearchResponse(
        collection=collection.name,
        query=payload.query,
        model=collection.embedding_model,
        dim=collection.dim,
        results=top_results,
        notes=notes,
    )


@router.post("/reset", response_model=ResetCollectionsResponse)
def reset_collections() -> ResetCollectionsResponse:
    """
    Danger button for the lab: wipe all in-memory collections.

    Real world:
    - You usually DON'T do this casually.
    - Useful here for quickly resetting the playground.
    """
    count = len(VECTOR_DB)
    VECTOR_DB.clear()

    notes = [
        f"Cleared {count} collection(s) from the in-memory vector DB.",
        "In production you'd guard this behind admin rights or never expose it.",
    ]
    return ResetCollectionsResponse(
        cleared_collections=count,
        notes=notes,
    )
