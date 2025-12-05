from typing import List, Dict, Any, Literal, Optional, Tuple
from math import sqrt

from fastapi import APIRouter
from pydantic import BaseModel, Field

"""
routers/embeddings.py

Educational FastAPI router that demonstrates core embedding concepts used in
modern GenAI systems.

What are embeddings?
--------------------
- Numeric vector representations of text (or images/audio/etc.)
- Capture semantic similarity: similar meaning -> similar vectors
- Used for RAG, search, recommendations, clustering, etc.

This module is **mocked**:
- No external ML libraries
- No real embedding models
- Deterministic, pure Python behavior

We simulate two simple "models":
- char_26       : 26-dim vector of character counts (a–z)
- word_hash_32  : 32-dim vector using a simple deterministic hash on words

The goal is to show:
- How an embedding API is structured
- How embeddings + cosine similarity behave in a system
- How you’d wire this up before using a real vendor/model
"""

router = APIRouter(prefix="/embeddings", tags=["embeddings"])

# ---------------------------------------------------------------------------
# Types & Pydantic models
# ---------------------------------------------------------------------------

EmbeddingModelType = Literal["char_26", "word_hash_32"]


class EmbeddingDebugStep(BaseModel):
    """Human-readable explanation of an internal step in embedding creation."""
    step: str = Field(..., description="Name of the step.")
    description: str = Field(..., description="What happened and why.")
    state: Dict[str, Any] = Field(
        ...,
        description="Intermediate state, e.g., tokens, counts, norms.",
    )


class EmbeddingRequest(BaseModel):
    """
    Request for the /demo endpoint.

    Real world:
    - You usually send one or many texts to an embeddings endpoint of an LLM API.
    - You specify model name (e.g. text-embedding-*.).
    """
    texts: List[str] = Field(
        ...,
        min_items=1,
        example=["Embeddings are vector representations of text."],
        description="One or more input texts to embed.",
    )
    model: EmbeddingModelType = Field(
        "char_26",
        description="Which mock embedding model to use.",
    )
    normalize: bool = Field(
        True,
        description="Whether to L2-normalize the embedding vectors.",
    )
    return_debug: bool = Field(
        False,
        description="If true, returns step-by-step debug info for each text.",
    )


class EmbeddingItem(BaseModel):
    """Per-text embedding result to mirror real embedding APIs."""
    text: str
    model: EmbeddingModelType
    dim: int
    embedding: List[float]
    notes: List[str]
    debug: Optional[List[EmbeddingDebugStep]] = None


class EmbeddingResponse(BaseModel):
    """Response from /demo endpoint."""
    model: EmbeddingModelType
    dim: int
    embeddings: List[EmbeddingItem]
    notes: List[str]


class SimilarityRequest(BaseModel):
    """
    Request for /similarity endpoint.

    Simulates semantic search:
    - embed query + documents
    - compute cosine similarity
    - return sorted scores
    """
    query: str = Field(..., example="What are embeddings?")
    documents: List[str] = Field(
        ...,
        min_items=1,
        example=[
            "Embeddings map text into a vector space.",
            "Tokenization splits text into smaller units.",
        ],
    )
    model: EmbeddingModelType = Field(
        "char_26",
        description="Which mock embedding model to use.",
    )
    normalize: bool = Field(
        True,
        description="Normalize embeddings before computing cosine similarity.",
    )
    return_embeddings: bool = Field(
        False,
        description="If true, include raw embeddings in the response for inspection.",
    )
    top_k: Optional[int] = Field(
        None,
        description="If set, return only the top_k most similar documents.",
    )
    return_debug: bool = Field(
        False,
        description="If true, include debug info for query/document embeddings.",
    )


class SimilarityResult(BaseModel):
    """Per-document similarity score."""
    index: int = Field(..., description="Index of the document in the input list.")
    document: str
    score: float = Field(..., description="Cosine similarity between query and document.")
    debug: Optional[List[EmbeddingDebugStep]] = None


class SimilarityResponse(BaseModel):
    model: EmbeddingModelType
    dim: int
    query: str
    query_embedding: Optional[List[float]] = None
    document_embeddings: Optional[List[List[float]]] = None
    results: List[SimilarityResult]
    notes: List[str]


# ---------------------------------------------------------------------------
# Helper: L2 normalization & cosine similarity
# ---------------------------------------------------------------------------

def _l2_norm(vec: List[float]) -> float:
    return sqrt(sum(x * x for x in vec)) if vec else 0.0


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = _l2_norm(vec)
    if norm == 0.0:
        return vec[:]  # return unchanged to avoid division by zero
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


# ---------------------------------------------------------------------------
# Helper: deterministic hashing for the word_hash_32 model
# ---------------------------------------------------------------------------

def _deterministic_hash(text: str) -> int:
    """
    Very simple deterministic hash function.

    We do NOT use Python's built-in hash() because:
    - It's randomized per process (hash randomization)
    - We want deterministic behavior across runs

    Polynomial rolling hash mod 2**32 is enough for this demo.
    """
    h = 0
    for ch in text:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF  # keep in 32-bit range
    return h


# ---------------------------------------------------------------------------
# Embedding model implementations (mock)
# ---------------------------------------------------------------------------

def _char_26_embedding(
    text: str,
    normalize: bool,
    return_debug: bool,
) -> Tuple[List[float], int, List[EmbeddingDebugStep]]:
    """
    char_26 model:
    - 26 dimensions, one per lowercase English letter a-z
    - Value is the count of that character in the text
    - Optionally L2-normalized

    Real world analogy:
    - This is like a *very crude* bag-of-characters model.
    - It has zero understanding of semantics, but the flow is similar
      to real embedding calls: text -> numeric vector -> normalization.
    """
    debug_steps: List[EmbeddingDebugStep] = []

    # Preprocess: lowercase and filter a-z
    lower = text.lower()
    letters_only = [ch for ch in lower if "a" <= ch <= "z"]

    if return_debug:
        debug_steps.append(
            EmbeddingDebugStep(
                step="preprocess",
                description="Lowercased text and kept only a-z characters.",
                state={"original": text, "lower": lower, "letters_only": "".join(letters_only)},
            )
        )

    vec = [0.0] * 26
    for ch in letters_only:
        idx = ord(ch) - ord("a")
        vec[idx] += 1.0

    if return_debug:
        debug_steps.append(
            EmbeddingDebugStep(
                step="char_counts",
                description="Counted frequency of each character a-z.",
                state={"counts": vec},
            )
        )

    if normalize:
        vec = _l2_normalize(vec)
        if return_debug:
            debug_steps.append(
                EmbeddingDebugStep(
                    step="normalize",
                    description="Applied L2 normalization to the 26-dim vector.",
                    state={"normalized": vec},
                )
            )

    dim = 26
    return vec, dim, debug_steps


def _word_hash_32_embedding(
    text: str,
    normalize: bool,
    return_debug: bool,
) -> Tuple[List[float], int, List[EmbeddingDebugStep]]:
    """
    word_hash_32 model:
    - 32 dimensions
    - Split text into words (by whitespace)
    - Each word is hashed to an index 0..31
    - Increment that index

    Real world analogy:
    - This is like a toy "feature hashing" or "bag-of-words" model.
    - Real embedding models learn dense vectors vs. simple counts.
    """
    debug_steps: List[EmbeddingDebugStep] = []

    lower = text.lower()
    words = [w for w in lower.split() if w.strip()]

    if return_debug:
        debug_steps.append(
            EmbeddingDebugStep(
                step="preprocess",
                description="Lowercased text and split on whitespace.",
                state={"original": text, "lower": lower, "words": words},
            )
        )

    dim = 32
    vec = [0.0] * dim

    for w in words:
        h = _deterministic_hash(w)
        idx = h % dim
        vec[idx] += 1.0
        if return_debug:
            debug_steps.append(
                EmbeddingDebugStep(
                    step="word_hash",
                    description="Hashed word to an index and incremented its bucket.",
                    state={"word": w, "hash": h, "index": idx, "vector_snapshot": vec.copy()},
                )
            )

    if normalize:
        vec = _l2_normalize(vec)
        if return_debug:
            debug_steps.append(
                EmbeddingDebugStep(
                    step="normalize",
                    description="Applied L2 normalization to the 32-dim vector.",
                    state={"normalized": vec},
                )
            )

    return vec, dim, debug_steps


def _compute_embedding_for_text(
    text: str,
    model: EmbeddingModelType,
    normalize: bool,
    return_debug: bool,
) -> Tuple[List[float], int, List[EmbeddingDebugStep], List[str]]:
    """
    Route to appropriate mock embedding model and add conceptual notes.
    """
    debug_steps: List[EmbeddingDebugStep] = []
    notes: List[str] = []

    if model == "char_26":
        embedding, dim, debug_steps = _char_26_embedding(text, normalize, return_debug)
        notes.append(
            "char_26 model: a simple bag-of-characters representation over a-z."
        )
    elif model == "word_hash_32":
        embedding, dim, debug_steps = _word_hash_32_embedding(text, normalize, return_debug)
        notes.append(
            "word_hash_32 model: uses a hash function to map words into a fixed 32-dim vector."
        )
    else:
        # Should not happen because of Literal, but keep a safe branch.
        embedding = []
        dim = 0
        notes.append("Unknown model; no embedding produced.")

    # Generic notes applicable to any embedding model
    notes.append(
        "In real GenAI systems, these embeddings would come from a trained model "
        "that encodes semantic meaning, not just counts."
    )
    notes.append(
        "These vectors can be stored in a vector database and queried using "
        "cosine similarity or other distance metrics for search and RAG."
    )

    return embedding, dim, debug_steps, notes


# ---------------------------------------------------------------------------
# Core batch logic
# ---------------------------------------------------------------------------

def _compute_embeddings_batch(
    texts: List[str],
    model: EmbeddingModelType,
    normalize: bool,
    return_debug: bool,
) -> Tuple[List[EmbeddingItem], int, List[str]]:
    """
    Compute embeddings for a batch of texts and wrap them into EmbeddingItem models.
    """
    items: List[EmbeddingItem] = []
    shared_dim: int = 0
    shared_notes: List[str] = []

    for text in texts:
        embedding, dim, debug_steps, notes = _compute_embedding_for_text(
            text=text,
            model=model,
            normalize=normalize,
            return_debug=return_debug,
        )
        if shared_dim == 0:
            shared_dim = dim
        shared_notes = notes  # they are the same structure per model

        items.append(
            EmbeddingItem(
                text=text,
                model=model,
                dim=dim,
                embedding=embedding,
                notes=notes,
                debug=debug_steps if return_debug else None,
            )
        )

    return items, shared_dim, shared_notes


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/demo", response_model=EmbeddingResponse)
def embeddings_demo(payload: EmbeddingRequest) -> EmbeddingResponse:
    """
    Main educational endpoint for embeddings.

    Simulates:
    - An embeddings API call where you pass a batch of texts
    - You get back vectors + metadata per text

    Mock behavior:
    - char_26       : character-count based 26-dim vector
    - word_hash_32  : word-hash based 32-dim vector
    """
    items, dim, notes = _compute_embeddings_batch(
        texts=payload.texts,
        model=payload.model,
        normalize=payload.normalize,
        return_debug=payload.return_debug,
    )

    # Add endpoint-level notes
    notes = notes + [
        f"Returned {len(items)} embedding(s).",
        f"Embedding dimension for model '{payload.model}' is {dim}.",
        "In a real system, you'd plug in a vendor/model call here instead "
        "of these simple rule-based vectors.",
    ]

    return EmbeddingResponse(
        model=payload.model,
        dim=dim,
        embeddings=items,
        notes=notes,
    )


@router.post("/similarity", response_model=SimilarityResponse)
def embeddings_similarity(payload: SimilarityRequest) -> SimilarityResponse:
    """
    Semantic similarity demo.

    Real world:
    - This is essentially what a semantic search service does:
        1. Embed the query text
        2. Embed each document
        3. Compute cosine similarity
        4. Rank and return the best matches

    Mock flow:
    - Uses simple deterministic embeddings
    - Cosine similarity is still real and behaves like in production
    """
    # Compute query embedding
    query_emb, dim, query_debug, _ = _compute_embedding_for_text(
        text=payload.query,
        model=payload.model,
        normalize=payload.normalize,
        return_debug=payload.return_debug,
    )

    # Compute document embeddings (no need to collect notes per doc here)
    doc_embeddings: List[List[float]] = []
    doc_debugs: List[List[EmbeddingDebugStep]] = []

    for doc in payload.documents:
        emb, d_dim, dbg, _ = _compute_embedding_for_text(
            text=doc,
            model=payload.model,
            normalize=payload.normalize,
            return_debug=payload.return_debug,
        )
        doc_embeddings.append(emb)
        doc_debugs.append(dbg if payload.return_debug else [])
        # Sanity: enforce same dimension
        if d_dim != dim:
            dim = min(dim, d_dim)

    # Compute cosine similarity
    results: List[SimilarityResult] = []
    for idx, (doc, emb, dbg) in enumerate(zip(payload.documents, doc_embeddings, doc_debugs)):
        score = _cosine_similarity(query_emb, emb)
        results.append(
            SimilarityResult(
                index=idx,
                document=doc,
                score=score,
                debug=dbg if payload.return_debug else None,
            )
        )

    # Sort by similarity descending
    results.sort(key=lambda r: r.score, reverse=True)

    # Apply top_k if requested
    if payload.top_k is not None:
        results = results[: payload.top_k]

    notes = [
        f"Computed embeddings using model '{payload.model}' with dimension {dim}.",
        "Cosine similarity is used to estimate how close each document is "
        "to the query in embedding space.",
        "In a production RAG setup, this endpoint mirrors the logic of a "
        "vector-search layer sitting in front of your LLM.",
    ]

    return SimilarityResponse(
        model=payload.model,
        dim=dim,
        query=payload.query,
        query_embedding=query_emb if payload.return_embeddings else None,
        document_embeddings=doc_embeddings if payload.return_embeddings else None,
        results=results,
        notes=notes,
    )
