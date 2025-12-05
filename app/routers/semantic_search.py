from typing import List, Dict, Any, Optional, Literal, Tuple
from math import sqrt

from fastapi import APIRouter
from pydantic import BaseModel, Field

"""
routers/semantic_search.py

Educational FastAPI router that demonstrates SEMANTIC SEARCH in a GenAI system.

What is semantic search?
------------------------
- Traditional search: matches exact keywords or simple text patterns.
- Semantic search: matches by MEANING using embeddings (vectors), not just words.
- Typical flow:
    1. Convert query + documents into embeddings (vectors).
    2. Use a distance metric (cosine similarity, dot product) to compare them.
    3. Return Top-K documents with highest similarity scores.

This module is **mocked**:
- Uses simple, deterministic embedding models (char_26, word_hash_32).
- Uses cosine similarity and a toy keyword-overlap score.
- No external APIs, no paid models, no real vector DB.

The goal:
- Show the *shape* of a semantic search API.
- Contrast keyword vs semantic vs hybrid scoring.
- Make it easy to plug into your existing FastAPI "genai_lab" app.
"""


router = APIRouter(prefix="/semantic-search", tags=["semantic-search"])

# ---------------------------------------------------------------------------
# Types & enums
# ---------------------------------------------------------------------------

EmbeddingModelType = Literal["char_26", "word_hash_32"]
ScoringMode = Literal["semantic_only", "keyword_only", "hybrid"]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SemanticDebugStep(BaseModel):
    """Human-readable explanation of a step in the scoring pipeline."""
    step: str = Field(..., description="Logical step name.")
    description: str = Field(..., description="What happened and why.")
    state: Dict[str, Any] = Field(
        ...,
        description="Intermediate state snapshot (tokens, scores, etc.).",
    )


class DocumentInput(BaseModel):
    """
    Input document for semantic search.

    Real systems often store:
    - id
    - raw text
    - metadata (tags, type, timestamp, etc.)
    """
    id: Optional[str] = Field(
        None,
        description="Optional document ID. If omitted, an index-based ID is used.",
        example="doc-123",
    )
    text: str = Field(..., example="Embeddings map text into a vector space.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata. Not used for scoring in this mock.",
    )


class SemanticSearchRequest(BaseModel):
    """
    Main request body for /semantic-search/demo.

    You provide:
    - query: what the user is asking
    - documents: candidate texts
    - model: how to embed text
    - scoring_mode: how to combine keyword and semantic scores
    """
    query: str = Field(..., example="What are embeddings?")
    documents: List[DocumentInput] = Field(
        ...,
        min_items=1,
        example=[
            {
                "id": "d1",
                "text": "Embeddings are vector representations of text.",
                "metadata": {"source": "docs"},
            },
            {
                "id": "d2",
                "text": "Tokenization splits text into smaller units.",
                "metadata": {"source": "docs"},
            },
        ],
    )
    model: EmbeddingModelType = Field(
        "word_hash_32",
        description="Mock embedding model to simulate semantic similarity.",
    )
    normalize: bool = Field(
        True,
        description="Whether to L2-normalize embeddings before similarity.",
    )
    scoring_mode: ScoringMode = Field(
        "semantic_only",
        description=(
            "semantic_only: use cosine similarity only.\n"
            "keyword_only : use keyword overlap only.\n"
            "hybrid       : combine both (simple weighted average)."
        ),
    )
    top_k: int = Field(
        3,
        ge=1,
        description="Number of top results to return.",
    )
    return_embeddings: bool = Field(
        False,
        description="If true, include raw query/document embeddings in the response.",
    )
    return_debug: bool = Field(
        False,
        description="If true, include step-by-step scoring explanation per result.",
    )


class SemanticSearchResult(BaseModel):
    """
    Per-document result item.
    """
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float = Field(..., description="Final score based on scoring_mode.")
    semantic_score: float = Field(
        ...,
        description="Cosine similarity score.",
    )
    keyword_score: float = Field(
        ...,
        description="Keyword overlap score (0–1).",
    )
    embedding: Optional[List[float]] = Field(
        None,
        description="Document embedding (if return_embeddings=True).",
    )
    debug: Optional[List[SemanticDebugStep]] = None


class SemanticSearchResponse(BaseModel):
    """
    Overall response for /demo.
    """
    query: str
    query_embedding: Optional[List[float]] = Field(
        None,
        description="Embedding of the query (if return_embeddings=True).",
    )
    model: EmbeddingModelType
    dim: int
    scoring_mode: ScoringMode
    results: List[SemanticSearchResult]
    notes: List[str]


class PairwiseAnalyzeRequest(BaseModel):
    """
    Simple pairwise analysis: one query vs one document.

    This is just a convenience endpoint to deeply inspect
    how scores are computed for a single pair.
    """
    query: str = Field(..., example="Explain embeddings.")
    document: str = Field(
        ...,
        example="Embeddings are numeric representations of text meaning.",
    )
    model: EmbeddingModelType = Field("word_hash_32")
    normalize: bool = Field(True)


class PairwiseAnalyzeResponse(BaseModel):
    query: str
    document: str
    model: EmbeddingModelType
    dim: int
    semantic_score: float
    keyword_score: float
    debug: List[SemanticDebugStep]
    notes: List[str]


# ---------------------------------------------------------------------------
# Math helpers (L2 + cosine)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Mock embedding models (same spirit as embeddings/vector_db modules)
# ---------------------------------------------------------------------------

def _deterministic_hash(text: str) -> int:
    """
    Deterministic 32-bit polynomial hash.

    We don't use Python's built-in hash() because it's randomized.
    """
    h = 0
    for ch in text:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return h


def _char_26_embed(text: str, normalize: bool) -> Tuple[List[float], int, List[SemanticDebugStep]]:
    """
    char_26 model:
    - 26-dim character count vector over a-z.
    - Very crude, but deterministic and easy to reason about.
    """
    debug: List[SemanticDebugStep] = []

    lower = text.lower()
    letters = [c for c in lower if "a" <= c <= "z"]

    debug.append(
        SemanticDebugStep(
            step="preprocess",
            description="Lowercased text and filtered to a-z.",
            state={"original": text, "lower": lower, "letters": "".join(letters)},
        )
    )

    vec = [0.0] * 26
    for ch in letters:
        vec[ord(ch) - ord("a")] += 1.0

    debug.append(
        SemanticDebugStep(
            step="char_counts",
            description="Counted frequency of each character.",
            state={"counts": vec},
        )
    )

    if normalize:
        vec = _l2_normalize(vec)
        debug.append(
            SemanticDebugStep(
                step="normalize",
                description="Applied L2 normalization to 26-dim vector.",
                state={"normalized": vec},
            )
        )

    return vec, 26, debug


def _word_hash_32_embed(text: str, normalize: bool) -> Tuple[List[float], int, List[SemanticDebugStep]]:
    """
    word_hash_32 model:
    - 32-dim vector, one bucket per hash index.
    - Each word increments a bucket.
    """
    debug: List[SemanticDebugStep] = []

    lower = text.lower()
    words = [w for w in lower.split() if w.strip()]

    debug.append(
        SemanticDebugStep(
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
        debug.append(
            SemanticDebugStep(
                step="word_hash",
                description="Hashed word into one of 32 buckets.",
                state={"word": w, "hash": h, "index": idx, "vector_snapshot": vec.copy()},
            )
        )

    if normalize:
        vec = _l2_normalize(vec)
        debug.append(
            SemanticDebugStep(
                step="normalize",
                description="Applied L2 normalization to 32-dim vector.",
                state={"normalized": vec},
            )
        )

    return vec, dim, debug


def _embed_text(
    text: str,
    model: EmbeddingModelType,
    normalize: bool,
) -> Tuple[List[float], int, List[SemanticDebugStep]]:
    """
    Tiny dispatcher to pick the embedding model.

    In a real system:
    - This is where you'd call your embeddings service (or vendor API).
    """
    if model == "char_26":
        return _char_26_embed(text, normalize)
    elif model == "word_hash_32":
        return _word_hash_32_embed(text, normalize)
    else:
        # Should not happen given Literal type.
        return [], 0, []


# ---------------------------------------------------------------------------
# Keyword tokenization & scoring (baseline search)
# ---------------------------------------------------------------------------

def _tokenize_for_keywords(text: str) -> List[str]:
    """
    Very simple keyword tokenizer:
    - Lowercase
    - Split on whitespace
    - Strip punctuation from both ends
    - Drop empty tokens

    This is closer to "traditional" search behavior.
    """
    import string

    lower = text.lower()
    tokens: List[str] = []
    for raw in lower.split():
        tok = raw.strip(string.punctuation)
        if tok:
            tokens.append(tok)
    return tokens


def _keyword_overlap_score(
    query_tokens: List[str],
    doc_tokens: List[str],
) -> float:
    """
    Jaccard similarity between sets of tokens:
    score = |intersection| / |union|

    Range: 0.0 (no overlap) to 1.0 (identical sets).
    """
    q_set = set(query_tokens)
    d_set = set(doc_tokens)
    if not q_set or not d_set:
        return 0.0
    inter = q_set & d_set
    union = q_set | d_set
    return len(inter) / len(union)


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------

def _score_single_document(
    query: str,
    doc: DocumentInput,
    model: EmbeddingModelType,
    normalize: bool,
    scoring_mode: ScoringMode,
    include_embedding: bool,
    include_debug: bool,
) -> SemanticSearchResult:
    """
    Compute scores for a single query-document pair.

    Produces:
    - semantic_score: cosine similarity of embeddings
    - keyword_score : Jaccard token overlap
    - final score   : based on scoring_mode
    """
    # Embed query & doc
    q_emb, dim, q_debug = _embed_text(query, model, normalize)
    d_emb, d_dim, d_debug = _embed_text(doc.text, model, normalize)

    # Basic safety: dimension check
    if dim != d_dim:
        
        # In this controlled lab, we simply clamp to min(dim, d_dim)
        min_dim = min(dim, d_dim)
        q_emb = q_emb[:min_dim]
        d_emb = d_emb[:min_dim]
        dim = min_dim

    # Semantic sc

