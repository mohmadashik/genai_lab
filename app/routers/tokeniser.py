from typing import List, Dict, Any, Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

"""
routers/tokenization.py

Educational FastAPI router that demonstrates core tokenization concepts used in
modern GenAI systems:

- BPE (Byte Pair Encoding)
- WordPiece
- SentencePiece
- Simple whitespace baseline

This module is **mocked**:
- No external tokenization libraries
- No actual model integration
- Deterministic, pure Python behavior

The goal is to show:
- How each tokenizer would *conceptually* work in a real system
- How we can simulate the behavior with simple, rule-based logic
"""

router = APIRouter(prefix="/tokenization", tags=["tokenization"])

# ---------------------------------------------------------------------------
# Mock vocabularies
# In real systems, these are large (30k–100k+). Here we keep them tiny and
# human-readable so that the API is easy to understand and experiment with.
# ---------------------------------------------------------------------------

# Minimal BPE-style vocab: base characters + some merged subwords
BPE_VOCAB: Dict[str, int] = {
    # Single characters
    "t": 1,
    "o": 2,
    "k": 3,
    "e": 4,
    "n": 5,
    "i": 6,
    "z": 7,
    "a": 8,
    "s": 9,
    "p": 10,
    "u": 11,
    " ": 12,
    ".": 13,
    ",": 14,
    "B": 15,
    "P": 16,
    "E": 17,
    "S": 18,
    # Common subwords/merged units
    "to": 100,
    "ken": 101,
    "token": 102,
    "tokenization": 103,
    "piece": 104,
    "sub": 105,
    "word": 106,
}

# Merge rules for BPE – ordered list of (left, right) merges
# In a real tokenizer these are learned; here we just fix a few to illustrate.
BPE_MERGES: List[tuple[str, str]] = [
    ("t", "o"),          # -> "to"
    ("t", "o"),          # repeat is fine; merge while present
    ("to", "ken"),       # -> "token"
    ("token", "ization")  # for "tokenization" broken into "token" + "ization"
]

# Minimal WordPiece-style vocab with "##" to denote continuation pieces
WORDPIECE_VOCAB: Dict[str, int] = {
    "[UNK]": 0,
    "token": 1,
    "tokenization": 2,
    "tok": 3,
    "##en": 4,
    "##ization": 5,
    "word": 6,
    "piece": 7,
    "##piece": 8,
    "sub": 9,
    "##word": 10,
}

# Minimal SentencePiece-style vocab using "▁" to mark word starts
SENTENCEPIECE_VOCAB: Dict[str, int] = {
    "▁": 0,              # just the word-start marker
    "▁token": 1,
    "ization": 2,
    "▁tokenization": 3,
    "▁word": 4,
    "▁piece": 5,
    "▁sub": 6,
    "▁subword": 7,
    "▁sentence": 8,
    "▁piece": 9,
    "▁is": 10,
}

# Simple whitespace vocab – just index tokens in order of appearance
# (we construct this on the fly per request in the mock implementation)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

TokenizerType = Literal["bpe", "wordpiece", "sentencepiece", "whitespace"]


class TokenDebugStep(BaseModel):
    """Human-readable explanation of a step inside the tokenizer."""
    step: str = Field(..., description="Name of the step or phase.")
    description: str = Field(..., description="What happened and why.")
    state: Dict[str, Any] = Field(
        ...,
        description="Intermediate state (tokens, merges, cursor position, etc.).",
    )


class TokenizationRequest(BaseModel):
    """
    Request body for tokenization demo.

    In real systems you’d pass:
    - tokenizer name / config
    - text
    - optional flags (e.g., add_special_tokens)
    """
    text: str = Field(..., example="tokenization with BPE and WordPiece")
    tokenizer_type: TokenizerType = Field(
        "bpe",
        description="Which tokenizer to simulate.",
    )
    return_debug: bool = Field(
        False,
        description="If true, return detailed step-by-step debug info.",
    )


class TokenizationResponse(BaseModel):
    """
    Response that mimics what a real tokenizer API might return.
    """
    tokenizer_type: TokenizerType
    tokens: List[str]
    token_ids: List[int]
    notes: List[str] = Field(
        ...,
        description="Educational notes about what happened / how to read the output.",
    )
    debug: Optional[List[TokenDebugStep]] = None


class TokenizerComparisonRequest(BaseModel):
    """
    Compare how different tokenizers split the same text.
    """
    text: str = Field(..., example="Tokenization is fun with subword models.")
    tokenizers: List[TokenizerType] = Field(
        default_factory=lambda: ["bpe", "wordpiece", "sentencepiece", "whitespace"],
        description="Which tokenizers to run side-by-side.",
    )


class TokenizerComparisonItem(BaseModel):
    tokenizer_type: TokenizerType
    tokens: List[str]
    token_ids: List[int]


class TokenizerComparisonResponse(BaseModel):
    text: str
    results: List[TokenizerComparisonItem]
    summary: List[str]


# ---------------------------------------------------------------------------
# Helper functions – mock implementations
# ---------------------------------------------------------------------------

def _whitespace_tokenize(text: str) -> List[str]:
    """
    The most basic tokenizer: split on spaces.

    Real-world use:
    - Almost never used for LLMs, but good as a conceptual baseline.
    """
    if not text.strip():
        return []

    # Maintain simple behavior: split on whitespace, keep punctuation attached
    return text.split()


def _tokens_to_ids(tokens: List[str], vocab: Dict[str, int], unk_token: str = "[UNK]") -> List[int]:
    """
    Map tokens to IDs using a given vocab, falling back to an UNK token ID or 0.
    """
    unk_id = vocab.get(unk_token, 0)
    return [vocab.get(t, unk_id) for t in tokens]


def _bpe_tokenize(text: str, debug: bool = False) -> tuple[List[str], List[TokenDebugStep]]:
    """
    Extremely simplified BPE tokenizer.

    Real-world BPE:
    - Start from bytes/characters
    - Iteratively merge the most frequent pairs based on a learned merge table
    - Produces subword tokens that balance length and coverage

    Here:
    - We start from characters
    - Apply a tiny, fixed merge list
    - Show how "tokenization" can collapse into larger subword units
    """
    debug_steps: List[TokenDebugStep] = []

    # Step 1: break into characters (including spaces)
    chars = list(text)
    debug_steps.append(
        TokenDebugStep(
            step="initial_chars",
            description="Split text into a list of characters.",
            state={"chars": chars},
        )
    )

    # Step 2: apply merges greedily for each rule until no more matches
    tokens = chars
    for left, right in BPE_MERGES:
        merged = True
        while merged:
            merged = False
            i = 0
            new_tokens: List[str] = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == left and tokens[i + 1] == right:
                    # Merge pair
                    new_tokens.append(left + right)
                    i += 2
                    merged = True
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            if merged:
                tokens = new_tokens
                if debug:
                    debug_steps.append(
                        TokenDebugStep(
                            step="bpe_merge",
                            description=f"Applied merge ({left}, {right}).",
                            state={"tokens": tokens},
                        )
                    )

    # Step 3: split on spaces to make token boundaries clearer
    final_tokens: List[str] = []
    buffer: List[str] = []
    for t in tokens:
        if t == " ":
            if buffer:
                final_tokens.append("".join(buffer))
                buffer = []
        else:
            buffer.append(t)
    if buffer:
        final_tokens.append("".join(buffer))

    debug_steps.append(
        TokenDebugStep(
            step="final_tokens",
            description="Grouped characters into word-like units based on spaces.",
            state={"tokens": final_tokens},
        )
    )

    return final_tokens, debug_steps


def _wordpiece_tokenize(text: str, debug: bool = False) -> tuple[List[str], List[TokenDebugStep]]:
    """
    Simplified WordPiece tokenizer.

    Real-world WordPiece:
    - Split on whitespace into words
    - For each word, greedily match the longest prefix in vocab
    - Remaining part is matched with '##' continuation tokens
    - Unknown parts become [UNK]

    Here:
    - Use a tiny fixed vocab
    - Greedily match substrings
    """
    debug_steps: List[TokenDebugStep] = []
    words = text.split()

    debug_steps.append(
        TokenDebugStep(
            step="split_words",
            description="Split text into whitespace-separated words.",
            state={"words": words},
        )
    )

    tokens: List[str] = []

    for word in words:
        # If word is fully in vocab, use it directly
        if word in WORDPIECE_VOCAB:
            tokens.append(word)
            if debug:
                debug_steps.append(
                    TokenDebugStep(
                        step="wordpiece_full_match",
                        description=f"Word '{word}' found directly in vocab.",
                        state={"word": word, "token": word},
                    )
                )
            continue

        # Otherwise, perform greedy matching
        start = 0
        word_tokens: List[str] = []
        while start < len(word):
            end = len(word)
            cur_sub = None
            while end > start:
                piece = word[start:end] if start == 0 else "##" + word[start:end]
                if piece in WORDPIECE_VOCAB:
                    cur_sub = piece
                    break
                end -= 1

            if cur_sub is None:
                # Fallback to [UNK]
                word_tokens = ["[UNK]"]
                if debug:
                    debug_steps.append(
                        TokenDebugStep(
                            step="wordpiece_unk",
                            description=f"No match found for part of '{word}'. Using [UNK].",
                            state={"word": word, "position": start},
                        )
                    )
                break

            word_tokens.append(cur_sub)
            if debug:
                debug_steps.append(
                    TokenDebugStep(
                        step="wordpiece_subword",
                        description=f"Matched subword piece '{cur_sub}' in '{word}'.",
                        state={"word": word, "subword": cur_sub, "start": start},
                    )
                )
            # Move cursor:
            if cur_sub.startswith("##"):
                start += len(cur_sub) - 2  # remove '##'
            else:
                start += len(cur_sub)

        tokens.extend(word_tokens)

    debug_steps.append(
        TokenDebugStep(
            step="final_tokens",
            description="Combined all subword pieces for all words.",
            state={"tokens": tokens},
        )
    )

    return tokens, debug_steps


def _sentencepiece_tokenize(text: str, debug: bool = False) -> tuple[List[str], List[TokenDebugStep]]:
    """
    Simplified SentencePiece tokenizer.

    Real-world SentencePiece:
    - Works directly on raw text (can be trained on characters, bytes, etc.)
    - Uses a vocabulary where '▁' marks the start of a word
    - Can handle languages without spaces

    Here:
    - We simulate '▁' as a word-start marker before each word
    - Try to match whole word with '▁word' from a tiny vocab
    - Otherwise split into '▁' + the entire word
    """
    debug_steps: List[TokenDebugStep] = []
    words = text.split()

    debug_steps.append(
        TokenDebugStep(
            step="split_words",
            description="Split text into whitespace-separated words.",
            state={"words": words},
        )
    )

    tokens: List[str] = []

    for word in words:
        candidate = f"▁{word}"
        if candidate in SENTENCEPIECE_VOCAB:
            # Use whole-word token
            tokens.append(candidate)
            if debug:
                debug_steps.append(
                    TokenDebugStep(
                        step="spm_full_match",
                        description=f"Word '{word}' matched as '{candidate}' in vocab.",
                        state={"word": word, "token": candidate},
                    )
                )
        else:
            # Fallback: emit '▁word' even if not in vocab
            tokens.append(candidate)
            if debug:
                debug_steps.append(
                    TokenDebugStep(
                        step="spm_fallback",
                        description=f"Word '{word}' not in vocab; emitted '{candidate}' as fallback.",
                        state={"word": word, "token": candidate},
                    )
                )

    debug_steps.append(
        TokenDebugStep(
            step="final_tokens",
            description="SentencePiece-like tokens with '▁' marking word boundaries.",
            state={"tokens": tokens},
        )
    )

    return tokens, debug_steps


# ---------------------------------------------------------------------------
# Core routing logic
# ---------------------------------------------------------------------------

def _run_tokenizer(
    text: str,
    tokenizer_type: TokenizerType,
    return_debug: bool,
) -> TokenizationResponse:
    """
    Route to appropriate tokenizer implementation and construct a response
    that mimics what a real GenAI backend might return.
    """
    debug_steps: List[TokenDebugStep] = []
    notes: List[str] = []

    if tokenizer_type == "whitespace":
        tokens = _whitespace_tokenize(text)
        # Build a per-request vocab to assign IDs
        vocab = {tok: idx + 1 for idx, tok in enumerate(sorted(set(tokens)))}
        token_ids = [vocab[t] for t in tokens]
        notes.append(
            "Whitespace tokenizer: splits only on spaces, keeps punctuation attached."
        )
    elif tokenizer_type == "bpe":
        tokens, debug_steps = _bpe_tokenize(text, debug=return_debug)
        token_ids = _tokens_to_ids(tokens, BPE_VOCAB)
        notes.append(
            "BPE tokenizer: starts from characters and merges frequent pairs into "
            "subword units. This mock uses a tiny, hand-written merge table."
        )
    elif tokenizer_type == "wordpiece":
        tokens, debug_steps = _wordpiece_tokenize(text, debug=return_debug)
        token_ids = _tokens_to_ids(tokens, WORDPIECE_VOCAB, unk_token="[UNK]")
        notes.append(
            "WordPiece tokenizer: greedily matches the longest subword from a vocab. "
            "'##' indicates continuation of a previous piece."
        )
    elif tokenizer_type == "sentencepiece":
        tokens, debug_steps = _sentencepiece_tokenize(text, debug=return_debug)
        token_ids = _tokens_to_ids(tokens, SENTENCEPIECE_VOCAB)
        notes.append(
            "SentencePiece tokenizer: uses '▁' to mark word boundaries and can be "
            "trained directly on raw text, handling languages without spaces."
        )
    else:
        # Should never happen due to Literal constraint
        tokens = []
        token_ids = []
        notes.append("Unknown tokenizer type; no tokens produced.")

    # General notes for all tokenizers
    notes.append(
        "In real LLM systems, these token IDs are what actually go into the model, "
        "not the raw text."
    )
    notes.append(
        "Different tokenizers for the same text will produce different token counts, "
        "which affects context length, memory usage, and cost."
    )

    return TokenizationResponse(
        tokenizer_type=tokenizer_type,
        tokens=tokens,
        token_ids=token_ids,
        notes=notes,
        debug=debug_steps if return_debug else None,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/demo", response_model=TokenizationResponse)
def tokenize_demo(payload: TokenizationRequest) -> TokenizationResponse:
    """
    Main educational endpoint.

    What it simulates (real world):
    - A service call from your backend where you:
        - Choose a tokenizer (BPE, WordPiece, SentencePiece, etc.)
        - Pass in raw text
        - Receive back tokens + IDs for downstream LLMs

    This mock:
    - Uses tiny, fixed vocabularies
    - Implements just enough logic to show how each algorithm behaves
    """
    return _run_tokenizer(
        text=payload.text,
        tokenizer_type=payload.tokenizer_type,
        return_debug=payload.return_debug,
    )


@router.post("/compare", response_model=TokenizerComparisonResponse)
def compare_tokenizers(payload: TokenizerComparisonRequest) -> TokenizerComparisonResponse:
    """
    Compare how multiple tokenizers split the same text.

    This is useful to visually understand:
    - Why some tokenizers create more tokens than others
    - How subword units differ between BPE, WordPiece, and SentencePiece
    """
    results: List[TokenizerComparisonItem] = []

    for t_type in payload.tokenizers:
        res = _run_tokenizer(
            text=payload.text,
            tokenizer_type=t_type,
            return_debug=False,
        )
        results.append(
            TokenizerComparisonItem(
                tokenizer_type=t_type,
                tokens=res.tokens,
                token_ids=res.token_ids,
            )
        )

    summary = [
        "Each tokenizer style produces a different number of tokens for the same text.",
        "BPE and WordPiece often create meaningful subwords (e.g., 'token', '##ization').",
        "SentencePiece uses '▁' markers to track word boundaries inside its tokens.",
        "Whitespace is easiest to read but least useful for real LLMs.",
    ]

    return TokenizerComparisonResponse(
        text=payload.text,
        results=results,
        summary=summary,
    )


"""

Test Cases 
1. Test BPE Tokenization
    - Input
    {
        "text": "tokenization",
        "tokenizer_type": "bpe",
        "return_debug": true
    }
2. Test WordPiece Tokenization
    - Input
    {
        "text": "tokenization",
        "tokenizer_type": "wordpiece",
        "return_debug": true
    }
3. Test SentencePiece Tokenization
    - Input
    {
        "text": "tokenization",
        "tokenizer_type": "sentencepiece",
        "return_debug": true
    }
4. Test Whitespace Tokenization
    - Input
    {
        "text": "tokenization with whitespace",
        "tokenizer_type": "whitespace",
        "return_debug": true
    }
5. Compare All Tokenizers
    - Input
    {
        "text": "Tokenization is fun with subword models.",
        "tokenizers": ["bpe", "wordpiece", "sentencepiece", "whitespace"]
    }

6. stress Test - mixed punctuation and spaces
    - Input
    {
        "text": "Tokenization, is fun! With: BPE; WordPiece? SentencePiece.",
        "tokenizer_type": "bpe",
        "return_debug": true
    }
7. Edge-Case Test - empty string
    - Input
    {
        "text": "",
        "tokenizer_type": "wordpiece",
        "return_debug": true
    }
8. Mixed casing test
    - Input
    {
        "text": "ToKeNiZaTiOn",
        "tokenizer_type": "sentencepiece",
        "return_debug": true
    }
"""