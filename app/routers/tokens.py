from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import re

router = APIRouter()

# ---------- Request/Response Models ----------

class TokenizeRequest(BaseModel):
    text: str

class TokenizeResponse(BaseModel):
    tokens: List[str]
    num_tokens: int
    char_length: int


# ---------- Simple Tokenizer (Mock BPE Style) ----------

def simple_tokenizer(text: str) -> List[str]:
    """
    A naive tokenizer that simulates how tokenization works:
    - Split by spaces
    - Remove special punctuation
    - Treat numbers as separate tokens
    - Keep apostrophes (important for contractions)
    """
    # Replace punctuations with spaces, except apostrophes
    cleaned = re.sub(r"[^\w'\s]", " ", text)

    # Collapse multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return []

    # Split by spaces
    raw_tokens = cleaned.split(" ")

    # Filter out empty tokens
    tokens = [t for t in raw_tokens if t.strip()]
    return tokens


# ---------- Endpoint ----------

@router.post("/count", response_model=TokenizeResponse)
def tokenize(req: TokenizeRequest):
    tokens = simple_tokenizer(req.text)
    return TokenizeResponse(
        tokens=tokens,
        num_tokens=len(tokens),
        char_length=len(req.text)
    )
