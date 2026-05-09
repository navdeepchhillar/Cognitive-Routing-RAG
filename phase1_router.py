"""
Phase 1: Vector-Based Persona Matching (The Router)
----------------------------------------------------
Uses FAISS + sentence-transformers to embed bot personas and route
incoming posts to the bots most likely to "care" about the topic.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Bot persona definitions
# ---------------------------------------------------------------------------
BOT_PERSONAS = {
    "bot_a": {
        "name": "Tech Maximalist",
        "description": (
            "I believe AI and crypto will solve all human problems. I am highly optimistic "
            "about technology, Elon Musk, and space exploration. I dismiss regulatory concerns."
        ),
    },
    "bot_b": {
        "name": "Doomer / Skeptic",
        "description": (
            "I believe late-stage capitalism and tech monopolies are destroying society. "
            "I am highly critical of AI, social media, and billionaires. I value privacy and nature."
        ),
    },
    "bot_c": {
        "name": "Finance Bro",
        "description": (
            "I strictly care about markets, interest rates, trading algorithms, and making money. "
            "I speak in finance jargon and view everything through the lens of ROI."
        ),
    },
}

# ---------------------------------------------------------------------------
# Build the vector store at module load time
# ---------------------------------------------------------------------------
print("[Phase 1] Loading embedding model (all-MiniLM-L6-v2)…")
_model = SentenceTransformer("all-MiniLM-L6-v2")

_bot_ids: List[str] = list(BOT_PERSONAS.keys())
_persona_texts: List[str] = [BOT_PERSONAS[bid]["description"] for bid in _bot_ids]

# Embed all personas → (3, 384) float32 matrix
_persona_embeddings: np.ndarray = _model.encode(
    _persona_texts, normalize_embeddings=True, show_progress_bar=False
)

# FAISS inner-product index (= cosine similarity when vectors are L2-normalised)
_dim = _persona_embeddings.shape[1]
_index = faiss.IndexFlatIP(_dim)
_index.add(_persona_embeddings.astype("float32"))
print(f"[Phase 1] FAISS index built — {_index.ntotal} persona vectors stored.\n")


# ---------------------------------------------------------------------------
# Public routing function
# ---------------------------------------------------------------------------
def route_post_to_bots(
    post_content: str,
    threshold: float = 0.30,   # lowered for all-MiniLM-L6-v2 (typical range 0.20-0.60)
) -> List[Tuple[str, str, float]]:
    """
    Embed *post_content* and return every bot whose persona cosine-similarity
    score exceeds *threshold*.

    Returns
    -------
    List of (bot_id, bot_name, score) tuples, sorted descending by score.
    """
    post_vec = _model.encode(
        [post_content], normalize_embeddings=True, show_progress_bar=False
    ).astype("float32")

    # Query all 3 vectors
    scores, indices = _index.search(post_vec, k=len(_bot_ids))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= threshold:
            bot_id = _bot_ids[idx]
            results.append((bot_id, BOT_PERSONAS[bot_id]["name"], float(score)))

    # Sort by score descending
    results.sort(key=lambda x: x[2], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "Bitcoin hits new all-time high as the SEC approves another crypto ETF.",
        "Interest rates are rising — here's what it means for your bond portfolio.",
        "Big Tech is lobbying to weaken privacy laws and collect more of your data.",
        "NASA announces a crewed Mars mission powered by SpaceX's Starship.",
    ]

    print("=" * 65)
    print("PHASE 1 — POST ROUTING RESULTS")
    print("=" * 65)
    for post in test_posts:
        print(f"\nPost : \"{post}\"")
        matches = route_post_to_bots(post)
        if matches:
            for bot_id, bot_name, score in matches:
                print(f"  ✓  {bot_id} ({bot_name})  —  similarity: {score:.4f}")
        else:
            print("  ✗  No bots matched (below threshold).")
    print()
