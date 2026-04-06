# rag_engine.py
# Semantic lore retrieval using sentence-transformers + cosine similarity
# No external vector DB needed — runs fully in memory

import json
import os
import numpy as np
from typing import List, Dict, Any, Optional

# Lazy import — only load when first used to avoid slow startup
_model = None
_lore_entries = []
_lore_embeddings = None  # shape: (N, 384)


def _get_model():
    """Load the sentence-transformer model once and cache it."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _load_lore(lore_path: str = None) -> List[Dict[str, Any]]:
    """Load lore entries from lore_database.json."""
    if lore_path is None:
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "lore_database.json"),
            os.path.join("lore_database.json"),
            os.path.join("data", "lore_database.json"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                lore_path = path
                break

    if lore_path is None or not os.path.exists(lore_path):
        print("Warning: lore_database.json not found. RAG will return empty results.")
        return []

    with open(lore_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_embeddings(entries: List[Dict[str, Any]]) -> np.ndarray:
    """Convert lore entries to embeddings."""
    model = _get_model()
    # Combine name + text for richer embeddings
    texts = [f"{e['name']} {e['game']} {e['category']} {e['text']}" for e in entries]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms


def initialize_rag(lore_path: str = None):
    """
    Load lore and build embeddings.
    Call this once at app startup.
    """
    global _lore_entries, _lore_embeddings
    _lore_entries = _load_lore(lore_path)
    if _lore_entries:
        print(f"RAG: Loaded {len(_lore_entries)} lore entries. Building embeddings...")
        _lore_embeddings = _build_embeddings(_lore_entries)
        print("RAG: Embeddings ready.")
    else:
        _lore_embeddings = None


def retrieve(
    query: str,
    game: str = None,
    category: str = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most relevant lore entries for a query.

    Args:
        query: Natural language search query
        game: Optional filter by game name
        category: Optional filter by category (location, item, quest, recipe, skill_tree, character_stats)
        top_k: Number of results to return

    Returns:
        List of lore entry dicts, sorted by relevance
    """
    global _lore_entries, _lore_embeddings

    # Initialize if not done yet
    if _lore_embeddings is None:
        initialize_rag()

    if not _lore_entries or _lore_embeddings is None:
        return []

    # Filter candidates by game/category if specified
    if game or category:
        candidates = []
        candidate_indices = []
        for i, entry in enumerate(_lore_entries):
            game_match = (game is None) or (game.lower() in entry["game"].lower())
            cat_match = (category is None) or (entry["category"] == category)
            if game_match and cat_match:
                candidates.append(entry)
                candidate_indices.append(i)

        if not candidates:
            # Fall back to no filter if nothing matches
            candidates = _lore_entries
            candidate_indices = list(range(len(_lore_entries)))

        candidate_embeddings = _lore_embeddings[candidate_indices]
    else:
        candidates = _lore_entries
        candidate_embeddings = _lore_embeddings

    # Embed the query
    model = _get_model()
    query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
    query_norm = np.linalg.norm(query_embedding)
    if query_norm > 0:
        query_embedding = query_embedding / query_norm

    # Cosine similarity (dot product of normalized vectors)
    scores = candidate_embeddings @ query_embedding

    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        entry = candidates[idx].copy()
        entry["relevance_score"] = float(scores[idx])
        results.append(entry)

    return results


def retrieve_for_function(function_name: str, arguments: Dict[str, Any]) -> Optional[str]:
    """
    High-level retrieval tailored to specific function calls.
    Returns a formatted lore string ready to inject into an LLM response.

    Args:
        function_name: e.g. "get_location_info", "get_quest_info"
        arguments: the function's arguments dict

    Returns:
        A formatted string of retrieved lore, or None if nothing found
    """
    # Map function names to categories
    category_map = {
        "get_location_info": "location",
        "get_quest_info": "quest",
        "get_item_info": "item",
        "get_recipe_info": "recipe",
        "get_skill_tree": "skill_tree",
        "get_character_stats": "character_stats",
    }

    category = category_map.get(function_name)
    game = arguments.get("game")

    # Build a natural language query from the arguments
    if function_name == "get_location_info":
        query = f"{arguments.get('location_name', '')} location enemies treasures secrets {game}"
    elif function_name == "get_quest_info":
        query = f"{arguments.get('quest_name', '')} quest objectives rewards {game}"
    elif function_name == "get_item_info":
        query = f"{arguments.get('item_name', '')} item stats rarity {game}"
    elif function_name == "get_recipe_info":
        query = f"{arguments.get('recipe_name', '')} recipe ingredients crafting {game}"
    elif function_name == "get_skill_tree":
        query = f"{arguments.get('character_class', '')} skill tree abilities progression {game}"
    elif function_name == "get_character_stats":
        query = f"{arguments.get('character_name', '')} character stats level health {game}"
    else:
        query = " ".join(str(v) for v in arguments.values())

    results = retrieve(query=query, game=game, category=category, top_k=2)

    if not results:
        return None

    # Format the results into a readable lore context
    lore_chunks = []
    for r in results:
        score = r.get("relevance_score", 0)
        if score > 0.2:  # Only include if reasonably relevant
            lore_chunks.append(
                f"[{r['game']} — {r['category'].replace('_', ' ').title()}] {r['name']}:\n{r['text']}"
            )

    if not lore_chunks:
        return None

    return "\n\n".join(lore_chunks)


def get_lore_stats() -> Dict[str, Any]:
    """Return stats about the loaded lore database — useful for debugging."""
    if not _lore_entries:
        return {"status": "not initialized", "entries": 0}

    from collections import Counter
    games = Counter(e["game"] for e in _lore_entries)
    categories = Counter(e["category"] for e in _lore_entries)

    return {
        "status": "ready",
        "total_entries": len(_lore_entries),
        "by_game": dict(games),
        "by_category": dict(categories),
        "embedding_shape": list(_lore_embeddings.shape) if _lore_embeddings is not None else None,
    }
