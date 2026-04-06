# function_executor.py
# RPG function executor with RAG-powered lore retrieval
# Each function now retrieves semantically relevant lore instead of returning mock data

import os
import sys

# Make sure rag_engine can be found whether we're in src/ or root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from rag_engine import retrieve_for_function, initialize_rag
    _rag_available = True
except ImportError:
    _rag_available = False
    print("Warning: rag_engine not found. Falling back to mock data.")

# Initialize RAG at import time
if _rag_available:
    try:
        initialize_rag()
    except Exception as e:
        print(f"Warning: RAG initialization failed: {e}")
        _rag_available = False


def _rag_or_mock(function_name: str, arguments: dict, mock_data: dict) -> dict:
    """
    Try to retrieve real lore via RAG.
    Falls back to mock_data if RAG is unavailable or returns nothing.
    """
    if _rag_available:
        try:
            lore = retrieve_for_function(function_name, arguments)
            if lore:
                return {
                    "source": "lore_database",
                    "retrieved_lore": lore,
                    **{k: v for k, v in arguments.items()},
                }
        except Exception as e:
            print(f"RAG retrieval error: {e}")
    return mock_data


# ─────────────────────────────────────────────
# RPG Functions
# ─────────────────────────────────────────────

def get_character_stats(character_name: str, game: str):
    """Retrieve character statistics using RAG lore."""
    mock = {
        "source": "mock",
        "character_name": character_name,
        "game": game,
        "level": 45,
        "health": 850,
        "mana": 420,
        "strength": 65,
        "intelligence": 40,
        "dexterity": 55,
        "class": "Warrior",
    }
    return _rag_or_mock(
        "get_character_stats",
        {"character_name": character_name, "game": game},
        mock,
    )


def get_quest_info(quest_name: str, game: str):
    """Retrieve quest information using RAG lore."""
    mock = {
        "source": "mock",
        "quest_name": quest_name,
        "game": game,
        "objectives": ["Defeat the dragon", "Retrieve the artifact", "Rescue the princess"],
        "rewards": {"gold": 5000, "experience": 2500, "items": ["Dragon Scale Armor"]},
        "requirements": {"level": 30, "completed_quests": ["The Ancient Prophecy"]},
    }
    return _rag_or_mock(
        "get_quest_info",
        {"quest_name": quest_name, "game": game},
        mock,
    )


def get_item_info(item_name: str, game: str):
    """Retrieve item information using RAG lore."""
    mock = {
        "source": "mock",
        "item_name": item_name,
        "game": game,
        "type": "Weapon",
        "rarity": "Legendary",
        "stats": {"attack": 120, "durability": 100},
        "location": "Dragon's Peak",
        "description": "A legendary sword forged from dragon scales",
    }
    return _rag_or_mock(
        "get_item_info",
        {"item_name": item_name, "game": game},
        mock,
    )


def get_location_info(location_name: str, game: str):
    """Retrieve location information using RAG lore."""
    mock = {
        "source": "mock",
        "location_name": location_name,
        "game": game,
        "enemies": ["Dragon", "Goblin Horde", "Dark Wizard"],
        "treasures": ["Legendary Sword", "Ancient Tome", "Magic Ring"],
        "secrets": ["Hidden passage behind waterfall", "Secret chest in basement"],
        "recommended_level": 35,
    }
    return _rag_or_mock(
        "get_location_info",
        {"location_name": location_name, "game": game},
        mock,
    )


def get_skill_tree(character_class: str, game: str):
    """Retrieve skill tree information using RAG lore."""
    mock = {
        "source": "mock",
        "character_class": character_class,
        "game": game,
        "skills": [
            {"name": "Fireball", "level": 1, "damage": 50},
            {"name": "Ice Shield", "level": 5, "defense": 30},
            {"name": "Lightning Strike", "level": 10, "damage": 100},
        ],
        "total_skills": 15,
        "max_level": 50,
    }
    return _rag_or_mock(
        "get_skill_tree",
        {"character_class": character_class, "game": game},
        mock,
    )


def get_recipe_info(recipe_name: str, game: str):
    """Retrieve crafting recipe information using RAG lore."""
    mock = {
        "source": "mock",
        "recipe_name": recipe_name,
        "game": game,
        "required_materials": {
            "Iron Ore": 5,
            "Mana Crystal": 2,
            "Dragon Scale": 1,
        },
        "resulting_item": "Enchanted Sword",
        "crafting_level": 25,
    }
    return _rag_or_mock(
        "get_recipe_info",
        {"recipe_name": recipe_name, "game": game},
        mock,
    )


# ─────────────────────────────────────────────
# Function map for execute()
# ─────────────────────────────────────────────
FUNCTION_MAP = {
    "get_character_stats": get_character_stats,
    "get_quest_info": get_quest_info,
    "get_item_info": get_item_info,
    "get_location_info": get_location_info,
    "get_skill_tree": get_skill_tree,
    "get_recipe_info": get_recipe_info,
}


def execute(function_call: dict) -> dict:
    """
    Execute a function call from the dialogue agent.

    Args:
        function_call: dict with 'name' and 'arguments' keys

    Returns:
        dict with retrieved lore or mock data
    """
    name = function_call.get("name")
    args = function_call.get("arguments", {})

    if name not in FUNCTION_MAP:
        return {"error": f"Unknown function: {name}"}

    try:
        return FUNCTION_MAP[name](**args)
    except TypeError as e:
        return {"error": f"Invalid arguments for {name}: {e}"}
    except Exception as e:
        return {"error": f"Execution error in {name}: {e}"}
