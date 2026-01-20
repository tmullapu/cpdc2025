 # This file is used to load the personas from the dataset.
import json
import os

# Default tool/function schemas used by the prompt
DEFAULT_FUNCTIONS = [
    {"name": "get_character_stats", "description": "Retrieves character statistics including level, health, mana, and attributes", "parameters": {"character_name": "string", "game": "string"}},
    {"name": "get_quest_info", "description": "Gets information about a specific quest including objectives, rewards, and requirements", "parameters": {"quest_name": "string", "game": "string"}},
    {"name": "get_item_info", "description": "Retrieves information about an item including stats, rarity, and where to find it", "parameters": {"item_name": "string", "game": "string"}},
    {"name": "get_location_info", "description": "Gets information about a game location including enemies, treasures, and secrets", "parameters": {"location_name": "string", "game": "string"}},
    {"name": "get_skill_tree", "description": "Retrieves the skill tree or ability progression for a character class", "parameters": {"character_class": "string", "game": "string"}},
    {"name": "get_recipe_info", "description": "Gets crafting recipe information including required materials and resulting item", "parameters": {"recipe_name": "string", "game": "string"}},
]

def _ensure_functions_on_persona(p):
    """Attach default functions if persona has none"""
    if "functions" not in p or not isinstance(p.get("functions"), list) or len(p.get("functions", [])) == 0:
        p["functions"] = DEFAULT_FUNCTIONS.copy()
    return p

def load_personas_from_dataset(dataset_path=None):
    """Load unique personas from the RPG dataset"""
    personas = {}
    
    # Trying multiple possible paths to find the dataset
    if dataset_path is None:
        possible_paths = [
            os.path.join("data", "rpg_persona_dataset.jsonl"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rpg_persona_dataset.jsonl"),
            os.path.join(os.path.dirname(__file__), "..", "data", "rpg_persona_dataset.jsonl"),
            "../data/rpg_persona_dataset.jsonl",
        ]
    else:
        possible_paths = [dataset_path]
    
    dataset_file = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_file = path
            break
    
    if dataset_file is None:
        # Return default personas if dataset doesn't exist
        return get_default_personas()
    
    try:
        with open(dataset_file, 'r', encoding='utf-8-sig') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    persona_name = data["persona"]["name"]
                    
                    if persona_name not in personas:
                        # Extract worldview - it's a string in the dataset under data["persona"]["worldview"]
                        worldview_str = data["persona"].get("worldview", "")
                        # Convert to dict format for compatibility with existing code
                        worldview = {"type": worldview_str} if worldview_str else {}
                        
                        persona_data = {
                            "name": data["persona"]["name"],
                            "role": data["persona"]["role"],
                            "traits": data["persona"]["traits"],
                            "worldview": worldview,
                            "functions": data.get("functions", [])
                        }
                        # Ensure functions are present
                        personas[persona_name] = _ensure_functions_on_persona(persona_data)
                except json.JSONDecodeError as json_err:
                    # Skip malformed lines but log the error
                    print(f"Warning: Skipping malformed JSON at line {line_num}: {json_err}")
                    if len(line) > 0:
                        print(f"   Line preview (first 100 chars): {line[:100]}")
                    continue
                except KeyError as key_err:
                    # Skip lines missing required fields
                    print(f"Warning: Skipping line {line_num} (missing field): {key_err}")
                    continue
    except Exception as e:
        # If there's a critical error loading, fall back to defaults
        print(f"Error loading personas from dataset: {e}")
        return get_default_personas()
    
    return personas

def get_default_personas():
    """Return default RPG personas if dataset is not available"""
    personas = {
        "Aether": {
            "name": "Aether",
            "role": "Heroic Adventurer",
            "traits": ["brave", "noble", "determined", "heroic"],
            "worldview": {
                "game_style": "heroic_journey",
                "favorite_genres": ["Fantasy RPG", "Action RPG", "JRPG"],
                "character_type": "Warrior/Paladin",
                "playstyle": "chivalrous",
                "preferred_platforms": ["PC", "Console"],
                "alignment": "Lawful Good"
            },
            "functions": [
                {
                    "name": "get_character_stats",
                    "description": "Retrieves character statistics including level, health, mana, and attributes",
                    "parameters": {"character_name": "string", "game": "string"}
                },
                {
                    "name": "get_quest_info",
                    "description": "Gets information about a specific quest including objectives, rewards, and requirements",
                    "parameters": {"quest_name": "string", "game": "string"}
                },
                {
                    "name": "get_item_info",
                    "description": "Retrieves information about an item including stats, rarity, and where to find it",
                    "parameters": {"item_name": "string", "game": "string"}
                },
                {
                    "name": "get_location_info",
                    "description": "Gets information about a game location including enemies, treasures, and secrets",
                    "parameters": {"location_name": "string", "game": "string"}
                },
                {
                    "name": "get_skill_tree",
                    "description": "Retrieves the skill tree or ability progression for a character class",
                    "parameters": {"character_class": "string", "game": "string"}
                },
                {
                    "name": "get_recipe_info",
                    "description": "Gets crafting recipe information including required materials and resulting item",
                    "parameters": {"recipe_name": "string", "game": "string"}
                }
            ]
        },
        "Myst": {
            "name": "Myst",
            "role": "Sage Wizard",
            "traits": ["wise", "knowledgeable", "mystical", "thoughtful"],
            "worldview": {
                "game_style": "strategic_exploration",
                "favorite_genres": ["Fantasy RPG", "Strategy RPG", "CRPG"],
                "character_type": "Mage/Scholar",
                "playstyle": "methodical",
                "preferred_platforms": ["PC"],
                "alignment": "Neutral Good"
            },
            "functions": [
                {
                    "name": "get_character_stats",
                    "description": "Retrieves character statistics including level, health, mana, and attributes",
                    "parameters": {"character_name": "string", "game": "string"}
                },
                {
                    "name": "get_quest_info",
                    "description": "Gets information about a specific quest including objectives, rewards, and requirements",
                    "parameters": {"quest_name": "string", "game": "string"}
                },
                {
                    "name": "get_item_info",
                    "description": "Retrieves information about an item including stats, rarity, and where to find it",
                    "parameters": {"item_name": "string", "game": "string"}
                },
                {
                    "name": "get_location_info",
                    "description": "Gets information about a game location including enemies, treasures, and secrets",
                    "parameters": {"location_name": "string", "game": "string"}
                },
                {
                    "name": "get_skill_tree",
                    "description": "Retrieves the skill tree or ability progression for a character class",
                    "parameters": {"character_class": "string", "game": "string"}
                },
                {
                    "name": "get_recipe_info",
                    "description": "Gets crafting recipe information including required materials and resulting item",
                    "parameters": {"recipe_name": "string", "game": "string"}
                }
            ]
        },
        "Shadow": {
            "name": "Shadow",
            "role": "Rogue Assassin",
            "traits": ["cunning", "stealthy", "independent", "resourceful"],
            "worldview": {
                "game_style": "stealth_adventure",
                "favorite_genres": ["Fantasy RPG", "Action RPG", "Stealth RPG"],
                "character_type": "Rogue/Assassin",
                "playstyle": "tactical",
                "preferred_platforms": ["PC", "Console"],
                "alignment": "Chaotic Neutral"
            },
            "functions": [
                {
                    "name": "get_character_stats",
                    "description": "Retrieves character statistics including level, health, mana, and attributes",
                    "parameters": {"character_name": "string", "game": "string"}
                },
                {
                    "name": "get_quest_info",
                    "description": "Gets information about a specific quest including objectives, rewards, and requirements",
                    "parameters": {"quest_name": "string", "game": "string"}
                },
                {
                    "name": "get_item_info",
                    "description": "Retrieves information about an item including stats, rarity, and where to find it",
                    "parameters": {"item_name": "string", "game": "string"}
                },
                {
                    "name": "get_location_info",
                    "description": "Gets information about a game location including enemies, treasures, and secrets",
                    "parameters": {"location_name": "string", "game": "string"}
                },
                {
                    "name": "get_skill_tree",
                    "description": "Retrieves the skill tree or ability progression for a character class",
                    "parameters": {"character_class": "string", "game": "string"}
                },
                {
                    "name": "get_recipe_info",
                    "description": "Gets crafting recipe information including required materials and resulting item",
                    "parameters": {"recipe_name": "string", "game": "string"}
                }
            ]
        }
    }
    
    # Ensure all default personas have functions
    for name in personas:
        personas[name] = _ensure_functions_on_persona(personas[name])
    
    return personas
