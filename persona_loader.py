# persona_loader.py
import json
import os

DEFAULT_FUNCTIONS = [
    {"name": "get_character_stats", "description": "Retrieves character statistics including level, health, mana, and attributes", "parameters": {"character_name": "string", "game": "string"}},
    {"name": "get_quest_info", "description": "Gets information about a specific quest including objectives, rewards, and requirements", "parameters": {"quest_name": "string", "game": "string"}},
    {"name": "get_item_info", "description": "Retrieves information about an item including stats, rarity, and where to find it", "parameters": {"item_name": "string", "game": "string"}},
    {"name": "get_location_info", "description": "Gets information about a game location including enemies, treasures, and secrets", "parameters": {"location_name": "string", "game": "string"}},
    {"name": "get_skill_tree", "description": "Retrieves the skill tree or ability progression for a character class", "parameters": {"character_class": "string", "game": "string"}},
    {"name": "get_recipe_info", "description": "Gets crafting recipe information including required materials and resulting item", "parameters": {"recipe_name": "string", "game": "string"}},
]

# ─────────────────────────────────────────────
# Player Personas — who the USER plays as
# NPCs adapt their tone and style based on this
# ─────────────────────────────────────────────
PLAYER_PERSONAS = {
    "Aggressive Warrior": {
        "name": "Aggressive Warrior",
        "description": "A battle-hardened fighter who values speed and power above all else.",
        "traits": ["impatient", "direct", "combat-focused", "bold"],
        "communication_style": "short, blunt, action-oriented",
        "wants": "quick answers, battle stats, no storytelling or poetry",
        "expects_npc_to": "get to the point immediately, be bold and decisive, skip emotional depth, focus on combat utility",
        "example_phrases": ["Just tell me how to win.", "Skip the story, give me the stats.", "What's the fastest way to kill it?"],
    },
    "Cautious Scholar": {
        "name": "Cautious Scholar",
        "description": "An analytical thinker who needs context, lore, and reasoning before acting.",
        "traits": ["analytical", "curious", "detail-oriented", "methodical"],
        "communication_style": "thorough, thoughtful, wants full context",
        "wants": "lore, explanations, background, reasoning behind everything",
        "expects_npc_to": "be thorough and explain reasoning, provide historical context, never rush, treat knowledge as sacred",
        "example_phrases": ["But why does it work that way?", "What's the history behind this location?", "I need to understand before I act."],
    },
    "Chaotic Rogue": {
        "name": "Chaotic Rogue",
        "description": "An unpredictable trickster who loves surprises, shortcuts, and bending the rules.",
        "traits": ["unpredictable", "humorous", "risk-taker", "irreverent"],
        "communication_style": "playful, sarcastic, unconventional",
        "wants": "shortcuts, hidden paths, unexpected angles, humor",
        "expects_npc_to": "match their playful energy, not take things too seriously, reveal secrets and loopholes, be a co-conspirator",
        "example_phrases": ["Is there a back door to this?", "What's the most chaotic option?", "Rules are suggestions, right?"],
    },
    "Empathetic Healer": {
        "name": "Empathetic Healer",
        "description": "A compassionate soul focused on relationships, team safety, and emotional connection.",
        "traits": ["compassionate", "relationship-focused", "cautious", "warm"],
        "communication_style": "gentle, emotionally aware, team-oriented",
        "wants": "emotional connection, team dynamics, story, safety of companions",
        "expects_npc_to": "be warm and check in emotionally, slow down, acknowledge feelings, prioritize people over objectives",
        "example_phrases": ["Is everyone going to be okay?", "How are you feeling about all this?", "Let's make sure no one gets hurt."],
    },
    "Red Witch": {
        "name": "Red Witch",
        "description": "A mysterious, prophetic figure who sees fate and hidden forces others cannot. Inspired by the Red Priestesses of Asshai.",
        "traits": ["mysterious", "manipulative", "prophetic", "patient", "ancient"],
        "communication_style": "cryptic, weighted, speaks in half-truths and riddles",
        "wants": "hidden knowledge, dark secrets, power dynamics, prophecy, fate",
        "expects_npc_to": "speak in riddles, acknowledge forces beyond the obvious, treat her as an equal or superior, never be dismissive, hint at deeper meanings",
        "example_phrases": ["The flame showed me this would happen.", "You already know the answer — you simply fear it.", "Fate does not ask permission."],
    },
}

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _ensure_functions_on_persona(p):
    """Attach default functions if persona has none."""
    if "functions" not in p or not isinstance(p.get("functions"), list) or len(p.get("functions", [])) == 0:
        p["functions"] = DEFAULT_FUNCTIONS.copy()
    return p


def get_player_personas():
    """Return the player persona dictionary."""
    return PLAYER_PERSONAS


def get_player_persona_tone(player_persona_name: str) -> str:
    """
    Returns a tone instruction string to inject into the NPC system prompt
    so the NPC adapts its communication style to the player.
    """
    p = PLAYER_PERSONAS.get(player_persona_name)
    if not p:
        return ""

    return (
        f"\n\n--- PLAYER CONTEXT ---\n"
        f"You are speaking with: {p['name']}\n"
        f"Their traits: {', '.join(p['traits'])}\n"
        f"Their communication style: {p['communication_style']}\n"
        f"What they want: {p['wants']}\n"
        f"How you should adapt: {p['expects_npc_to']}\n"
        f"--- END PLAYER CONTEXT ---\n"
    )


# ─────────────────────────────────────────────
# NPC Persona loading
# ─────────────────────────────────────────────
def load_personas_from_dataset(dataset_path=None):
    """Load unique NPC personas from the RPG dataset."""
    personas = {}

    if dataset_path is None:
        possible_paths = [
            os.path.join("rpg_persona_dataset.jsonl"),
            os.path.join("data", "rpg_persona_dataset.jsonl"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rpg_persona_dataset.jsonl"),
            os.path.join(os.path.dirname(__file__), "..", "data", "rpg_persona_dataset.jsonl"),
        ]
    else:
        possible_paths = [dataset_path]

    dataset_file = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_file = path
            break

    if dataset_file is None:
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
                        worldview_str = data["persona"].get("worldview", "")
                        worldview = {"type": worldview_str} if worldview_str else {}
                        persona_data = {
                            "name": data["persona"]["name"],
                            "role": data["persona"]["role"],
                            "traits": data["persona"]["traits"],
                            "worldview": worldview,
                            "functions": data.get("functions", [])
                        }
                        personas[persona_name] = _ensure_functions_on_persona(persona_data)
                except json.JSONDecodeError as json_err:
                    print(f"Warning: Skipping malformed JSON at line {line_num}: {json_err}")
                    continue
                except KeyError as key_err:
                    print(f"Warning: Skipping line {line_num} (missing field): {key_err}")
                    continue
    except Exception as e:
        print(f"Error loading personas from dataset: {e}")
        return get_default_personas()

    return personas


def get_default_personas():
    """Return default NPC personas if dataset is not available."""
    personas = {
        "Aether": {
            "name": "Aether",
            "role": "Heroic Adventurer",
            "traits": ["brave", "noble", "determined", "heroic"],
            "worldview": {"type": "heroic_journey"},
            "functions": DEFAULT_FUNCTIONS.copy()
        },
        "Myst": {
            "name": "Myst",
            "role": "Sage Wizard",
            "traits": ["wise", "knowledgeable", "mystical", "thoughtful"],
            "worldview": {"type": "strategic_exploration"},
            "functions": DEFAULT_FUNCTIONS.copy()
        },
        "Shadow": {
            "name": "Shadow",
            "role": "Rogue Assassin",
            "traits": ["cunning", "stealthy", "independent", "resourceful"],
            "worldview": {"type": "stealth_adventure"},
            "functions": DEFAULT_FUNCTIONS.copy()
        },
    }
    for name in personas:
        personas[name] = _ensure_functions_on_persona(personas[name])
    return personas
