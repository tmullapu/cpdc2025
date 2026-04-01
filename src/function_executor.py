def get_weather(city: str):
    return {
        "weather": "sunny",
        "temperature": 21,
        "city": city
    }

def get_player_stats(player_name: str, game: str):
    """Returns player statistics for a game"""
    return {
        "player_name": player_name,
        "game": game,
        "win_rate": 0.65,
        "kda": 2.3,
        "rank": "Diamond",
        "total_matches": 450
    }

def get_leaderboard(game: str, mode: str):
    """Returns leaderboard for a game and mode"""
    return {
        "game": game,
        "mode": mode,
        "top_players": [
            {"rank": 1, "player": "ShadowKiller", "rating": 2500},
            {"rank": 2, "player": "ProGamer99", "rating": 2450},
            {"rank": 3, "player": "ElitePlayer", "rating": 2400}
        ]
    }

def get_game_info(game_name: str):
    """Returns information about a game"""
    return {
        "game_name": game_name,
        "release_date": "2020-01-01",
        "genre": "FPS",
        "rating": 4.5,
        "developer": "Game Studios",
        "platforms": ["PC", "Console"]
    }

def get_match_history(player_name: str, game: str, limit: int = 10):
    """Returns match history for a player"""
    return {
        "player_name": player_name,
        "game": game,
        "matches": [
            {"match_id": f"match_{i}", "result": "win", "kda": 2.5, "date": "2024-01-15"}
            for i in range(limit)
        ]
    }

def get_team_composition(game: str, mode: str):
    """Returns suggested team composition"""
    return {
        "game": game,
        "mode": mode,
        "recommended_composition": {
            "tank": "Reinhardt",
            "dps": ["Soldier: 76", "Tracer"],
            "support": ["Mercy", "Ana"]
        },
        "strategy": "Aggressive push composition"
    }

# RPG-specific functions
def get_character_stats(character_name: str, game: str):
    """Returns character statistics for an RPG"""
    return {
        "character_name": character_name,
        "game": game,
        "level": 45,
        "health": 850,
        "mana": 420,
        "strength": 65,
        "intelligence": 40,
        "dexterity": 55,
        "class": "Warrior"
    }

def get_quest_info(quest_name: str, game: str):
    """Returns quest information"""
    return {
        "quest_name": quest_name,
        "game": game,
        "objectives": ["Defeat the dragon", "Retrieve the artifact", "Rescue the princess"],
        "rewards": {"gold": 5000, "experience": 2500, "items": ["Dragon Scale Armor"]},
        "requirements": {"level": 30, "completed_quests": ["The Ancient Prophecy"]}
    }

def get_item_info(item_name: str, game: str):
    """Returns item information"""
    return {
        "item_name": item_name,
        "game": game,
        "type": "Weapon",
        "rarity": "Legendary",
        "stats": {"attack": 120, "durability": 100},
        "location": "Dragon's Peak",
        "description": "A legendary sword forged from dragon scales"
    }

def get_location_info(location_name: str, game: str):
    """Returns location information"""
    return {
        "location_name": location_name,
        "game": game,
        "enemies": ["Dragon", "Goblin Horde", "Dark Wizard"],
        "treasures": ["Legendary Sword", "Ancient Tome", "Magic Ring"],
        "secrets": ["Hidden passage behind waterfall", "Secret chest in basement"],
        "recommended_level": 35
    }

def get_skill_tree(character_class: str, game: str):
    """Returns skill tree information"""
    return {
        "character_class": character_class,
        "game": game,
        "skills": [
            {"name": "Fireball", "level": 1, "damage": 50},
            {"name": "Ice Shield", "level": 5, "defense": 30},
            {"name": "Lightning Strike", "level": 10, "damage": 100}
        ],
        "total_skills": 15,
        "max_level": 50
    }

def get_recipe_info(recipe_name: str, game: str):
    """Returns crafting recipe information"""
    return {
        "recipe_name": recipe_name,
        "game": game,
        "required_materials": {
            "Iron Ore": 5,
            "Mana Crystal": 2,
            "Dragon Scale": 1
        },
        "resulting_item": "Enchanted Sword",
        "crafting_level": 25
    }

FUNCTION_MAP = {
    "get_weather": get_weather,
    "get_player_stats": get_player_stats,
    "get_leaderboard": get_leaderboard,
    "get_game_info": get_game_info,
    "get_match_history": get_match_history,
    "get_team_composition": get_team_composition,
    "get_character_stats": get_character_stats,
    "get_quest_info": get_quest_info,
    "get_item_info": get_item_info,
    "get_location_info": get_location_info,
    "get_skill_tree": get_skill_tree,
    "get_recipe_info": get_recipe_info
}

def execute(function_call: dict):
    name = function_call.get("name")
    args = function_call.get("arguments", {})
    if name in FUNCTION_MAP:
        return FUNCTION_MAP[name](**args)
    return {"error": f"Unknown function: {name}"}
