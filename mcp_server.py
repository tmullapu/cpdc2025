# mcp_server.py
# MCP server exposing RPG game knowledge via RAG pipeline
# Connect this to Claude Desktop or any MCP-compatible client

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
import asyncio
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_engine import retrieve_for_function, initialize_rag, get_lore_stats, retrieve

# Create the MCP server instance
app = Server("cpdc2025-game-knowledge")

# Initialize RAG at startup
initialize_rag()


# ─────────────────────────────────────────────
# Register available tools
# ─────────────────────────────────────────────
@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_location_info",
            description="Retrieve rich lore about a game location including enemies, treasures, and secrets",
            inputSchema={
                "type": "object",
                "properties": {
                    "location_name": {"type": "string", "description": "Name of the location"},
                    "game": {"type": "string", "description": "Game name e.g. Elden Ring, Skyrim, Divinity: Original Sin 2, Baldur's Gate 3"}
                },
                "required": ["location_name", "game"]
            }
        ),
        types.Tool(
            name="get_quest_info",
            description="Retrieve lore and details about a specific quest including objectives and rewards",
            inputSchema={
                "type": "object",
                "properties": {
                    "quest_name": {"type": "string", "description": "Name of the quest"},
                    "game": {"type": "string", "description": "Game name"}
                },
                "required": ["quest_name", "game"]
            }
        ),
        types.Tool(
            name="get_item_info",
            description="Retrieve lore about a game item including stats, rarity, and where to find it",
            inputSchema={
                "type": "object",
                "properties": {
                    "item_name": {"type": "string", "description": "Name of the item"},
                    "game": {"type": "string", "description": "Game name"}
                },
                "required": ["item_name", "game"]
            }
        ),
        types.Tool(
            name="get_recipe_info",
            description="Retrieve crafting recipe details including required ingredients and resulting item",
            inputSchema={
                "type": "object",
                "properties": {
                    "recipe_name": {"type": "string", "description": "Name of the recipe"},
                    "game": {"type": "string", "description": "Game name"}
                },
                "required": ["recipe_name", "game"]
            }
        ),
        types.Tool(
            name="get_skill_tree",
            description="Retrieve skill tree and ability progression for a character class",
            inputSchema={
                "type": "object",
                "properties": {
                    "character_class": {"type": "string", "description": "Character class e.g. Warrior, Mage, Rogue, Bard"},
                    "game": {"type": "string", "description": "Game name"}
                },
                "required": ["character_class", "game"]
            }
        ),
        types.Tool(
            name="get_character_stats",
            description="Retrieve character statistics, build information, and progression details",
            inputSchema={
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "Character name"},
                    "game": {"type": "string", "description": "Game name"}
                },
                "required": ["character_name", "game"]
            }
        ),
        types.Tool(
            name="search_lore",
            description="Free-text semantic search across all game lore — use when you need broad knowledge retrieval across multiple categories",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query e.g. 'dangerous cave with ice enemies in Skyrim'"},
                    "game": {"type": "string", "description": "Optional: filter results to a specific game"},
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="get_lore_database_stats",
            description="Returns statistics about the lore database — how many entries, which games and categories are covered",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
    ]


# ─────────────────────────────────────────────
# Handle tool calls
# ─────────────────────────────────────────────
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        if name == "search_lore":
            query = arguments.get("query", "")
            game = arguments.get("game")
            results = retrieve(query=query, game=game, top_k=3)

            if not results:
                text = "No relevant lore found for that query."
            else:
                chunks = []
                for r in results:
                    chunks.append(
                        f"**{r['name']}** ({r['game']} — {r['category'].replace('_', ' ').title()})\n"
                        f"Relevance: {r['relevance_score']:.2f}\n\n"
                        f"{r['text']}"
                    )
                text = "\n\n---\n\n".join(chunks)

        elif name == "get_lore_database_stats":
            stats = get_lore_stats()
            text = json.dumps(stats, indent=2)

        elif name in [
            "get_location_info",
            "get_quest_info",
            "get_item_info",
            "get_recipe_info",
            "get_skill_tree",
            "get_character_stats",
        ]:
            lore = retrieve_for_function(name, arguments)

            if lore:
                text = lore
            else:
                text = (
                    f"No specific lore found for '{list(arguments.values())[0]}' "
                    f"in {arguments.get('game', 'the specified game')}. "
                    f"Try rephrasing or checking the game name. "
                    f"Available games: Elden Ring, Skyrim, Divinity: Original Sin 2, Baldur's Gate 3."
                )
        else:
            text = f"Unknown tool: {name}"

    except Exception as e:
        text = f"Error executing {name}: {str(e)}"

    return [types.TextContent(type="text", text=text)]


# ─────────────────────────────────────────────
# Register lore database as a resource
# ─────────────────────────────────────────────
@app.list_resources()
async def list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="lore://database/stats",
            name="Lore Database Stats",
            description="Statistics about the game lore database",
            mimeType="application/json",
        ),
        types.Resource(
            uri="lore://database/games",
            name="Supported Games",
            description="List of games covered in the lore database",
            mimeType="application/json",
        ),
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "lore://database/stats":
        stats = get_lore_stats()
        return json.dumps(stats, indent=2)
    elif uri == "lore://database/games":
        games = ["Elden Ring", "Skyrim", "Divinity: Original Sin 2", "Baldur's Gate 3"]
        return json.dumps({"supported_games": games}, indent=2)
    else:
        return json.dumps({"error": f"Unknown resource: {uri}"})


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
