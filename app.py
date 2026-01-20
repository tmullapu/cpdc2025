import streamlit as st
from groq import Groq
from openai import OpenAI
import os
import json
import sys
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv

# Need to add the parent directory to Python's path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from streamlit_app.persona_loader import load_personas_from_dataset, get_default_personas
except ImportError:
    # If the normal import doesn't work, trying to load it manually as a fallback
    import importlib.util
    spec = importlib.util.spec_from_file_location("persona_loader", os.path.join(os.path.dirname(__file__), "persona_loader.py"))
    persona_loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(persona_loader)
    load_personas_from_dataset = persona_loader.load_personas_from_dataset
    get_default_personas = persona_loader.get_default_personas

from src.function_executor import execute

# Trying to grab the evaluation functions we need
try:
    from streamlit_app.evaluation import score_example, METRIC_KEYS, aggregate_rows
except ImportError:
    # Same fallback trick if the import fails
    import importlib.util
    spec = importlib.util.spec_from_file_location("evaluation", os.path.join(os.path.dirname(__file__), "evaluation.py"))
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    score_example = evaluation.score_example
    METRIC_KEYS = evaluation.METRIC_KEYS
    aggregate_rows = evaluation.aggregate_rows

# Look for a .env file to load API keys and stuff - check a couple common spots
env_paths = [
    os.path.join(os.path.dirname(__file__), ".env"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
]
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        break
else:
    # If we didn't find one, just try loading from wherever we are
    load_dotenv()

# Set up the page title and layout
st.set_page_config(
    page_title="AI Chat Interface",
    page_icon="",
    layout="wide"
)

# These are the different ways we can prompt the AI - user picks one
STRATEGIES = [
    "Zero-Shot",
    "Few-Shot",
    "Chain of Thought",
    "Persona Sandwich",
]

# Set up all the stuff we need to remember between page refreshes
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_provider" not in st.session_state:
    st.session_state.api_provider = "openai"  # Default to OpenAI
if "selected_persona" not in st.session_state:
    st.session_state.selected_persona = None
if "selected_strategy" not in st.session_state:
    st.session_state.selected_strategy = STRATEGIES[0]
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
if "groq_client" not in st.session_state:
    st.session_state.groq_client = None
if "openai_client" not in st.session_state:
    st.session_state.openai_client = None
if "enable_action_first" not in st.session_state:
    st.session_state.enable_action_first = False

# Try to load personas from the dataset, but if that fails, use some defaults
try:
    PERSONAS = load_personas_from_dataset()
except:
    PERSONAS = get_default_personas()

def initialize_groq_client(api_key: str):
    """Initialize GROQ client with API key"""
    try:
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error initializing GROQ client: {str(e)}")
        return None

def initialize_openai_client(api_key: str):
    """Initialize OpenAI client with API key"""
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        return None

def get_groq_response(client: Groq, messages: List[Dict], model: str = "llama-3.3-70b-versatile"):
    """Get response from GROQ API"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def get_openai_response(client: OpenAI, messages: List[Dict], model: str = "gpt-4o-mini"):
    """Get response from OpenAI API"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# === Strategy rules with explicit JSON schema (allows persona chat + function_call) ===
def strategy_rules(strategy_name: str, enable_action_first: bool = False) -> str:
    # This is the base format we want the AI to output - JSON with response and optional function_call
    BASE_JSON_SCHEMA = (
        "OUTPUT JSON ONLY (exact keys):\n"
        "{\n"
        '  "response": "<short in-character line confirming what you\'ll fetch>",\n'
        '  "function_call": {"name": "<function name>", "arguments": { /* exact keys/values */ }}\n'
        "}\n"
        "If no function is needed, omit 'function_call'.\n"
        "IMPORTANT: Do NOT use 'function'/'parameters' - only use 'function_call'/'arguments'."
    )

    if strategy_name == "Zero-Shot":
        return (
            "Be helpful, concise, and stay in character.\n"
            "Use a function when needed.\n\n" + BASE_JSON_SCHEMA
        )
    elif strategy_name == "Few-Shot":
        return (
            "Stay strictly in character. Use a function when clearly needed.\n"
            "Keep replies concise.\n\n" + BASE_JSON_SCHEMA
        )
    elif strategy_name == "Chain of Thought":
        return (
            "Think step by step:\n"
            "1) Analyze what the user is asking\n"
            "2) Determine if a function call is needed\n"
            "3) If yes, identify the correct function and arguments\n"
            "4) Respond in character\n\n" + BASE_JSON_SCHEMA
        )
    else:  # Persona Sandwich
        rules = []
        if enable_action_first:
            rules.append(
                "1) ACTION-FIRST: If the user asks for retrievable game data (stats, quests, items, locations, skills, recipes), "
                "call the correct function FIRST (before any persona text).\n"
            )
        rules.append("2) SINGLE-SHOT: Make at most one function call this turn.\n")
        rules.append("3) NO-CALL FOR CHITCHAT/AMBIGUITY: For opinions/feelings/ambiguous asks, do NOT call any function.\n")
        rules.append("4) " + BASE_JSON_SCHEMA + "\n")
        if enable_action_first:
            rules.append("If you call a function, you may leave 'response' empty; the app will reply after executing the tool.")
        
        return "RULES:\n" + "".join(rules)

def build_persona_prompt(persona_data: Dict[str, Any],
                         user_message: str,
                         chat_history: List[Dict[str, str]],
                         strategy_name: str,
                         enable_action_first: bool = False) -> str:
    """
    Returns a single system prompt string that encodes persona, tools, and strategy rules.
    The model must output JSON:
    {
      "response": "persona text (may be empty if action-first)",
      "function_call": {"name": "...", "arguments": {...}}   # optional
    }
    """
    if not persona_data:
        return user_message

    traits = ", ".join(persona_data.get("traits", []))
    # The worldview might be stored as a dict or just a string, so we need to handle both
    worldview_obj = persona_data.get("worldview", {})
    if isinstance(worldview_obj, dict):
        worldview = worldview_obj.get("type", str(worldview_obj))
    else:
        worldview = str(worldview_obj) if worldview_obj else "—"
    functions = persona_data.get("functions", [])

    # Grab the last 5 messages from the conversation for context
    context = "\n".join([
        f"{'User' if m['role']=='user' else persona_data['name']}: {m['content']}"
        for m in chat_history[-5:]
    ])

    rules = strategy_rules(strategy_name, enable_action_first=enable_action_first)

    return f"""You are {persona_data['name']}, a {persona_data['role']}.
Traits: {traits}
Worldview: {worldview}

Available functions (only for game data: stats, quests, items, locations, skills, recipes):
{json.dumps(functions, indent=2)}

{rules}

Previous conversation:
{context}

Respond to the next user message as {persona_data['name']} with the JSON format."""

def normalize_pred_schema(raw_obj):
    """
    Convert model output to:
    {
      "response": str,
      "function_call": {"name": str, "arguments": dict} | None,
      "num_calls": int,
      "text_before_call": bool
    }
    Accepts: correct schema, or common variants like {"function": "...", "parameters": {...}}, or plain text.
    """
    
    if isinstance(raw_obj, dict) and ("function_call" in raw_obj or "response" in raw_obj):
        call = raw_obj.get("function_call")
        call = call if isinstance(call, dict) else None
        response_text = (raw_obj.get("response") or "")
        return {
            "response": response_text,
            "function_call": call,
            "num_calls": 1 if call else 0,
            "text_before_call": bool(response_text and call),
        }

    # Sometimes the model uses "function" and "parameters" instead of "function_call" and "arguments"
    if isinstance(raw_obj, dict) and "function" in raw_obj and "parameters" in raw_obj:
        return {
            "response": (raw_obj.get("response") or ""),
            "function_call": {
                "name": raw_obj.get("function"),
                "arguments": raw_obj.get("parameters") or {},
            },
            "num_calls": 1,
            "text_before_call": False,
        }

    # If it's just a string or something we don't recognize, treat it as text with no function call
    return {
        "response": str(raw_obj),
        "function_call": None,
        "num_calls": 0,
        "text_before_call": False,
    }

def load_gold_for_utterance(utterance: str) -> tuple[Dict[str, Any], str]:
    """
    Looks up a gold row by exact user_utterance match in data/rpg_persona_dataset.jsonl.
    Falls back to a 'no-call' gold if not found.
    Returns: (gold_dict, gold_response)
    """
    # The dataset might be in a few different places depending on how the app is run
    possible_paths = [
        os.path.join("data", "rpg_persona_dataset.jsonl"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rpg_persona_dataset.jsonl"),
        os.path.join(os.path.dirname(__file__), "..", "data", "rpg_persona_dataset.jsonl"),
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        ex = json.loads(line)
                        
                        # See if this line matches what the user said
                        if ex.get("user_utterance") == utterance:
                            # Found it! Return the expected answer
                            gold = ex.get("gold", {"needs_call": False, "one_call_only": True})
                            gold_response = ex.get("gold_response", "")
                            return gold, gold_response
        except Exception:
            continue
    
    # If we couldn't find a match, assume no function call is needed
    return {"needs_call": False, "one_call_only": True}, ""

# --- Persona style snippets (removed for consistency) ---
def persona_style_prefix(persona_name: str):
    """
    Returns empty string for all personas to ensure consistency.
    Previously had persona-specific prefixes, but removed to avoid:
    - Inconsistency across personas (only 4/9 had prefixes)
    - Artificial boilerplate text in transcripts
    - Indirect influence on model behavior via chat history
    """
    return ""

def friendly_func_name(name: str) -> str:
    return name.replace("_", " ")

def narrate_call(persona_name: str, call: dict) -> str:
    """
    Turn a function call into a friendly, in-persona sentence.
    Example: {"name":"get_quest_info","arguments":{"quest_name":"The Lost Artifact","game":"Divinity: Original Sin 2"}}
    """
    if not call: 
        return ""
    prefix = persona_style_prefix(persona_name)
    fname  = friendly_func_name(call.get("name",""))
    args   = call.get("arguments", {}) or {}

    # Make each function call sound natural based on what it does
    if call.get("name") == "get_character_stats":
        who = args.get("character_name", "the character")
        game = args.get("game", "the game")
        text = f"I'm fetching {who}'s current stats in {game}."
    elif call.get("name") == "get_quest_info":
        q = args.get("quest_name", "the quest")
        game = args.get("game", "the game")
        text = f"I'm pulling details for the quest \"{q}\" in {game}."
    elif call.get("name") == "get_item_info":
        item = args.get("item_name", "the item")
        game = args.get("game", "the game")
        text = f"I'm looking up information on \"{item}\" in {game}."
    elif call.get("name") == "get_location_info":
        loc = args.get("location_name", "the location")
        game = args.get("game", "the game")
        text = f"I'm surveying threats and notes for {loc} in {game}."
    elif call.get("name") == "get_skill_tree":
        cls = args.get("character_class", "the class")
        game = args.get("game", "the game")
        text = f"I'm opening the {cls} skill tree in {game}."
    elif call.get("name") == "get_recipe_info":
        rec = args.get("recipe_name", "the recipe")
        game = args.get("game", "the game")
        text = f"I'm retrieving the full recipe for \"{rec}\" in {game}."
    else:
        # If we don't have a specific message for this function, just list the arguments
        flat = ", ".join(f"{k}={v}" for k, v in args.items())
        text = f"I'm executing {fname} with {flat or 'no arguments'}."

    # Add the persona name to the message
    if prefix:
        return f"{persona_name}: {prefix} {text}"
    else:
        return f"{persona_name}: {text}"

def narrate_result(persona_name: str, call: dict, tool_result: dict | str | None) -> str:
    """
    Optional: weave in a tiny hint from tool_result (kept generic to avoid schema coupling).
    Fall back to a simple confirmation line if result structure is unknown.
    """
    if tool_result is None:
        return ""
    
    if isinstance(tool_result, dict):
        #Summarizing the tool result.
        keys = list(tool_result.keys())[:3]
        if keys:
            kv = "; ".join(f"{k}: {tool_result[k]}" for k in keys)
            return f"{persona_name}: I found these key details — {kv}."
    # If we can't summarize it, just say we got something
    return f"{persona_name}: I've gathered the requested information."

def render_response_with_functions(pred: Dict[str, Any], persona_data: Dict[str, Any]):
    """Render response and function calls with proper display"""
    call = pred.get("function_call")
    persona_name = (persona_data or {}).get("name", "Assistant")

    # Start with whatever text the AI generated
    assistant_text = (pred.get("response") or "").strip()

    # If the AI wants to call a function, explain what it's doing in a friendly way
    if call:
        call_line = narrate_call(persona_name, call)
        if assistant_text:
            assistant_text = f"{assistant_text}\n\n{call_line}"
        else:
            assistant_text = call_line

    # Actually run the function and get the result
    tool_result = None
    if call:
        try:
            tool_result = execute(call)
        except Exception as e:
            tool_result = {"error": str(e)}

    # Add a quick summary of what we found
    if call:
        assistant_text = f"{assistant_text}\n\n{narrate_result(persona_name, call, tool_result)}".strip()

    # Make sure we always have something to show
    if not assistant_text:
        assistant_text = f"{persona_name}: How can I help?"
    
    st.markdown(assistant_text)

    # Show the raw function call details in an expandable section for debugging
    if call:
        with st.expander("Function call (debug)"):
            st.json({"name": call.get("name"), "arguments": call.get("arguments", {})})
        if tool_result is not None:
            with st.expander("Tool result (debug)"):
                st.json(tool_result)
    
    return assistant_text

# Set up the API clients automatically if we have keys from the .env file
if st.session_state.groq_api_key and st.session_state.groq_client is None:
    st.session_state.groq_client = initialize_groq_client(st.session_state.groq_api_key)
if st.session_state.openai_api_key and st.session_state.openai_client is None:
    st.session_state.openai_client = initialize_openai_client(st.session_state.openai_api_key)

# Build the sidebar where users configure everything
with st.sidebar:
    st.title("Configuration")
    
    # Let the user pick which API they want to use
    provider = st.radio(
        "API Provider",
        options=["openai", "groq"],
        index=0 if st.session_state.api_provider == "openai" else 1,
        help="Choose between OpenAI or GROQ API"
    )
    
    if provider != st.session_state.api_provider:
        st.session_state.api_provider = provider
        st.rerun()
    
    st.divider()
    
    # Show the right API key input field depending on which provider they picked
    if st.session_state.api_provider == "openai":
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key,
            help="Enter your OpenAI API key. Get one at https://platform.openai.com/api-keys"
        )
        
        if api_key_input and api_key_input != st.session_state.openai_api_key:
            st.session_state.openai_api_key = api_key_input
            st.session_state.openai_client = initialize_openai_client(api_key_input)
            if st.session_state.openai_client:
                st.success("OpenAI API key configured successfully!")
        
        # Let the user pick which OpenAI model to use
        model_options = {
            "GPT-4o Mini": "gpt-4o-mini",
            "GPT-4o": "gpt-4o",
            "GPT-4 Turbo": "gpt-4-turbo",
            "GPT-3.5 Turbo": "gpt-3.5-turbo",
            "GPT OSS 20B": "openai/gpt-oss-20b",
        }
        
        selected_model = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0,
            help="Choose the OpenAI model to use"
        )
    else:
        api_key_input = st.text_input(
            "GROQ API Key",
            type="password",
            value=st.session_state.groq_api_key,
            help="Enter your GROQ API key. Get one at https://console.groq.com/"
        )
        
        if api_key_input and api_key_input != st.session_state.groq_api_key:
            st.session_state.groq_api_key = api_key_input
            st.session_state.groq_client = initialize_groq_client(api_key_input)
            if st.session_state.groq_client:
                st.success("GROQ API key configured successfully!")
        
        # Letting the user pick which GROQ/Llama model to use
        model_options = {
            "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
            "Llama 3.1 8B Instant": "llama-3.1-8b-instant",
            "Llama 3 70B Versatile": "llama-3-70b-8192",
            "Llama 3 8B Instant": "llama-3-8b-8192",
            "Mixtral 8x7B": "mixtral-8x7b-32768",
        }
        
        selected_model = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0,
            help="Choose the Llama model to use"
        )
    
    st.divider()
    
    # Let the user choose how they want to prompt the AI
    st.subheader("🧠 Prompt Strategy")
    strategy_name = st.selectbox(
        "Choose strategy", 
        STRATEGIES, 
        index=STRATEGIES.index(st.session_state.selected_strategy) if st.session_state.selected_strategy in STRATEGIES else 0
    )
    
    if strategy_name != st.session_state.selected_strategy:
        st.session_state.selected_strategy = strategy_name
        st.rerun()
    
    st.divider()
    
    # Let the user pick which character to roleplay as
    st.subheader("Character Selection")
    persona_names = list(PERSONAS.keys())
    
    if not persona_names:
        st.warning("No personas available")
    else:
        selected_persona_name = st.selectbox(
            "Choose a Character",
            options=persona_names,
            index=0 if st.session_state.selected_persona is None else persona_names.index(st.session_state.selected_persona) if st.session_state.selected_persona in persona_names else 0,
            help="Select a character persona for the AI to roleplay"
        )
        
        if selected_persona_name != st.session_state.selected_persona:
            st.session_state.selected_persona = selected_persona_name
            st.session_state.messages = []  # Clear chat when persona changes
            st.rerun()
        
        # Show some info about the selected character
        if st.session_state.selected_persona:
            persona = PERSONAS[st.session_state.selected_persona]
            with st.expander(f"About {persona['name']}"):
                st.write(f"**Role:** {persona['role']}")
                st.write(f"**Traits:** {', '.join(persona['traits'])}")
                # The worldview might be stored as a dict or string, so handle both
                worldview = persona.get('worldview', {})
                if isinstance(worldview, dict):
                    worldview_display = worldview.get('type', worldview.get('alignment', 'N/A'))
                else:
                    worldview_display = str(worldview) if worldview else 'N/A'
                st.write(f"**Worldview:** {worldview_display}")
    
    st.divider()
    
    # Some extra options for how the AI should behave
    st.subheader("Prompting Options")
    enable_action_first = st.checkbox(
        "Enable ACTION-FIRST",
        value=st.session_state.get('enable_action_first', False),
        help="If enabled, function calls will come FIRST (before any persona text). The response field can be empty when a function is called."
    )
    
    if enable_action_first != st.session_state.get('enable_action_first', False):
        st.session_state.enable_action_first = enable_action_first
    
    st.divider()
    
    # Clear chat button
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Display chat info
    st.info(f"{len(st.session_state.messages)} messages in chat")

# Main chat interface
if st.session_state.selected_persona:
    persona = PERSONAS[st.session_state.selected_persona]
    st.title(f"Chatting as {persona['name']}")
    st.caption(f"The {persona['role']} - {', '.join(persona['traits'])} | Strategy: {st.session_state.selected_strategy}")
else:
    if st.session_state.api_provider == "openai":
        st.title("AI Chat with OpenAI")
        st.caption("Chat with OpenAI models (GPT-4o, GPT-3.5, etc.)")
    else:
        st.title("🦙 AI Chat with GROQ")
        st.caption("Chat with Meta's Llama models powered by GROQ")
    
    if not st.session_state.selected_persona:
        st.info("👆 Select a character from the sidebar to start roleplaying!")

# Check if we have an API key set up, otherwise show a warning
if st.session_state.api_provider == "openai":
    if not st.session_state.openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to start chatting.")
        st.info("Don't have an API key? Get one at https://platform.openai.com/api-keys")
    else:
        # Make sure the client is initialized if we haven't done that yet
        if st.session_state.openai_client is None:
            st.session_state.openai_client = initialize_openai_client(st.session_state.openai_api_key)
else:
    if not st.session_state.groq_api_key:
        st.warning("Please enter your GROQ API key in the sidebar to start chatting.")
        st.info("Don't have an API key? Get one at https://console.groq.com/")
    else:
        # doing the same thing for GROQ model
        if st.session_state.groq_client is None:
            st.session_state.groq_client = initialize_groq_client(st.session_state.groq_api_key)


# code for showing all the messages we've sent so far
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# This is where the user types their message
if prompt := st.chat_input("Type your message here..."):
    # Make sure they picked a character first
    if not st.session_state.selected_persona:
        st.error("Please select a character from the sidebar first!")
    # Check if API key and client are ready
    elif st.session_state.api_provider == "openai":
        if not st.session_state.openai_api_key:
            st.error("Please enter your OpenAI API key in the sidebar first.")
        elif st.session_state.openai_client is None:
            st.error("Failed to initialize OpenAI client. Please check your API key.")
        else:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get persona data
                    persona_data = PERSONAS[st.session_state.selected_persona]
                    
                    # Ensure persona has functions (safety guard)
                    if persona_data is not None and "functions" not in persona_data:
                        persona_data["functions"] = [
                            {"name": "get_character_stats", "description": "Retrieves character statistics", "parameters": {"character_name": "string", "game": "string"}},
                            {"name": "get_quest_info", "description": "Gets quest information", "parameters": {"quest_name": "string", "game": "string"}},
                            {"name": "get_item_info", "description": "Retrieves item information", "parameters": {"item_name": "string", "game": "string"}},
                            {"name": "get_location_info", "description": "Gets location information", "parameters": {"location_name": "string", "game": "string"}},
                            {"name": "get_skill_tree", "description": "Retrieves skill tree", "parameters": {"character_class": "string", "game": "string"}},
                            {"name": "get_recipe_info", "description": "Gets recipe information", "parameters": {"recipe_name": "string", "game": "string"}},
                        ]
                    
                    # Now we build the actual prompt that'll go to the AI, mixing in the persona's personality and the strategy rules
                    system_prompt = build_persona_prompt(
                        persona_data, 
                        prompt, 
                        st.session_state.messages, 
                        st.session_state.selected_strategy,
                        enable_action_first=st.session_state.get('enable_action_first', False)
                    )
                    
                    # Package everything up in the format the API expects - system message and user message
                    api_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    
                    # Time to actually call the AI and see what it says
                    model_name = model_options[selected_model]
                    response_raw = get_openai_response(
                        st.session_state.openai_client,
                        api_messages,
                        model_name
                    )
                    
                    # The response might be a JSON string or already parsed, so let's handle both cases gracefully
                    try:
                        raw_obj = json.loads(response_raw) if isinstance(response_raw, str) else response_raw
                    except Exception:
                        # If parsing fails, just use whatever we got
                        raw_obj = response_raw
                    
                    # Normalize it into our standard format so we can work with it consistently
                    pred = normalize_pred_schema(raw_obj)
                    
                    # Grab the "correct" answer from our test data and see how we did
                    gold, gold_response = load_gold_for_utterance(prompt)
                    metrics = score_example(pred, gold, gold_response)
                    
                    # Turn the AI's response into something nice to show the user
                    assistant_text = render_response_with_functions(pred, persona_data)
                    
                    # Show off how well we did (or didn't do) on this turn
                    st.subheader("Task-1 Metrics (this turn)")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Function Name Exact", "-" if metrics["fn_exact"] is None else metrics["fn_exact"])
                    c2.metric("Argument Exact", "-" if metrics["arg_exact"] is None else metrics["arg_exact"])
                    c3.metric("Over-Call", metrics["over_call"])
                    d1, d2 = st.columns(2)
                    d1.metric("Under-Call", metrics["under_call"])
                    d2.metric("Single-Shot Violation", metrics["single_shot_violation"])
                    
                    # If we have text quality scores, might as well show those too
                    if metrics.get("rouge_l_f1") is not None or metrics.get("bertscore_f1") is not None:
                        st.subheader("Text Quality Metrics")
                        t1, t2, t3 = st.columns(3)
                        if metrics.get("rouge_l_f1") is not None:
                            t1.metric("RougeL F1", f"{metrics['rouge_l_f1']:.3f}")
                        if metrics.get("bertscore_f1") is not None:
                            t2.metric("BERTScore F1", f"{metrics['bertscore_f1']:.3f}")
            
            # Add assistant response to chat
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
            
            # Explaining the code above:
            # 1. We grabbed the persona data and made sure it has functions to work with
            # 2. Built a prompt that includes the persona's personality, conversation history, and strategy rules
            # 3. Sent it to the AI API and got back a response
            # 4. Parsed and normalized the response into a consistent format
            # 5. Compared it to the "correct" answer from our test data to see how well it did
            # 6. Turned the response into something nice to display to the user
            # 7. Showed metrics so you can see how accurate the function calls and text were
    else:
        if not st.session_state.groq_api_key:
            st.error("Please enter your GROQ API key in the sidebar first.")
        elif st.session_state.groq_client is None:
            st.error("Failed to initialize GROQ client. Please check your API key.")
        else:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get persona data
                    persona_data = PERSONAS[st.session_state.selected_persona]
                    
                    # Ensure persona has functions (safety guard)
                    if persona_data is not None and "functions" not in persona_data:
                        persona_data["functions"] = [
                            {"name": "get_character_stats", "description": "Retrieves character statistics", "parameters": {"character_name": "string", "game": "string"}},
                            {"name": "get_quest_info", "description": "Gets quest information", "parameters": {"quest_name": "string", "game": "string"}},
                            {"name": "get_item_info", "description": "Retrieves item information", "parameters": {"item_name": "string", "game": "string"}},
                            {"name": "get_location_info", "description": "Gets location information", "parameters": {"location_name": "string", "game": "string"}},
                            {"name": "get_skill_tree", "description": "Retrieves skill tree", "parameters": {"character_class": "string", "game": "string"}},
                            {"name": "get_recipe_info", "description": "Gets recipe information", "parameters": {"recipe_name": "string", "game": "string"}},
                        ]
                    
                    # Now we build the actual prompt that'll go to the AI, mixing in the persona's personality and the strategy rules
                    system_prompt = build_persona_prompt(
                        persona_data, 
                        prompt, 
                        st.session_state.messages, 
                        st.session_state.selected_strategy,
                        enable_action_first=st.session_state.get('enable_action_first', False)
                    )
                    
                    # Package everything up in the format the API expects - system message and user message
                    api_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    
                    # Time to actually call the AI and see what it says
                    model_name = model_options[selected_model]
                    response_raw = get_groq_response(
                        st.session_state.groq_client,
                        api_messages,
                        model_name
                    )
                    
                    # The response might be a JSON string or already parsed, so let's handle both cases gracefully
                    try:
                        raw_obj = json.loads(response_raw) if isinstance(response_raw, str) else response_raw
                    except Exception:
                        # If parsing fails, just use whatever we got
                        raw_obj = response_raw
                    
                    # Normalize it into our standard format so we can work with it consistently
                    pred = normalize_pred_schema(raw_obj)
                    
                    # Grab the "correct" answer from our test data and see how we did
                    gold, gold_response = load_gold_for_utterance(prompt)
                    metrics = score_example(pred, gold, gold_response)
                    
                    # Turn the AI's response into something nice to show the user
                    assistant_text = render_response_with_functions(pred, persona_data)
                    
                    # Show off how well we did (or didn't do) on this turn
                    st.subheader("Function call Metrics")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Function Name Exact", "-" if metrics["fn_exact"] is None else metrics["fn_exact"])
                    c2.metric("Argument Exact", "-" if metrics["arg_exact"] is None else metrics["arg_exact"])
                    c3.metric("Over-Call", metrics["over_call"])
                    d1, d2 = st.columns(2)
                    d1.metric("Under-Call", metrics["under_call"])
                    d2.metric("Single-Shot Violation", metrics["single_shot_violation"])
                    
                    # If we have text quality scores, might as well show those too
                    if metrics.get("rouge_l_f1") is not None or metrics.get("bertscore_f1") is not None:
                        st.subheader("Text Quality Metrics")
                        t1, t2, t3 = st.columns(3)
                        if metrics.get("rouge_l_f1") is not None:
                            t1.metric("RougeL F1", f"{metrics['rouge_l_f1']:.3f}")
                        if metrics.get("bertscore_f1") is not None:
                            t2.metric("BERTScore F1", f"{metrics['bertscore_f1']:.3f}")
            
            # Add assistant response to chat
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
            
            # What just happened above (GROQ version):
            # Same flow as OpenAI - grab persona, build prompt, call API, parse response,
            # compare to gold standard, display nicely, and show metrics

# Batch Evaluation Section
st.markdown("---")
st.subheader("Evaluation Options")

# Two tabs: one for quick eval, one for full comparison
eval_tab1, eval_tab2 = st.tabs(["Quick Batch Eval", "Full Evaluation & Comparison"])

with eval_tab1:
    st.subheader(" Quick Batch Evaluation")
    st.caption("Run evaluation for the currently selected strategy (immediate results)")
    
    # Row limit control
    col_limit, col_info = st.columns([1, 2])
    with col_limit:
        max_rows = st.number_input(
            "Max rows to evaluate",
            min_value=1,
            max_value=120,
            value=120,
            key="quick_batch_max_rows",
            help="Limit the number of dataset rows to evaluate (1-120)"
        )
    with col_info:
        st.caption(f"Evaluating {max_rows} rows will be faster than the full dataset")
    
    run_batch = st.button("Run Batch Eval with current Strategy", use_container_width=True)
    
    if run_batch:
        dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rpg_persona_dataset.jsonl")
        
        # Try a few different places where the dataset might be
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join("data", "rpg_persona_dataset.jsonl")
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "rpg_persona_dataset.jsonl")

        # Load the dataset file
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except FileNotFoundError:
            st.error("Dataset not found at data/rpg_persona_dataset.jsonl")
            st.stop()

        total_original = len(lines)
        if total_original == 0:
            st.warning("Dataset file is empty.")
            st.stop()
        
        # Only evaluate up to the limit the user set
        lines = lines[:max_rows]
        total = len(lines)
        if total < total_original:
            st.info(f"Evaluating {total} rows (limited from {total_original} total)")

        progress = st.progress(0)
        per_item_rows = []

        # Go through each example in the dataset
        for idx, line in enumerate(lines, start=1):
            ex = json.loads(line)

            # Use the persona and tools from the dataset item so we're testing fairly
            persona_data = ex.get("persona", {}) or {}
            if "functions" not in persona_data:
                persona_data["functions"] = [
                    {"name": "get_character_stats", "description": "Retrieves character statistics", "parameters": {"character_name": "string", "game": "string"}},
                    {"name": "get_quest_info", "description": "Gets quest information", "parameters": {"quest_name": "string", "game": "string"}},
                    {"name": "get_item_info", "description": "Retrieves item information", "parameters": {"item_name": "string", "game": "string"}},
                    {"name": "get_location_info", "description": "Gets location information", "parameters": {"location_name": "string", "game": "string"}},
                    {"name": "get_skill_tree", "description": "Retrieves skill tree", "parameters": {"character_class": "string", "game": "string"}},
                    {"name": "get_recipe_info", "description": "Gets recipe information", "parameters": {"recipe_name": "string", "game": "string"}},
                ]

            user_utterance = ex.get("user_utterance", "")
            gold = ex.get("gold", {})

            # Build the prompt with the strategy rules baked in
            system_prompt = build_persona_prompt(
                persona_data=persona_data,
                user_message=user_utterance,
                chat_history=[],                # batch = single-turn, so no history
                strategy_name=st.session_state.selected_strategy,
                enable_action_first=st.session_state.get('enable_action_first', False)
            )

            # Create the messages in the format the API wants
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_utterance},
            ]

            # Call the model - use whichever API provider is selected
            if st.session_state.api_provider == "openai" and st.session_state.openai_client:
                pred_raw = get_openai_response(
                    st.session_state.openai_client,
                    messages,
                    model_options[selected_model]
                )
            elif st.session_state.api_provider == "groq" and st.session_state.groq_client:
                pred_raw = get_groq_response(
                    st.session_state.groq_client,
                    messages,
                    model_options[selected_model]
                )
            else:
                st.error("No API client available")
                break

            # Normalize the model output so we can score it
            if isinstance(pred_raw, dict) and ("response" in pred_raw or "function_call" in pred_raw):
                pred = normalize_pred_schema(pred_raw)
            else:
                # Try to parse it as JSON if it's a string
                try:
                    obj = json.loads(pred_raw) if isinstance(pred_raw, str) else pred_raw
                except Exception:
                    obj = pred_raw
                pred = normalize_pred_schema(obj)

            # Score this example against the gold standard
            gold_response = ex.get("gold_response", "")
            m = score_example(pred, gold, gold_response)
            m["id"] = ex.get("id", f"item_{idx}")
            m["strategy"] = st.session_state.selected_strategy
            m["persona"] = persona_data.get("name", "Unknown")
            m["user_utterance"] = user_utterance[:50] + "..." if len(user_utterance) > 50 else user_utterance
            per_item_rows.append(m)

            progress.progress(idx / total)

        # Show the results and aggregate them
        st.success(f"Batch evaluation completed on {total} items." + (f" (Limited from {total_original} total)" if total < total_original else ""))
        df = pd.DataFrame(per_item_rows)
        st.dataframe(df, use_container_width=True)

        # What we just did:
        # We went through each example in the dataset, sent it to the AI with the selected strategy,
        # got back responses, normalized them, scored them against the gold standard, and collected
        # all the metrics. Now we're showing the per-item results and computing averages.

        st.subheader("Aggregate (Avg metrics)")
        summary = aggregate_rows(per_item_rows)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Function Name Exact (avg)", f"{summary['fn_exact']:.3f}" if summary['fn_exact'] is not None else "-")
            st.metric("Argument Exact (avg)", f"{summary['arg_exact']:.3f}" if summary['arg_exact'] is not None else "-")
        with col2:
            st.metric("Over-Call (avg)", f"{summary['over_call']:.3f}" if summary['over_call'] is not None else "-")
            st.metric("Under-Call (avg)", f"{summary['under_call']:.3f}" if summary['under_call'] is not None else "-")
        with col3:
            st.metric("Single-Shot Violation (avg)", f"{summary['single_shot_violation']:.3f}" if summary['single_shot_violation'] is not None else "-")
        
        # Show text quality scores if we have them
        if summary.get("rouge_l_f1") is not None or summary.get("bertscore_f1") is not None:
            st.subheader("Text Quality Metrics (avg)")
            t1, t2 = st.columns(2)
            if summary.get("rouge_l_f1") is not None:
                t1.metric("RougeL F1 (avg)", f"{summary['rouge_l_f1']:.3f}")
            if summary.get("bertscore_f1") is not None:
                t2.metric("BERTScore F1 (avg)", f"{summary['bertscore_f1']:.3f}")
        
        st.json(summary)

        # Let them download the results as a CSV file
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download per-item metrics (CSV)",
            data=csv,
            file_name=f"task1_metrics_{st.session_state.selected_strategy.replace(' ', '_').lower()}.csv",
            mime="text/csv"
        )

with eval_tab2:
    st.subheader("Full Evaluation & Comparison")
    st.caption("Two-phase evaluation: Collect responses first, then compute metrics. Compare multiple strategies side-by-side.")
    
    # Map the display names to the internal names we use
    STRATEGY_MAPPING = {
        "Zero-Shot": "zero_shot",
        "Few-Shot": "few_shot",
        "Chain of Thought": "chain_of_thought",
        "Persona Sandwich": "persona_sandwich"
    }
    REVERSE_MAPPING = {v: k for k, v in STRATEGY_MAPPING.items()}
    
    # Phase 1: Collect Responses
    st.markdown("### Phase 1: Collect Responses")
    st.caption("Select strategies and collect model responses (FAST - no metric computation)")
    
    # Row limit control
    col_limit, col_info = st.columns([1, 2])
    with col_limit:
        max_rows_eval = st.number_input(
            "Max rows to evaluate",
            min_value=1,
            max_value=120,
            value=120,
            key="full_eval_max_rows",
            help="Limit the number of dataset rows to evaluate (1-120)"
        )
    with col_info:
        st.caption(f"Evaluating {max_rows_eval} rows will be faster than the full dataset")
    
    selected_strategies_ui = st.multiselect(
        "Select Strategies to Evaluate:",
        options=list(STRATEGY_MAPPING.keys()),
        default=["Zero-Shot"],
        help="Select one or more strategies to evaluate"
    )
    
    # Convert the nice display names to the internal names we use in code
    selected_strategies = [STRATEGY_MAPPING[s] for s in selected_strategies_ui]
    
    collect_btn = st.button("Collect Responses", use_container_width=True, type="primary")
    
    if collect_btn:
        if not selected_strategies:
            st.error("Please select at least one strategy!")
        else:
            try:
                from run_evaluation import collect_responses
                import sys
                
                # Look for the dataset in a few possible locations
                dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rpg_persona_dataset.jsonl")
                if not os.path.exists(dataset_path):
                    dataset_path = os.path.join("data", "rpg_persona_dataset.jsonl")
                if not os.path.exists(dataset_path):
                    dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "rpg_persona_dataset.jsonl")
                
                if not os.path.exists(dataset_path):
                    st.error("Dataset not found at data/rpg_persona_dataset.jsonl")
                else:
                    with st.spinner(f"Collecting responses for {len(selected_strategies_ui)} strategy/strategies..."):
                        # Figure out which model to use - default to something reasonable if we can't tell
                        try:
                            if st.session_state.api_provider == "openai":
                                model_name = "gpt-4o-mini"
                            else:
                                model_name = "llama-3.3-70b-versatile"
                        except:
                            model_name = "gpt-4o-mini"
                        
                        # Get the API key for whichever provider they're using
                        api_key = None
                        if st.session_state.api_provider == "groq":
                            api_key = st.session_state.groq_api_key
                        else:
                            api_key = st.session_state.openai_api_key
                        
                        results = collect_responses(
                            dataset_path=dataset_path,
                            strategies=selected_strategies,
                            output_dir="outputs",
                            model=model_name,
                            api_provider=st.session_state.api_provider,
                            api_key=api_key,
                            max_rows=max_rows_eval,
                            enable_action_first=st.session_state.get('enable_action_first', False)
                        )
                        
                        st.success(f"Collected responses for {len(results)} strategy/strategies!")
                        for strategy in selected_strategies:
                            st.info(f"Saved: `outputs/{strategy}_responses.jsonl`")
            except Exception as e:
                st.error(f"Error collecting responses: {str(e)}")
                st.exception(e)
    
    st.divider()
    
    # Phase 2: Compute Metrics
    st.markdown("### Phase 2: Compute Metrics")
    st.caption("Compute BERTScore and RougeL from saved responses (SLOW - can take time)")
    
    # See what response files we already have saved
    import glob
    response_files = glob.glob("outputs/*_responses.jsonl")
    available_strategies = []
    for f in response_files:
        strategy_name = os.path.basename(f).replace("_responses.jsonl", "")
        if strategy_name in REVERSE_MAPPING:
            available_strategies.append(strategy_name)
    
    if available_strategies:
        st.info(f"Found response files for: {', '.join([REVERSE_MAPPING.get(s, s) for s in available_strategies])}")
        metrics_strategies = st.multiselect(
            "Select strategies to compute metrics for:",
            options=available_strategies,
            default=available_strategies,
            format_func=lambda x: REVERSE_MAPPING.get(x, x)
        )
        
        compute_btn = st.button("Compute Metrics", use_container_width=True, type="primary")
        
        if compute_btn:
            if not metrics_strategies:
                st.error("Please select at least one strategy!")
            else:
                try:
                    from run_evaluation import compute_metrics_from_responses
                    
                    with st.spinner(f"Computing metrics for {len(metrics_strategies)} strategy/strategies (this may take a while)..."):
                        
                        device = "cpu"
                        
                        metrics = compute_metrics_from_responses(
                            strategies=metrics_strategies,
                            output_dir="outputs",
                            device=device
                        )
                        
                        st.success(f"Computed metrics for {len(metrics)} strategy/strategies!")
                        for strategy in metrics_strategies:
                            st.info(f"Saved: `outputs/{strategy}_metrics.jsonl`")
                except Exception as e:
                    st.error(f"Error computing metrics: {str(e)}")
                    st.exception(e)
    else:
        st.warning("No response files found. Please run Phase 1 first.")
    
    st.divider()
    
    # Phase 3: Generate Comparison
    st.markdown("### Phase 3: Generate Comparison")
    st.caption("Compare all strategies side-by-side")
    
    # See what metric files we have available
    metric_files = glob.glob("outputs/*_metrics.jsonl")
    available_metric_strategies = []
    for f in metric_files:
        strategy_name = os.path.basename(f).replace("_metrics.jsonl", "")
        if strategy_name in REVERSE_MAPPING:
            available_metric_strategies.append(strategy_name)
    
    if available_metric_strategies:
        compare_btn = st.button("Generate Comparison", use_container_width=True, type="primary")
        
        if compare_btn:
            try:
                from run_evaluation import compare_strategies
                
                with st.spinner("Generating comparison..."):
                    comparison = compare_strategies(
                        strategies=available_metric_strategies,
                        output_dir="outputs"
                    )
                    
                    st.success("Comparison generated!")
                    
                    # Showing the comparison in a nice table
                    st.subheader("Strategy Comparison")
                    
                    # First show how well each strategy did at calling functions correctly
                    st.write("**Function Call Accuracy:**")
                    fn_data = {
                        "Strategy": [],
                        "Function Accuracy": [],
                        "Argument Accuracy": [],
                        "Over-Call": [],
                        "Under-Call": []
                    }
                    
                    for strategy, metrics in comparison.items():
                        display_name = REVERSE_MAPPING.get(strategy, strategy)
                        fn_data["Strategy"].append(display_name)
                        fn_data["Function Accuracy"].append(f"{metrics.get('fn_exact', 0):.3f}" if metrics.get('fn_exact') is not None else "N/A")
                        fn_data["Argument Accuracy"].append(f"{metrics.get('arg_exact', 0):.3f}" if metrics.get('arg_exact') is not None else "N/A")
                        fn_data["Over-Call"].append(f"{metrics.get('over_call', 0):.3f}" if metrics.get('over_call') is not None else "N/A")
                        fn_data["Under-Call"].append(f"{metrics.get('under_call', 0):.3f}" if metrics.get('under_call') is not None else "N/A")
                    
                    fn_df = pd.DataFrame(fn_data)
                    st.dataframe(fn_df, use_container_width=True, hide_index=True)
                    
                    # showing the text quality metrics.
                    st.write("**Text Quality Metrics:**")
                    text_data = {
                        "Strategy": [],
                        "RougeL F1": [],
                        "BERTScore F1": [],
                        "RougeL Precision": [],
                        "RougeL Recall": [],
                        "BERTScore Precision": [],
                        "BERTScore Recall": []
                    }
                    
                    for strategy, metrics in comparison.items():
                        display_name = REVERSE_MAPPING.get(strategy, strategy)
                        text_data["Strategy"].append(display_name)
                        text_data["RougeL F1"].append(f"{metrics.get('rouge_l_f1', 0):.3f}" if metrics.get('rouge_l_f1') is not None else "N/A")
                        text_data["BERTScore F1"].append(f"{metrics.get('bertscore_f1', 0):.3f}" if metrics.get('bertscore_f1') is not None else "N/A")
                        text_data["RougeL Precision"].append(f"{metrics.get('rouge_l_precision', 0):.3f}" if metrics.get('rouge_l_precision') is not None else "N/A")
                        text_data["RougeL Recall"].append(f"{metrics.get('rouge_l_recall', 0):.3f}" if metrics.get('rouge_l_recall') is not None else "N/A")
                        text_data["BERTScore Precision"].append(f"{metrics.get('bertscore_precision', 0):.3f}" if metrics.get('bertscore_precision') is not None else "N/A")
                        text_data["BERTScore Recall"].append(f"{metrics.get('bertscore_recall', 0):.3f}" if metrics.get('bertscore_recall') is not None else "N/A")
                    
                    text_df = pd.DataFrame(text_data)
                    st.dataframe(text_df, use_container_width=True, hide_index=True)
                    
                    # Let them download the comparison as a CSV
                    comparison_csv = text_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Comparison (CSV)",
                        data=comparison_csv,
                        file_name="strategy_comparison.csv",
                        mime="text/csv"
                    )
                    
                    # Show the raw JSON data in case they want to see everything
                    with st.expander("📄 Full Comparison Data (JSON)"):
                        st.json(comparison)
                        
            except Exception as e:
                st.error(f"Error generating comparison: {str(e)}")
                st.exception(e)
    else:
        st.warning("No metric files found. Please run Phase 1 and Phase 2 first.")

# Footer
st.divider()
if st.session_state.api_provider == "openai":
    st.caption("Powered by OpenAI API")
else:
    st.caption("Powered by GROQ API | Using Meta Llama Models")

#References:
# GROQ API:
# https://console.groq.com/docs/python-sdk
# https://github.com/groq/groq-python

# Streamlit:
# https://docs.streamlit.io/
# https://github.com/streamlit/streamlit

