import streamlit as st
from groq import Groq
from openai import OpenAI
import os
import json
import sys
import pandas as pd
from typing import List, Dict, Any, Optional
from functools import lru_cache
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Import persona_loader ---
try:
    from persona_loader import load_personas_from_dataset, get_default_personas
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "persona_loader", os.path.join(os.path.dirname(__file__), "persona_loader.py")
    )
    persona_loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(persona_loader)
    load_personas_from_dataset = persona_loader.load_personas_from_dataset
    get_default_personas = persona_loader.get_default_personas

# --- Import function executor ---
try:
    from src.function_executor import execute
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "function_executor", os.path.join(os.path.dirname(__file__), "src", "function_executor.py")
    )
    fe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fe)
    execute = fe.execute

# --- Import evaluation ---
try:
    from evaluation import score_example, METRIC_KEYS, aggregate_rows
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "evaluation", os.path.join(os.path.dirname(__file__), "evaluation.py")
    )
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    score_example = evaluation.score_example
    METRIC_KEYS = evaluation.METRIC_KEYS
    aggregate_rows = evaluation.aggregate_rows

# --- Load .env ---
for env_path in [
    os.path.join(os.path.dirname(__file__), ".env"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
]:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        break
else:
    load_dotenv()

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
STRATEGIES = ["Zero-Shot", "Few-Shot", "Chain of Thought", "Persona Sandwich"]

DEFAULT_FUNCTIONS = [
    {"name": "get_character_stats", "description": "Retrieves character statistics", "parameters": {"character_name": "string", "game": "string"}},
    {"name": "get_quest_info", "description": "Gets quest information", "parameters": {"quest_name": "string", "game": "string"}},
    {"name": "get_item_info", "description": "Retrieves item information", "parameters": {"item_name": "string", "game": "string"}},
    {"name": "get_location_info", "description": "Gets location information", "parameters": {"location_name": "string", "game": "string"}},
    {"name": "get_skill_tree", "description": "Retrieves skill tree", "parameters": {"character_class": "string", "game": "string"}},
    {"name": "get_recipe_info", "description": "Gets recipe information", "parameters": {"recipe_name": "string", "game": "string"}},
]

BASE_JSON_SCHEMA = (
    "OUTPUT JSON ONLY (exact keys):\n"
    "{\n"
    '  "response": "<short in-character line>",\n'
    '  "function_call": {"name": "<function name>", "arguments": { /* exact keys/values */ }}\n'
    "}\n"
    "If no function is needed, omit 'function_call'.\n"
    "IMPORTANT: Do NOT use 'function'/'parameters' — only use 'function_call'/'arguments'."
)

OPENAI_MODELS = {
    "GPT-4o Mini": "gpt-4o-mini",
    "GPT-4o": "gpt-4o",
    "GPT-4 Turbo": "gpt-4-turbo",
    "GPT-3.5 Turbo": "gpt-3.5-turbo",
}

GROQ_MODELS = {
    "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
    "Llama 3.1 8B Instant": "llama-3.1-8b-instant",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
}

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CPDC 2025 — Persona Dialogue Agent",
    page_icon="🎮",
    layout="wide"
)

# ─────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────
DEFAULTS = {
    "messages": [],
    "api_provider": "groq",
    "selected_persona": None,
    "selected_strategy": STRATEGIES[0],
    "groq_api_key": os.getenv("GROQ_API_KEY", ""),
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "groq_client": None,
    "openai_client": None,
    "enable_action_first": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# Load personas
# ─────────────────────────────────────────────
try:
    PERSONAS = load_personas_from_dataset()
except Exception:
    PERSONAS = get_default_personas()

# ─────────────────────────────────────────────
# Dataset cache — load once, not per message
# ─────────────────────────────────────────────
@st.cache_data
def load_dataset_index() -> Dict[str, dict]:
    """Load the gold dataset into a dict keyed by user_utterance for O(1) lookup."""
    index = {}
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "rpg_persona_dataset.jsonl"),
        os.path.join("rpg_persona_dataset.jsonl"),
        os.path.join("data", "rpg_persona_dataset.jsonl"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ex = json.loads(line)
                        utt = ex.get("user_utterance", "")
                        if utt:
                            index[utt] = ex
                    except Exception:
                        continue
            break
    return index

DATASET_INDEX = load_dataset_index()

def load_gold_for_utterance(utterance: str):
    ex = DATASET_INDEX.get(utterance)
    if ex:
        return ex.get("gold", {"needs_call": False, "one_call_only": True}), ex.get("gold_response", "")
    return {"needs_call": False, "one_call_only": True}, ""

# ─────────────────────────────────────────────
# Client helpers
# ─────────────────────────────────────────────
def init_client(provider: str, api_key: str):
    try:
        if provider == "openai":
            return OpenAI(api_key=api_key)
        else:
            return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing {provider} client: {e}")
        return None

def get_llm_response(provider: str, client, messages: List[Dict], model: str) -> str:
    """Single unified function to call either OpenAI or GROQ."""
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

def get_active_client():
    provider = st.session_state.api_provider
    return st.session_state.get(f"{provider}_client")

def get_active_model(model_options: dict, selected_label: str) -> str:
    return model_options.get(selected_label, list(model_options.values())[0])

# ─────────────────────────────────────────────
# Prompt building
# ─────────────────────────────────────────────
def strategy_rules(strategy_name: str, enable_action_first: bool = False) -> str:
    if strategy_name == "Zero-Shot":
        return "Be helpful, concise, and stay in character.\nUse a function when needed.\n\n" + BASE_JSON_SCHEMA
    elif strategy_name == "Few-Shot":
        return "Stay strictly in character. Use a function when clearly needed.\nKeep replies concise.\n\n" + BASE_JSON_SCHEMA
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
                "1) ACTION-FIRST: If the user asks for retrievable game data (stats, quests, items, "
                "locations, skills, recipes), call the correct function FIRST.\n"
            )
        rules.append("2) SINGLE-SHOT: Make at most one function call this turn.\n")
        rules.append("3) NO-CALL FOR CHITCHAT/AMBIGUITY: For opinions/feelings/ambiguous asks, do NOT call any function.\n")
        rules.append("4) " + BASE_JSON_SCHEMA + "\n")
        if enable_action_first:
            rules.append("If you call a function, you may leave 'response' empty.")
        return "RULES:\n" + "".join(rules)


def build_persona_prompt(
    persona_data: Dict[str, Any],
    user_message: str,
    chat_history: List[Dict[str, str]],
    strategy_name: str,
    enable_action_first: bool = False,
) -> str:
    if not persona_data:
        return user_message

    traits = ", ".join(persona_data.get("traits", []))
    worldview_obj = persona_data.get("worldview", {})
    worldview = worldview_obj.get("type", str(worldview_obj)) if isinstance(worldview_obj, dict) else str(worldview_obj or "—")
    functions = persona_data.get("functions", [])
    context = "\n".join([
        f"{'User' if m['role'] == 'user' else persona_data['name']}: {m['content']}"
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


# ─────────────────────────────────────────────
# Response normalization
# ─────────────────────────────────────────────
def normalize_pred_schema(raw_obj) -> Dict[str, Any]:
    if isinstance(raw_obj, dict) and ("function_call" in raw_obj or "response" in raw_obj):
        call = raw_obj.get("function_call")
        call = call if isinstance(call, dict) else None
        return {
            "response": (raw_obj.get("response") or ""),
            "function_call": call,
            "num_calls": 1 if call else 0,
            "text_before_call": bool((raw_obj.get("response") or "") and call),
        }
    if isinstance(raw_obj, dict) and "function" in raw_obj and "parameters" in raw_obj:
        return {
            "response": (raw_obj.get("response") or ""),
            "function_call": {"name": raw_obj["function"], "arguments": raw_obj.get("parameters") or {}},
            "num_calls": 1,
            "text_before_call": False,
        }
    return {"response": str(raw_obj), "function_call": None, "num_calls": 0, "text_before_call": False}


# ─────────────────────────────────────────────
# Function call narration
# ─────────────────────────────────────────────
def narrate_call(persona_name: str, call: dict) -> str:
    if not call:
        return ""
    args = call.get("arguments", {}) or {}
    name = call.get("name", "")
    narrations = {
        "get_character_stats": f"I'm fetching {args.get('character_name', 'the character')}'s current stats in {args.get('game', 'the game')}.",
        "get_quest_info": f"I'm pulling details for the quest \"{args.get('quest_name', 'the quest')}\" in {args.get('game', 'the game')}.",
        "get_item_info": f"I'm looking up information on \"{args.get('item_name', 'the item')}\" in {args.get('game', 'the game')}.",
        "get_location_info": f"I'm surveying threats and notes for {args.get('location_name', 'the location')} in {args.get('game', 'the game')}.",
        "get_skill_tree": f"I'm opening the {args.get('character_class', 'class')} skill tree in {args.get('game', 'the game')}.",
        "get_recipe_info": f"I'm retrieving the full recipe for \"{args.get('recipe_name', 'the recipe')}\" in {args.get('game', 'the game')}.",
    }
    text = narrations.get(name, f"I'm executing {name.replace('_', ' ')} with {', '.join(f'{k}={v}' for k, v in args.items()) or 'no arguments'}.")
    return f"{persona_name}: {text}"


def narrate_result(persona_name: str, tool_result) -> str:
    if not tool_result:
        return ""
    if isinstance(tool_result, dict):
        keys = list(tool_result.keys())[:3]
        if keys:
            kv = "; ".join(f"{k}: {tool_result[k]}" for k in keys)
            return f"{persona_name}: I found these key details — {kv}."
    return f"{persona_name}: I've gathered the requested information."


# ─────────────────────────────────────────────
# Render response
# ─────────────────────────────────────────────
def render_response_with_functions(pred: Dict[str, Any], persona_data: Dict[str, Any]) -> str:
    call = pred.get("function_call")
    persona_name = (persona_data or {}).get("name", "Assistant")
    assistant_text = (pred.get("response") or "").strip()

    if call:
        call_line = narrate_call(persona_name, call)
        assistant_text = f"{assistant_text}\n\n{call_line}".strip() if assistant_text else call_line

    tool_result = None
    if call:
        try:
            tool_result = execute(call)
        except Exception as e:
            tool_result = {"error": str(e)}
        assistant_text = f"{assistant_text}\n\n{narrate_result(persona_name, tool_result)}".strip()

    if not assistant_text:
        assistant_text = f"{persona_name}: How can I help?"

    st.markdown(assistant_text)

    if call:
        with st.expander("🔧 Function call (debug)"):
            st.json({"name": call.get("name"), "arguments": call.get("arguments", {})})
        if tool_result is not None:
            with st.expander("📦 Tool result (debug)"):
                st.json(tool_result)

    return assistant_text


# ─────────────────────────────────────────────
# Metrics display
# ─────────────────────────────────────────────
def display_metrics(metrics: dict):
    st.subheader("📊 Turn Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Fn Name Exact", "-" if metrics["fn_exact"] is None else metrics["fn_exact"])
    c2.metric("Arg Exact", "-" if metrics["arg_exact"] is None else metrics["arg_exact"])
    c3.metric("Over-Call", metrics["over_call"])
    d1, d2 = st.columns(2)
    d1.metric("Under-Call", metrics["under_call"])
    d2.metric("Single-Shot Violation", metrics["single_shot_violation"])

    if metrics.get("rouge_l_f1") is not None or metrics.get("bertscore_f1") is not None:
        st.subheader("📝 Text Quality")
        t1, t2 = st.columns(2)
        if metrics.get("rouge_l_f1") is not None:
            t1.metric("RougeL F1", f"{metrics['rouge_l_f1']:.3f}")
        if metrics.get("bertscore_f1") is not None:
            t2.metric("BERTScore F1", f"{metrics['bertscore_f1']:.3f}")


# ─────────────────────────────────────────────
# Core chat handler (single unified function)
# ─────────────────────────────────────────────
def handle_chat(prompt: str, model_options: dict, selected_model_label: str):
    provider = st.session_state.api_provider
    client = get_active_client()

    if not client:
        st.error(f"No {provider} client. Please enter your API key in the sidebar.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            persona_data = PERSONAS.get(st.session_state.selected_persona, {})
            if "functions" not in persona_data:
                persona_data["functions"] = DEFAULT_FUNCTIONS

            system_prompt = build_persona_prompt(
                persona_data,
                prompt,
                st.session_state.messages,
                st.session_state.selected_strategy,
                enable_action_first=st.session_state.enable_action_first,
            )

            api_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            model_name = get_active_model(model_options, selected_model_label)
            response_raw = get_llm_response(provider, client, api_messages, model_name)

            try:
                raw_obj = json.loads(response_raw) if isinstance(response_raw, str) else response_raw
            except Exception:
                raw_obj = response_raw

            pred = normalize_pred_schema(raw_obj)
            gold, gold_response = load_gold_for_utterance(prompt)
            metrics = score_example(pred, gold, gold_response)
            assistant_text = render_response_with_functions(pred, persona_data)
            display_metrics(metrics)

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})


# ─────────────────────────────────────────────
# Auto-initialize clients from env keys
# ─────────────────────────────────────────────
if st.session_state.groq_api_key and st.session_state.groq_client is None:
    st.session_state.groq_client = init_client("groq", st.session_state.groq_api_key)
if st.session_state.openai_api_key and st.session_state.openai_client is None:
    st.session_state.openai_client = init_client("openai", st.session_state.openai_api_key)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    provider = st.radio(
        "API Provider",
        options=["groq", "openai"],
        index=0 if st.session_state.api_provider == "groq" else 1,
        help="GROQ is free and fast. OpenAI requires a paid key.",
    )
    if provider != st.session_state.api_provider:
        st.session_state.api_provider = provider
        st.rerun()

    st.divider()

    # API key input + model selector — unified for both providers
    if provider == "groq":
        model_options = GROQ_MODELS
        key_label, key_help, key_url = "GROQ API Key", "Get one free at https://console.groq.com/", "groq_api_key"
    else:
        model_options = OPENAI_MODELS
        key_label, key_help, key_url = "OpenAI API Key", "Get one at https://platform.openai.com/api-keys", "openai_api_key"

    api_key_input = st.text_input(key_label, type="password", value=st.session_state[key_url], help=key_help)
    if api_key_input and api_key_input != st.session_state[key_url]:
        st.session_state[key_url] = api_key_input
        st.session_state[f"{provider}_client"] = init_client(provider, api_key_input)
        if st.session_state[f"{provider}_client"]:
            st.success("✅ API key configured!")

    selected_model_label = st.selectbox("Model", options=list(model_options.keys()), index=0)

    st.divider()

    # Strategy
    st.subheader("🧠 Prompt Strategy")
    strategy_name = st.selectbox(
        "Choose strategy",
        STRATEGIES,
        index=STRATEGIES.index(st.session_state.selected_strategy),
    )
    if strategy_name != st.session_state.selected_strategy:
        st.session_state.selected_strategy = strategy_name
        st.rerun()

    st.divider()

    # Persona
    st.subheader("🎭 Character")
    persona_names = list(PERSONAS.keys())
    if persona_names:
        default_idx = 0
        if st.session_state.selected_persona in persona_names:
            default_idx = persona_names.index(st.session_state.selected_persona)

        selected_persona_name = st.selectbox("Choose a Character", options=persona_names, index=default_idx)
        if selected_persona_name != st.session_state.selected_persona:
            st.session_state.selected_persona = selected_persona_name
            st.session_state.messages = []
            st.rerun()

        if st.session_state.selected_persona:
            persona = PERSONAS[st.session_state.selected_persona]
            with st.expander(f"About {persona['name']}"):
                st.write(f"**Role:** {persona['role']}")
                st.write(f"**Traits:** {', '.join(persona['traits'])}")
                worldview = persona.get("worldview", {})
                wv_display = worldview.get("type", worldview.get("alignment", "N/A")) if isinstance(worldview, dict) else str(worldview)
                st.write(f"**Worldview:** {wv_display}")

    st.divider()

    # Options
    st.subheader("🔧 Options")
    enable_action_first = st.checkbox(
        "Enable ACTION-FIRST",
        value=st.session_state.enable_action_first,
        help="Function calls come before persona text.",
    )
    st.session_state.enable_action_first = enable_action_first

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.info(f"💬 {len(st.session_state.messages)} messages in chat")

# ─────────────────────────────────────────────
# Main chat UI
# ─────────────────────────────────────────────
if st.session_state.selected_persona:
    persona = PERSONAS[st.session_state.selected_persona]
    st.title(f"🎮 {persona['name']}")
    st.caption(f"*{persona['role']}* · {', '.join(persona['traits'])} · Strategy: **{st.session_state.selected_strategy}**")
else:
    st.title("🎮 CPDC 2025 — Persona Dialogue Agent")
    st.info("👈 Select a character from the sidebar to start.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message here..."):
    if not st.session_state.selected_persona:
        st.error("Please select a character from the sidebar first!")
    else:
        handle_chat(prompt, model_options, selected_model_label)

# ─────────────────────────────────────────────
# Batch Evaluation
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("🧪 Batch Evaluation")

eval_tab1, eval_tab2 = st.tabs(["Quick Batch Eval", "Strategy Comparison"])

with eval_tab1:
    st.caption("Run evaluation for the currently selected strategy.")
    col_limit, _ = st.columns([1, 2])
    with col_limit:
        max_rows = st.number_input("Max rows", min_value=1, max_value=120, value=20, key="quick_batch_max_rows")

    if st.button("▶️ Run Batch Eval", use_container_width=True):
        client = get_active_client()
        if not client:
            st.error("No API client available. Please set your API key.")
        elif not DATASET_INDEX:
            st.error("Dataset not found. Make sure rpg_persona_dataset.jsonl is in the repo root.")
        else:
            lines = list(DATASET_INDEX.values())[:max_rows]
            progress = st.progress(0)
            per_item_rows = []

            for idx, ex in enumerate(lines, start=1):
                persona_data = ex.get("persona", {}) or {}
                if "functions" not in persona_data:
                    persona_data["functions"] = DEFAULT_FUNCTIONS

                user_utterance = ex.get("user_utterance", "")
                gold = ex.get("gold", {})
                gold_response = ex.get("gold_response", "")

                system_prompt = build_persona_prompt(
                    persona_data=persona_data,
                    user_message=user_utterance,
                    chat_history=[],
                    strategy_name=st.session_state.selected_strategy,
                    enable_action_first=st.session_state.enable_action_first,
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_utterance},
                ]

                model_name = get_active_model(model_options, selected_model_label)
                pred_raw = get_llm_response(st.session_state.api_provider, client, messages, model_name)

                try:
                    obj = json.loads(pred_raw) if isinstance(pred_raw, str) else pred_raw
                except Exception:
                    obj = pred_raw

                pred = normalize_pred_schema(obj)
                m = score_example(pred, gold, gold_response)
                m["id"] = ex.get("id", f"item_{idx}")
                m["strategy"] = st.session_state.selected_strategy
                m["persona"] = persona_data.get("name", "Unknown")
                m["utterance"] = user_utterance[:50] + "..." if len(user_utterance) > 50 else user_utterance
                per_item_rows.append(m)
                progress.progress(idx / len(lines))

            st.success(f"✅ Evaluated {len(lines)} items.")
            df = pd.DataFrame(per_item_rows)
            st.dataframe(df, use_container_width=True)

            summary = aggregate_rows(per_item_rows)
            st.subheader("Aggregate Metrics")
            c1, c2, c3 = st.columns(3)
            c1.metric("Fn Name Exact", f"{summary['fn_exact']:.3f}" if summary["fn_exact"] is not None else "-")
            c2.metric("Arg Exact", f"{summary['arg_exact']:.3f}" if summary["arg_exact"] is not None else "-")
            c3.metric("Over-Call", f"{summary['over_call']:.3f}" if summary["over_call"] is not None else "-")
            d1, d2 = st.columns(2)
            d1.metric("Under-Call", f"{summary['under_call']:.3f}" if summary["under_call"] is not None else "-")
            d2.metric("Single-Shot Violation", f"{summary['single_shot_violation']:.3f}" if summary["single_shot_violation"] is not None else "-")

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download CSV",
                data=csv,
                file_name=f"eval_{st.session_state.selected_strategy.replace(' ', '_').lower()}.csv",
                mime="text/csv",
            )

with eval_tab2:
    st.caption("Compare two strategies side by side on the same dataset rows.")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        strategy_a = st.selectbox("Strategy A", STRATEGIES, index=0, key="strat_a")
    with col_s2:
        strategy_b = st.selectbox("Strategy B", STRATEGIES, index=2, key="strat_b")

    max_rows_compare = st.number_input("Max rows", min_value=1, max_value=50, value=10, key="compare_max_rows")

    if st.button("▶️ Run Comparison", use_container_width=True):
        client = get_active_client()
        if not client:
            st.error("No API client available.")
        elif not DATASET_INDEX:
            st.error("Dataset not found.")
        elif strategy_a == strategy_b:
            st.warning("Please pick two different strategies.")
        else:
            lines = list(DATASET_INDEX.values())[:max_rows_compare]
            results = {strategy_a: [], strategy_b: []}
            model_name = get_active_model(model_options, selected_model_label)
            progress = st.progress(0)
            total_steps = len(lines) * 2

            for step, (strategy, rows_list) in enumerate(results.items()):
                for idx, ex in enumerate(lines, start=1):
                    persona_data = ex.get("persona", {}) or {}
                    if "functions" not in persona_data:
                        persona_data["functions"] = DEFAULT_FUNCTIONS

                    user_utterance = ex.get("user_utterance", "")
                    gold = ex.get("gold", {})
                    gold_response = ex.get("gold_response", "")

                    system_prompt = build_persona_prompt(
                        persona_data=persona_data,
                        user_message=user_utterance,
                        chat_history=[],
                        strategy_name=strategy,
                        enable_action_first=False,
                    )
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_utterance},
                    ]
                    pred_raw = get_llm_response(st.session_state.api_provider, client, messages, model_name)
                    try:
                        obj = json.loads(pred_raw) if isinstance(pred_raw, str) else pred_raw
                    except Exception:
                        obj = pred_raw

                    pred = normalize_pred_schema(obj)
                    m = score_example(pred, gold, gold_response)
                    m["strategy"] = strategy
                    m["utterance"] = user_utterance[:40] + "..." if len(user_utterance) > 40 else user_utterance
                    rows_list.append(m)
                    progress.progress((step * len(lines) + idx) / total_steps)

            summary_a = aggregate_rows(results[strategy_a])
            summary_b = aggregate_rows(results[strategy_b])

            st.subheader("📊 Side-by-Side Results")
            col_a, col_b = st.columns(2)
            for col, name, summary in [(col_a, strategy_a, summary_a), (col_b, strategy_b, summary_b)]:
                with col:
                    st.markdown(f"**{name}**")
                    st.metric("Fn Name Exact", f"{summary['fn_exact']:.3f}" if summary["fn_exact"] is not None else "-")
                    st.metric("Arg Exact", f"{summary['arg_exact']:.3f}" if summary["arg_exact"] is not None else "-")
                    st.metric("Over-Call", f"{summary['over_call']:.3f}" if summary["over_call"] is not None else "-")
                    st.metric("Under-Call", f"{summary['under_call']:.3f}" if summary["under_call"] is not None else "-")

            # Combined downloadable CSV
            df_all = pd.DataFrame(results[strategy_a] + results[strategy_b])
            csv = df_all.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Comparison CSV",
                data=csv,
                file_name="strategy_comparison.csv",
                mime="text/csv",
            )
