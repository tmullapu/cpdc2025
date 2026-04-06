# CPDC 2025 — Persona-Grounded Multi-Agent Dialogue System

🎮 **Live Demo:** [cpdc2025.streamlit.app](https://cpdc2025.streamlit.app)

A research implementation of a persona-grounded, task-oriented dialogue agent inspired by the Sony CPDC 2025 challenge. The system enables RPG-style NPC characters to hold contextually aware conversations, execute game-relevant function calls, adapt their communication style based on the player's persona, and evaluate response quality in real time.

---

## What This Solves

Most game NPCs are fully scripted — they say the same lines regardless of who the player is or what they need. This system solves three problems:

1. **Persona consistency** — NPCs stay in character across any conversation topic using structured persona-grounded prompting
2. **Grounded information retrieval** — Instead of hallucinating game data, NPCs retrieve real lore via a RAG pipeline built on semantic similarity search
3. **Player-adaptive dialogue** — The same NPC produces measurably different responses based on the player's persona (e.g. Aggressive Warrior vs Red Witch), demonstrated and evaluated live

---

## System Architecture

```
User Input
    ↓
Player Persona Injection  ←─── persona_loader.py (5 player personas)
    ↓
NPC Persona + Strategy Prompt  ←─── 4 prompting strategies
    ↓
LLM (Llama 3.3 70B via GROQ / GPT-4o via OpenAI)
    ↓
Function Call Decision
    ↓ (if function needed)
RAG Retrieval  ←─── rag_engine.py + lore_database.json
    ↓
Grounded Response Generation
    ↓
Real-Time Evaluation  ←─── evaluation.py (ROUGE-L + BERTScore + fn_exact)
    ↓
Final Persona-Consistent Output
```

---

## Key Features

### Multi-Agent Persona System
- **9 NPC personas** loaded from a gold-labeled dataset (Aether, Myst, Shadow, Lyra, Ronan, Seraphina, Kael, Vex, Elandra)
- **5 player personas** that dynamically shift NPC tone and style: Aggressive Warrior, Cautious Scholar, Chaotic Rogue, Empathetic Healer, Red Witch
- Player persona context is injected into every system prompt, causing measurable response adaptation

### RAG Pipeline
- Custom narrative lore database (`lore_database.json`) — 42 rich lore entries across 4 games: Elden Ring, Skyrim, Divinity: Original Sin 2, Baldur's Gate 3
- Semantic retrieval using `sentence-transformers` (`all-MiniLM-L6-v2`) with cosine similarity search
- No external vector database — runs fully in-memory with numpy
- Graceful fallback to mock data when no relevant lore is found (relevance threshold: 0.2)

### Function Calling
- 6 game functions: `get_character_stats`, `get_quest_info`, `get_item_info`, `get_location_info`, `get_skill_tree`, `get_recipe_info`
- Structured JSON output format enforced via prompt schema
- Schema normalization handles common LLM output variants

### Prompting Strategies (4)
| Strategy | Description |
|---|---|
| Zero-Shot | Direct persona instruction with output schema |
| Few-Shot | In-character examples with constrained output |
| Chain of Thought | Step-by-step reasoning before function decision |
| Persona Sandwich | Strict action-first rules with single-shot enforcement |

### Real-Time Evaluation
Every response is scored against a gold-labeled dataset of 120 examples:

| Metric | Description |
|---|---|
| `fn_exact` | Did the model call the correct function? |
| `arg_exact` | Did it pass the correct arguments? |
| `over_call` | Did it call a function when it shouldn't have? |
| `under_call` | Did it fail to call when it should have? |
| `rouge_l_f1` | Lexical overlap with gold response |
| `bertscore_f1` | Semantic similarity with gold response |

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Llama 3.3 70B (GROQ) / GPT-4o (OpenAI) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Similarity Search | NumPy cosine similarity |
| UI | Streamlit |
| Evaluation | rouge-score, bert-score |
| Deployment | Streamlit Cloud |

---

## Project Structure

```
cpdc2025/
├── app.py                    # Main Streamlit app — chat UI + evaluation dashboard
├── rag_engine.py             # RAG pipeline — embeddings + semantic retrieval
├── lore_database.json        # 42 narrative lore entries across 4 games
├── evaluation.py             # Scoring functions — fn_exact, ROUGE-L, BERTScore
├── persona_loader.py         # NPC + player persona definitions and tone injection
├── requirements.txt
├── rpg_persona_dataset.jsonl # 120 gold-labeled evaluation examples
├── run.py                    # CLI runner
├── run_dataset.py            # Dataset-based batch runner
└── src/
    ├── __init__.py
    ├── agent.py              # Core agent logic
    ├── evaluator.py          # Evaluation utilities
    └── function_executor.py  # RAG-powered function implementations
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- A free GROQ API key from [console.groq.com](https://console.groq.com) (or OpenAI key)

### Installation

```bash
git clone https://github.com/tmullapu/cpdc2025.git
cd cpdc2025
pip install -r requirements.txt
```

### Running Locally

```bash
# Create a .env file with your API key
echo "GROQ_API_KEY=your_key_here" > .env

# Run the app
streamlit run app.py
```

### How to Use
1. Select an **NPC character** from the sidebar (e.g. Aether, Myst, Shadow)
2. Select **your player persona** (e.g. Red Witch, Aggressive Warrior) — watch the NPC adapt
3. Choose a **prompting strategy**
4. Chat — try asking about quests, items, character stats, or locations
5. Scroll down to **Batch Evaluation** to run strategy comparisons

---

## Research Findings

Running Zero-Shot strategy across 20 dataset examples (Llama 3.3 70B):

| Metric | Score |
|---|---|
| Function Name Exact | ~0.93 |
| Argument Exact | ~0.79 |
| Over-Call Rate | ~0.30 |
| BERTScore F1 | ~0.88 |

Key finding: The model correctly identifies which function to call 93% of the time but over-calls on chitchat turns at a 30% rate — a known limitation of Zero-Shot prompting that Chain of Thought reduces significantly.

---

## Acknowledgements

- Inspired by the [Sony CPDC 2025](https://www.sonycpdc.com/) dialogue challenge
- Lore database is original synthetic content created for this project
- Built as an independent research project extending coursework at George Mason University

---

## Author

**Tejaharshita Mullapudi**
M.S. Information Systems, George Mason University
[GitHub](https://github.com/tmullapu) · [LinkedIn](https://www.linkedin.com/in/mullapuditejaharshita)
