# CPDC 2025 – Persona-Grounded Dialogue Agent (Baseline)

## Overview
This repository contains a baseline implementation of a persona-grounded,
task-oriented dialogue agent developed for the Sony CPDC 2025 challenge.

The focus of this baseline was to validate the end-to-end dialogue pipeline
and ensure persona consistency before applying advanced optimizations.

## User Interface
This project includes a Streamlit-based user interface used for interacting
with the persona-grounded dialogue agent during development and testing.
The Streamlit app was used to visualize responses, test persona consistency,
and validate end-to-end system behavior.

## Pipeline
User Input →
Persona/System Prompt Injection →
Intent Routing →
LLM Response Generation →
Final Persona-Consistent Output

## Key Components
- Persona-grounded prompting
- Baseline intent handling
- Dataset-driven evaluation
- Scripted runners for experiments

## Project Structure
- `src/` – Agent logic and prompts
- `data/` – Persona and dialogue datasets
- `run.py` – Agent execution
- `run_dataset.py` – Dataset-based evaluation

## My Role
- Designed the baseline agent architecture
- Implemented persona prompting strategies
- Built evaluation scripts
- Validated the system pipeline

## Notes
This is a research/academic prototype created for evaluation purposes.
