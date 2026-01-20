import json
from openai import OpenAI
from src.function_executor import execute

client = OpenAI()

def build_prompt(sample):
    persona = sample["persona"]
    worldview = sample["worldview"]
    functions = sample["functions"]
    history = sample["history"]
    user_msg = history[-1]["text"] if history else ""
    
    # Handle worldview - it might be a string or object
    if isinstance(worldview, str):
        worldview_display = worldview
    elif isinstance(worldview, dict) and "type" in worldview:
        worldview_display = worldview["type"]
    else:
        worldview_display = json.dumps(worldview)

    prompt = f"""
<SYSTEM>
You are a Task-Oriented Dialogue Agent.
Your job is to respond to the Human and call functions when appropriate.
Follow persona, worldview, and output rules exactly.
</SYSTEM>

<PERSONA>
Name: {persona['name']}
Role: {persona['role']}
Traits: {', '.join(persona['traits'])}
</PERSONA>

<WORLDVIEW>
{worldview_display}
</WORLDVIEW>

<FUNCTIONS>
{json.dumps(functions)}
</FUNCTIONS>

<CONTEXT>
{"\\n".join(f"{msg['speaker']}: {msg['text']}" for msg in history)}
</CONTEXT>

<GOAL>
1) Understand what the user wants.
2) Decide whether the request requires a function.
3) If function is needed → output FUNCTION JSON ONLY.
4) If normal reply → output RESPONSE JSON ONLY.
5) Maintain persona + worldview.
6) Do not explain or add commentary outside JSON.
</GOAL>

<OUTPUT_FORMAT>
If calling a function:
{{
  "function_call": {{
    "name": "<function_name>",
    "arguments": {{ ... }}
  }}
}}

If not calling a function:
{{
  "response": "<text>"
}}
</OUTPUT_FORMAT>

<USER>
{user_msg}
</USER>
"""
    return prompt


def process(sample):
    """
    Send the formatted prompt to the OpenAI model, get response, 
    and execute function if needed.
    """
    prompt = build_prompt(sample)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  #  smaller, cheaper, fast model
            messages=[
                {"role": "system", "content": "You are a task-oriented agent."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content.strip()

        # Try to parse JSON-like outputs
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON", "raw_output": content}

        # If model requested a function call → simulate execution
        if "function_call" in parsed:
            result = execute(parsed["function_call"])
            return {
                "response": f"Executed: {parsed['function_call']['name']} → {result}",
                "function_call": parsed["function_call"]
            }

        return parsed

    except Exception as e:
        return {"error": str(e)}
