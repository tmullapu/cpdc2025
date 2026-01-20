import json
from src.agent import process

def load_jsonl(path):
    return [json.loads(l) for l in open(path)]

def run_on_dataset(input_path, output_path):
    data = load_jsonl(input_path)
    results = []
    for sample in data:
        result = process(sample)
        results.append({
            "id": sample["id"],
            "model_output": result
        })
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    run_on_dataset("data/persona_dataset.jsonl", "outputs/predictions.jsonl")
