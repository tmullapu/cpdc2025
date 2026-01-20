import json
from src.agent import process

def main():
    with open("data/rpg_persona_dataset.jsonl") as f:
        sample = json.loads(next(f))
        
    result = process(sample)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
