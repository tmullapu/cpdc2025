import json
from src.agent import process

def main():
    with open("data/sample_input.json") as f:
        sample = json.load(f)
    result = process(sample)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
