import json

def load_jsonl(path):
    return [json.loads(l) for l in open(path)]

def evaluate(gold_path, pred_path):
    gold = load_jsonl(gold_path)
    pred = load_jsonl(pred_path)

    fn_correct = 0
    arg_correct = 0
    arg_total = 0

    for g, p in zip(gold, pred):
        g_fn = g.get("expected_function")
        p_fn = p["model_output"].get("function_call", {}).get("name")
        if g_fn == p_fn:
            fn_correct += 1

            g_args = g.get("expected_arguments", {})
            p_args = p["model_output"].get("function_call", {}).get("arguments", {})

            for k in g_args:
                arg_total += 1
                if k in p_args and p_args[k] == g_args[k]:
                    arg_correct += 1
        elif g_fn is None and p_fn is None:
            fn_correct += 1

    total = len(gold)
    fn_acc = fn_correct / total
    arg_acc = arg_correct / arg_total if arg_total else 1.0
    final = (fn_acc + arg_acc) / 2

    print("Function Accuracy:", fn_acc)
    print("Argument Accuracy:", arg_acc)
    print("Final Score:", final)

if __name__ == "__main__":
    evaluate("data/persona_dataset.jsonl", "outputs/predictions.jsonl")
