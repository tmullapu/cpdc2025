# evaluation.py

from typing import Dict, Any, Optional, List

# Add imports for text quality metrics
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("Warning: bert-score not installed. BERTScore metrics will be unavailable.")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not installed. RougeL metrics will be unavailable.")

METRIC_KEYS = [
    "fn_exact",
    "arg_exact",
    "over_call",
    "under_call",
    "single_shot_violation",
    "rouge_l_f1",
    "rouge_l_precision",
    "rouge_l_recall",
    "bertscore_f1",
    "bertscore_precision",
    "bertscore_recall",
]

def compute_text_quality_metrics(predicted_response: str, gold_response: str, device='cpu') -> Dict[str, float]:
    """
    Compute RougeL and BERTScore between predicted and gold responses.
    
    Args:
        predicted_response: The model's generated text response
        gold_response: The ideal/expected response from dataset
    
    Returns:
        Dictionary with rouge_l_f1, bertscore_f1, bertscore_precision, bertscore_recall
    """
    metrics = {}
    
    # Handle empty responses
    if not predicted_response or not gold_response:
        return {
            "rouge_l_f1": 0.0,
            "rouge_l_precision": 0.0,
            "rouge_l_recall": 0.0,
            "bertscore_f1": 0.0,
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0
        }
    
    # 1. Computing RougeL
    if ROUGE_AVAILABLE:
        try:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(gold_response, predicted_response)
            metrics["rouge_l_f1"] = rouge_scores['rougeL'].fmeasure
            metrics["rouge_l_precision"] = rouge_scores['rougeL'].precision
            metrics["rouge_l_recall"] = rouge_scores['rougeL'].recall
        except Exception as e:
            print(f"Error computing RougeL: {e}")
            metrics["rouge_l_f1"] = 0.0
            metrics["rouge_l_precision"] = 0.0
            metrics["rouge_l_recall"] = 0.0
    else:
        metrics["rouge_l_f1"] = None
        metrics["rouge_l_precision"] = None
        metrics["rouge_l_recall"] = None
    
    # 2. Compute BERTScore
    if BERTSCORE_AVAILABLE:
        try:
            # BERTScore expects lists of strings
            P, R, F1 = bert_score(
                [predicted_response], 
                [gold_response], 
                lang='en', 
                verbose=False,
                device=device  # Use parameter instead of hardcoded
            )
            metrics["bertscore_f1"] = F1.item()
            metrics["bertscore_precision"] = P.item()
            metrics["bertscore_recall"] = R.item()
        except Exception as e:
            print(f"Error computing BERTScore: {e}")
            metrics["bertscore_f1"] = 0.0
            metrics["bertscore_precision"] = 0.0
            metrics["bertscore_recall"] = 0.0
    else:
        metrics["bertscore_f1"] = None
        metrics["bertscore_precision"] = None
        metrics["bertscore_recall"] = None
    
    return metrics

def score_example(pred: Dict[str, Any], gold: Dict[str, Any], gold_response: str = "", device='cpu') -> Dict[str, Optional[float]]:
    """
    pred = {
      "response": str,
      "function_call": {"name": str, "arguments": dict} | None,
      "num_calls": int,                 # optional; default 1 if function_call present else 0
      "text_before_call": bool          # optional; default False
    }
    gold = {
      "needs_call": bool,
      "function_name": str (if needs_call),
      "arguments": dict (if needs_call),
      "one_call_only": bool
    }
    gold_response: str  # The ideal response text from dataset
    device: str  # Device for BERTScore computation ('cpu' or 'cuda')
    """
    needs = gold.get("needs_call", False)
    call = pred.get("function_call")
    num_calls = pred.get("num_calls", 1 if call else 0)
    text_before_call = pred.get("text_before_call", False)
    predicted_response = pred.get("response", "")

    # exact matches (only when a call is required)
    fn_exact = None
    arg_exact = None
    if needs:
        fn_exact = int(bool(call) and call.get("name") == gold.get("function_name"))
        arg_exact = int(bool(call) and call.get("arguments", {}) == gold.get("arguments", {}))

    # over/under
    over_call = int((not needs) and bool(call))
    under_call = int(needs and (not call))

    # policy checks
    single_shot_violation = int(gold.get("one_call_only", True) and num_calls > 1)

    # NEW: Compute text quality metrics (RougeL + BERTScore)
    text_metrics = {}
    if gold_response:  # Only compute if gold_response is provided
        text_metrics = compute_text_quality_metrics(predicted_response, gold_response, device=device)
    else:
        # Default values if no gold_response
        text_metrics = {
            "rouge_l_f1": None,
            "rouge_l_precision": None,
            "rouge_l_recall": None,
            "bertscore_f1": None,
            "bertscore_precision": None,
            "bertscore_recall": None
        }

    return {
        # Function call metrics (existing)
        "fn_exact": fn_exact,
        "arg_exact": arg_exact,
        "over_call": over_call,
        "under_call": under_call,
        "single_shot_violation": single_shot_violation,
        # Text quality metrics (NEW)
        **text_metrics
    }

def aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """
    rows: list of per-example metric dicts from score_example(...) PLUS 'id' per item.
    Returns mean of each numeric metric (ignores None).
    """
    keys = [
        "fn_exact",
        "arg_exact",
        "over_call",
        "under_call",
        "single_shot_violation",
        "rouge_l_f1",
        "rouge_l_precision",
        "rouge_l_recall",
        "bertscore_f1",
        "bertscore_precision",
        "bertscore_recall",
    ]
    out: Dict[str, Optional[float]] = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = (sum(vals) / len(vals)) if vals else None
    out["n_items"] = len(rows)
    return out
