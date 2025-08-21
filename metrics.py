import re
from typing import List, Tuple
from langchain.schema import Document
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon

_WORD = re.compile(r"\w+")

def _f1(a: str, b: str) -> float:
    a_tokens, b_tokens = _WORD.findall(a.lower()), _WORD.findall(b.lower())
    if not a_tokens or not b_tokens:
        return float(a_tokens == b_tokens)
    common = len(set(a_tokens) & set(b_tokens))
    if common == 0:
        return 0.0
    precision = common / len(a_tokens)
    recall    = common / len(b_tokens)
    return 2 * precision * recall / (precision + recall)

def per_example_metrics(
    preds: List[str], golds: List[str]
) -> Tuple[List[int], List[float]]:
    em  = [int(p.strip().lower() == g.strip().lower()) for p, g in zip(preds, golds)]
    f1s = [_f1(p, g) for p, g in zip(preds, golds)]
    return em, f1s

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[!?\.,;:'\"\-]", "", text)
    text = re.sub(r"\b(a|an|the)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def check_recall(chunks: List[Document], answer: str) -> int:
    norm_ans = normalize(answer)
    for chunk in chunks:
        if norm_ans in normalize(chunk.page_content):
            return 1
    return 0


def compute_em_f1(preds: List[str], golds: List[str]) -> (float, float):
    total = len(preds)
    em_sum = 0.0
    f1_sum = 0.0
    for pred, gold in zip(preds, golds):
        n_pred = normalize(pred)
        n_gold = normalize(gold)
        # EM
        em = 1.0 if n_pred == n_gold else 0.0
        em_sum += em
        # F1
        pred_tokens = n_pred.split()
        gold_tokens = n_gold.split()
        common = 0
        for token in set(pred_tokens + gold_tokens):
            common += min(pred_tokens.count(token), gold_tokens.count(token))
        if pred_tokens and gold_tokens:
            precision = common / len(pred_tokens)
            recall = common / len(gold_tokens)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            f1 = 1.0 if not pred_tokens and not gold_tokens else 0.0
        f1_sum += f1
    # convert to percentages
    return em_sum / total * 100, f1_sum / total * 100


def report_results(method: str, em: float, f1: float, recall: float, chunk_time: float):
    print(f"\n=== Results for {method.upper()} Chunking ===")
    print(f"Exact Match (EM):    {em:.2f}%")
    print(f"F1 Score:            {f1:.2f}%")
    print(f"Recall@K:           {recall:.2f}%")
    print(f"Avg Chunk Time:      {chunk_time:.3f} ms/doc")
