"""
benchmark_eval.py
─────────────────
Evaluate a GPT-style model on any subset of:
  • SQuAD        – Exact-Match (EM) + F1
  • CoQA         – Exact-Match (EM) + F1
  • WMT-14 En→Fr – BLEU
  • WMT-14 En→De – BLEU
  • CNN/DailyMail – ROUGE-1, ROUGE-2, ROUGE-L

For every metric, a 95 % confidence interval is computed via
non-parametric bootstrap (10 000 resamples by default).

Usage
─────
from benchmark_eval import evaluate_model

results = evaluate_model(
    model        = my_model,          # any object with a .generate(prompt) -> str method
    benchmarks   = ["SQuAD", "WMT-14 En-Fr", "CNN/DailyMail"],
    data_loaders = {                  # {benchmark_name: iterable of dicts}
        "SQuAD":         squad_data,
        "WMT-14 En-Fr":  wmt_fr_data,
        "CNN/DailyMail": cnn_data,
    },
    n_bootstrap  = 10_000,            # resamples for CI  (default 10 000)
    ci_level     = 0.95,              # confidence level  (default 0.95)
    max_samples  = None,              # cap samples per benchmark (None = all)
    seed         = 42,
)
# results is a dict: benchmark -> {metric -> {score, ci_lower, ci_upper}}
"""

from __future__ import annotations

import re
import string
import collections
import numpy as np
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# ── optional heavy deps (imported lazily so the module loads without them) ─────
def _require(pkg: str):
    import importlib, sys
    if pkg not in sys.modules:
        try:
            return importlib.import_module(pkg)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"'{pkg}' is required for this benchmark. "
                f"Install it with:  pip install {pkg}"
            )
    return sys.modules[pkg]


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_answer(s: str) -> str:
    """Lower-case, remove punctuation/articles/extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


# ══════════════════════════════════════════════════════════════════════════════
#  PER-SAMPLE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def _exact_match(pred: str, gold: str) -> float:
    return float(_normalize_answer(pred) == _normalize_answer(gold))


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens  = _normalize_answer(pred).split()
    gold_tokens  = _normalize_answer(gold).split()
    common       = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same     = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _bleu_sentence(hypothesis: str, reference: str) -> float:
    """Sentence-level BLEU-4 using sacrebleu."""
    sacrebleu = _require("sacrebleu")
    result = sacrebleu.sentence_bleu(hypothesis, [reference])
    return result.score / 100.0          # sacrebleu returns 0-100


def _rouge_scores(pred: str, reference: str) -> Dict[str, float]:
    """Returns {'rouge1': f, 'rouge2': f, 'rougeL': f}."""
    rouge = _require("rouge_score.rouge_scorer")
    scorer = rouge.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, pred)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK RUNNERS
#  Every runner receives (model, data_loader, max_samples)
#  and returns a dict  metric_name -> List[float]  (one value per sample)
# ══════════════════════════════════════════════════════════════════════════════

def _run_squad(model, loader, max_samples) -> Dict[str, List[float]]:
    """
    Each item in loader must be a dict with keys:
        'context'  : str
        'question' : str
        'answers'  : List[str]   # one or more acceptable answers
    """
    em_scores, f1_scores = [], []

    for i, sample in enumerate(_cap(loader, max_samples)):
        prompt    = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"
        pred      = model.generate(prompt)
        gold_list = sample["answers"]

        # take best score over all acceptable answers
        em_scores.append(max(_exact_match(pred, g) for g in gold_list))
        f1_scores.append(max(_token_f1(pred, g)    for g in gold_list))

    return {"EM": em_scores, "F1": f1_scores}


def _run_coqa(model, loader, max_samples) -> Dict[str, List[float]]:
    """
    CoQA is conversational; each item should have:
        'story'    : str
        'question' : str
        'history'  : List[dict]  # [{'question': ..., 'answer': ...}, ...]
        'answers'  : List[str]
    """
    em_scores, f1_scores = [], []

    for sample in _cap(loader, max_samples):
        history_str = "\n".join(
            f"Q: {h['question']}\nA: {h['answer']}"
            for h in sample.get("history", [])
        )
        prompt = (
            f"Story: {sample['story']}\n"
            f"{history_str}\n"
            f"Q: {sample['question']}\nA:"
        )
        pred      = model.generate(prompt)
        gold_list = sample["answers"]

        em_scores.append(max(_exact_match(pred, g) for g in gold_list))
        f1_scores.append(max(_token_f1(pred, g)    for g in gold_list))

    return {"EM": em_scores, "F1": f1_scores}


def _run_wmt(model, loader, max_samples, src_lang: str, tgt_lang: str) -> Dict[str, List[float]]:
    """
    Each item:
        'source' : str   (sentence in src_lang)
        'target' : str   (reference in tgt_lang)
    """
    bleu_scores = []

    for sample in _cap(loader, max_samples):
        prompt = f"Translate from {src_lang} to {tgt_lang}: {sample['source']}"
        pred   = model.generate(prompt)
        bleu_scores.append(_bleu_sentence(pred, sample["target"]))

    return {"BLEU": bleu_scores}


def _run_cnn_dm(model, loader, max_samples) -> Dict[str, List[float]]:
    """
    Each item:
        'article' : str
        'summary' : str   (reference summary)
    """
    r1, r2, rL = [], [], []

    for sample in _cap(loader, max_samples):
        prompt  = f"Summarize the following article:\n{sample['article']}\nSummary:"
        pred    = model.generate(prompt)
        scores  = _rouge_scores(pred, sample["summary"])
        r1.append(scores["rouge1"])
        r2.append(scores["rouge2"])
        rL.append(scores["rougeL"])

    return {"ROUGE-1": r1, "ROUGE-2": r2, "ROUGE-L": rL}


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIDENCE INTERVAL (bootstrap)
# ══════════════════════════════════════════════════════════════════════════════

def _bootstrap_ci(
    scores    : List[float],
    n_resample: int   = 10_000,
    ci_level  : float = 0.95,
    rng       : np.random.Generator = None,
) -> Tuple[float, float, float]:
    """
    Returns (mean, ci_lower, ci_upper) via percentile bootstrap.
    """
    arr  = np.array(scores, dtype=float)
    mean = float(arr.mean())

    if rng is None:
        rng = np.random.default_rng()

    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_resample)
    ])

    alpha = 1.0 - ci_level
    lo    = float(np.percentile(boot_means, 100 * alpha / 2))
    hi    = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean, lo, hi


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _cap(iterable: Iterable, n: Optional[int]):
    for i, item in enumerate(iterable):
        if n is not None and i >= n:
            break
        yield item


# benchmark name → runner function
_BENCHMARK_REGISTRY: Dict[str, Callable] = {
    "squad"          : lambda m, l, k: _run_squad(m, l, k),
    "coqa"           : lambda m, l, k: _run_coqa(m, l, k),
    "wmt-14 en-fr"   : lambda m, l, k: _run_wmt(m, l, k, "English", "French"),
    "wmt-14 en-de"   : lambda m, l, k: _run_wmt(m, l, k, "English", "German"),
    "cnn/dailymail"  : lambda m, l, k: _run_cnn_dm(m, l, k),
    "cnn/daily mail" : lambda m, l, k: _run_cnn_dm(m, l, k),   # alias
}


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model       : Any,
    benchmarks  : List[str],
    data_loaders: Dict[str, Iterable],
    n_bootstrap : int   = 10_000,
    ci_level    : float = 0.95,
    max_samples : Optional[int] = None,
    seed        : int   = 42,
    verbose     : bool  = True,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate `model` on each benchmark in `benchmarks`.

    Parameters
    ----------
    model        : object
        Must expose a `.generate(prompt: str) -> str` method.
    benchmarks   : list[str]
        Names from {"SQuAD", "CoQA", "WMT-14 En-Fr",
                    "WMT-14 En-De", "CNN/DailyMail"}.
    data_loaders : dict[str, Iterable[dict]]
        Keys must match entries in `benchmarks`.
    n_bootstrap  : int
        Number of bootstrap resamples for CI (default 10 000).
    ci_level     : float
        Confidence level, e.g. 0.95 for 95 % CI (default 0.95).
    max_samples  : int | None
        Limit samples per benchmark (useful for quick tests).
    seed         : int
        Random seed for reproducibility.
    verbose      : bool
        Print a formatted summary table.

    Returns
    -------
    dict  benchmark_name -> metric_name -> {"score", "ci_lower", "ci_upper"}
    """
    rng     = np.random.default_rng(seed)
    results = {}

    for bm in benchmarks:
        bm_key = bm.strip().lower()

        if bm_key not in _BENCHMARK_REGISTRY:
            raise ValueError(
                f"Unknown benchmark '{bm}'. "
                f"Available: {list(_BENCHMARK_REGISTRY.keys())}"
            )
        if bm not in data_loaders:
            raise KeyError(
                f"No data loader provided for benchmark '{bm}'. "
                f"Please add it to data_loaders."
            )

        if verbose:
            print(f"\n▶  Running {bm} …")

        runner     = _BENCHMARK_REGISTRY[bm_key]
        raw_scores = runner(model, data_loaders[bm], max_samples)

        bm_result = {}
        for metric, scores in raw_scores.items():
            mean, lo, hi = _bootstrap_ci(scores, n_bootstrap, ci_level, rng)
            bm_result[metric] = {
                "score"   : round(mean, 4),
                "ci_lower": round(lo,   4),
                "ci_upper": round(hi,   4),
                "n_samples": len(scores),
            }

        results[bm] = bm_result

    if verbose:
        _print_table(results, ci_level)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PRETTY PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def _print_table(results: dict, ci_level: float):
    pct = int(ci_level * 100)
    header = f"{'Benchmark':<22} {'Metric':<12} {'Score':>8}  {pct}% CI"
    print("\n" + "═" * len(header))
    print(header)
    print("═" * len(header))

    for bm, metrics in results.items():
        first = True
        for metric, vals in metrics.items():
            bm_col = bm if first else ""
            first  = False
            score  = vals["score"]
            lo, hi = vals["ci_lower"], vals["ci_upper"]
            print(f"{bm_col:<22} {metric:<12} {score:>8.4f}  [{lo:.4f}, {hi:.4f}]")
        print("─" * len(header))


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK SMOKE-TEST  (run: python benchmark_eval.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import random

    class _DummyModel:
        """Returns a random short string — purely for API smoke-testing."""
        def generate(self, prompt: str) -> str:
            words = ["the", "cat", "sat", "on", "the", "mat"]
            return " ".join(random.choices(words, k=random.randint(3, 8)))

    _squad_data = [
        {
            "context" : "The Eiffel Tower is located in Paris.",
            "question": "Where is the Eiffel Tower?",
            "answers" : ["Paris", "in Paris"],
        }
    ] * 30

    _coqa_data = [
        {
            "story"   : "Alice went to the market.",
            "question": "Where did Alice go?",
            "history" : [],
            "answers" : ["the market", "market"],
        }
    ] * 30

    _wmt_fr_data = [
        {"source": "Hello, how are you?", "target": "Bonjour, comment allez-vous?"}
    ] * 20

    _wmt_de_data = [
        {"source": "Hello, how are you?", "target": "Hallo, wie geht es Ihnen?"}
    ] * 20

    _cnn_data = [
        {
            "article": "Scientists discovered a new species of frog in the Amazon rainforest.",
            "summary": "A new frog species was found in the Amazon.",
        }
    ] * 30

    evaluate_model(
        model        = _DummyModel(),
        benchmarks   = [
            "SQuAD",
            "CoQA",
            "WMT-14 En-Fr",
            "WMT-14 En-De",
            "CNN/DailyMail",
        ],
        data_loaders = {
            "SQuAD"        : _squad_data,
            "CoQA"         : _coqa_data,
            "WMT-14 En-Fr" : _wmt_fr_data,
            "WMT-14 En-De" : _wmt_de_data,
            "CNN/DailyMail": _cnn_data,
        },
        n_bootstrap  = 1_000,   # fewer resamples for the quick test
        max_samples  = 20,
        verbose      = True,
    )
    
    


    
