from typing import Dict, Iterable, List, Optional


def _require_datasets():
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "'datasets' is required to load benchmark datasets. "
            "Install it with: pip install datasets"
        ) from exc
    return load_dataset


def _cap(samples: Iterable[dict], max_samples: Optional[int]) -> List[dict]:
    if max_samples is None:
        return list(samples)
    return list(samples)[:max_samples]


def load_squad(max_samples: Optional[int] = None) -> List[dict]:
    load_dataset = _require_datasets()
    dataset = load_dataset("squad", split="validation")

    samples = []
    for row in dataset:
        answers = [answer.strip() for answer in row["answers"]["text"] if answer.strip()]
        if not answers:
            continue
        samples.append(
            {
                "context": row["context"],
                "question": row["question"],
                "answers": answers,
            }
        )
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def load_wmt14_en_fr(max_samples: Optional[int] = None) -> List[dict]:
    load_dataset = _require_datasets()
    dataset = load_dataset("wmt14", "fr-en", split="test")

    samples = []
    for row in dataset:
        translation = row["translation"]
        source = translation["en"].strip()
        target = translation["fr"].strip()
        if not source or not target:
            continue
        samples.append({"source": source, "target": target})
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def load_wmt14_en_de(max_samples: Optional[int] = None) -> List[dict]:
    load_dataset = _require_datasets()
    dataset = load_dataset("wmt14", "de-en", split="test")

    samples = []
    for row in dataset:
        translation = row["translation"]
        source = translation["en"].strip()
        target = translation["de"].strip()
        if not source or not target:
            continue
        samples.append({"source": source, "target": target})
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def load_cnn_dailymail(max_samples: Optional[int] = None) -> List[dict]:
    load_dataset = _require_datasets()
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")

    samples = []
    for row in dataset:
        article = row["article"].strip()
        summary = row["highlights"].strip()
        if not article or not summary:
            continue
        samples.append({"article": article, "summary": summary})
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def load_coqa(max_samples: Optional[int] = None) -> List[dict]:
    load_dataset = _require_datasets()
    dataset = load_dataset("coqa", split="validation")

    samples = []
    for row in dataset:
        story = row["story"].strip()
        questions = row["questions"]
        answers = row["answers"]["input_text"]
        history = []

        for question, answer in zip(questions, answers):
            question = question.strip()
            answer = answer.strip()
            if not question or not answer:
                continue
            samples.append(
                {
                    "story": story,
                    "question": question,
                    "history": list(history),
                    "answers": [answer],
                }
            )
            history.append({"question": question, "answer": answer})
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    return samples


_LOADERS = {
    "squad": load_squad,
    "coqa": load_coqa,
    "wmt-14 en-fr": load_wmt14_en_fr,
    "wmt-14 en-de": load_wmt14_en_de,
    "cnn/dailymail": load_cnn_dailymail,
    "cnn/daily mail": load_cnn_dailymail,
}


def build_data_loaders(
    benchmarks: List[str],
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, List[dict]]:
    del seed

    data_loaders: Dict[str, List[dict]] = {}
    for benchmark in benchmarks:
        key = benchmark.strip().lower()
        if key not in _LOADERS:
            available = ", ".join(sorted(_LOADERS))
            raise ValueError(f"Unknown benchmark '{benchmark}'. Available: {available}")
        data_loaders[benchmark] = _LOADERS[key](max_samples=max_samples)
    return data_loaders
