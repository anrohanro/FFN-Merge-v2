# FFN Merge v2

FFN clustering and merging pipeline for GPT-2 models.

`v5.py` is the main file. It loads a Hugging Face GPT-2 model, evaluates it on WikiText, merges similar FFN layers, optionally uptrains, and writes run logs.

## Setup

```bash
pip install torch transformers datasets tqdm
```

For benchmark/testing files, also install:

```bash
pip install numpy scipy sacrebleu rouge-score
```

Use a CUDA GPU if available. The script can run on CPU, but it will be slow.

## Run

```bash
python v5.py
```

Run with custom options:

```bash
python v5.py --model_name gpt2 --target_clusters 11 --rank 8 --save 1
```

## Options

- `--model_name`: Hugging Face model name. Default: `gpt2`
- `--target_clusters`: final number of FFN clusters. Default: `11`
- `--rank`: LoRA rank used during adaptation. Default: `8`
- `--save`: set `1` to save final model artifacts, or `0` to only log results

## Outputs

- Run logs are saved in `logs/`
- Saved model artifacts are saved in `outputs/` when `--save 1` is used

## Files

- `v5.py`: main FFN merge pipeline
- `inference.py`: loads a saved model bundle, checks perplexity, and generates text
- `experiment.py`: runs multiple `v5.py` experiments with different cluster/rank settings
- `eval.py`: simple benchmark run for GPT-2
- `benchmark_datasets.py`: loads benchmark datasets
- `benchmark_eval.py`: benchmark metrics and confidence intervals
- `benchmark_runner.py`: connects Hugging Face models to the benchmark evaluator
- `test.py`: HellaSwag evaluation/testing script
- `v1.py`, `v4.py`: older versions of the merge pipeline
- `results.txt`: saved benchmark result notes

## Helper File Usage

Use `inference.py` to load a saved compressed model, compute perplexity, and generate text:

```bash
python inference.py --bundle_dir outputs/gpt2_clusters11_rank8 --prompt "Once upon a time"
```

Use `eval.py` to run a simple GPT-2 benchmark:

```bash
python eval.py
```

Use `experiment.py` to run multiple experiments from the grid inside the file:

```bash
python experiment.py
```

Use `test.py` to run HellaSwag evaluation:

```bash
python test.py
```

Before using `test.py`, update `BUNDLE_DIR` inside the file to your saved model path.

Use benchmark helper files by importing them:

```python
from benchmark_runner import run_model_benchmarks
from benchmark_datasets import build_data_loaders
from benchmark_eval import evaluate_model
```

- `benchmark_datasets.py`: dataset loading helpers
- `benchmark_eval.py`: metric calculation helpers
- `benchmark_runner.py`: `run_model_benchmarks(...)` wrapper for Hugging Face models

No usage commands are needed for `v1.py`, `v4.py`, or `v5.py` in this section.
