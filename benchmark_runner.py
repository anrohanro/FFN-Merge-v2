from typing import Dict, List, Optional

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from benchmark_datasets import build_data_loaders
from benchmark_eval import evaluate_model


class HFTextGenerator:
    """Adapter for benchmark_eval.py's generate(prompt) -> str interface."""

    def __init__(
        self,
        model: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
        device: torch.device,
        max_input_length: int = 1024,
        max_new_tokens: int = 64,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_positions = getattr(self.model.config, "n_positions", None)
        if self.max_positions is None:
            self.max_positions = getattr(self.model.config, "max_position_embeddings", None)

    def generate(self, prompt: str) -> str:
        safe_input_length = self.max_input_length
        if self.max_positions is not None:
            # Keep room for generation so GPT-2 never exceeds its position limit.
            safe_input_length = min(
                self.max_input_length,
                max(1, self.max_positions - self.max_new_tokens),
            )

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=safe_input_length,
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}

        available_new_tokens = self.max_new_tokens
        if self.max_positions is not None:
            prompt_length = encoded["input_ids"].shape[1]
            available_new_tokens = max(1, self.max_positions - prompt_length)

        with torch.no_grad():
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=available_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_length = encoded["input_ids"].shape[1]
        generated_ids = output_ids[0][prompt_length:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def run_model_benchmarks(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    benchmarks: List[str],
    device: Optional[torch.device] = None,
    max_samples: Optional[int] = 100,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
    max_input_length: int = 1024,
    max_new_tokens: int = 64,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    if device is None:
        device = next(model.parameters()).device

    data_loaders = build_data_loaders(
        benchmarks=benchmarks,
        max_samples=max_samples,
        seed=seed,
    )

    text_model = HFTextGenerator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
    )

    return evaluate_model(
        model=text_model,
        benchmarks=benchmarks,
        data_loaders=data_loaders,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        max_samples=max_samples,
        seed=seed,
        verbose=True,
    )
