import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel

from v5 import assert_shared_ffn_ties, wrap_mlp_with_lora, ClusterRegistry, Cluster


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL_DIR = Path("/data/user7/c2/outputs/gpt2_clusters6_rank4")


def load_json_if_exists(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def describe_bundle(bundle_dir: Path) -> dict:
    model_dir = bundle_dir / "model"
    tokenizer_dir = bundle_dir / "tokenizer"
    return {
        "bundle_dir": str(bundle_dir),
        "model_dir": str(model_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "has_pytorch_weights": (model_dir / "pytorch_model.bin").exists(),
        "has_safe_weights": (model_dir / "model.safetensors").exists(),
        "has_tokenizer": tokenizer_dir.exists(),
        "has_config_json": (bundle_dir / "config.json").exists(),
        "has_metadata_json": (bundle_dir / "metadata.json").exists(),
        "has_clusters_json": (bundle_dir / "clusters.json").exists(),
    }


def resolve_tokenizer_source(bundle_dir: Path, config_data: dict) -> str:
    tokenizer_dir = bundle_dir / "tokenizer"
    model_dir = bundle_dir / "model"

    if tokenizer_dir.exists():
        return str(tokenizer_dir)
    if (model_dir / "tokenizer_config.json").exists() or (model_dir / "vocab.json").exists():
        return str(model_dir)

    model_name = config_data.get("model_name")
    if model_name:
        return model_name
    return str(model_dir)


def load_bundle(bundle_dir: Path):
    model_dir = bundle_dir / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    config_data = load_json_if_exists(bundle_dir / "config.json")
    metadata_data = load_json_if_exists(bundle_dir / "metadata.json")
    cluster_data = load_json_if_exists(bundle_dir / "clusters.json")
    tokenizer_source = resolve_tokenizer_source(bundle_dir, config_data)
    bundle_info = describe_bundle(bundle_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=Path(tokenizer_source).exists(),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = int(1e30)

    if bundle_info["has_pytorch_weights"]:
        print("Loading shared-weight export from PyTorch checkpoint: model/pytorch_model.bin")
    elif bundle_info["has_safe_weights"]:
        print("Loading safetensors export from model/model.safetensors")
    else:
        raise FileNotFoundError(
            f"No supported model weights found in {model_dir}. "
            "Expected `pytorch_model.bin` or `model.safetensors`."
        )

    model_name = config_data.get("model_name")
    rank = config_data.get("rank")
    if not model_name:
        raise ValueError(f"Missing `model_name` in {bundle_dir / 'config.json'}")
    if rank is None:
        raise ValueError(f"Missing `rank` in {bundle_dir / 'config.json'}")

    model = rebuild_compressed_model(
        bundle_dir=bundle_dir,
        model_name=model_name,
        rank=int(rank),
        cluster_data=cluster_data,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, config_data, metadata_data, cluster_data, bundle_info


def apply_cluster_tying(model, cluster_data: dict) -> ClusterRegistry:
    num_layers = len(model.transformer.h)
    registry = ClusterRegistry(num_layers)
    registry.clusters = {}
    registry._next_id = 0

    raw_clusters = cluster_data.get("clusters", {})
    if not raw_clusters:
        raise ValueError("clusters.json is missing a `clusters` mapping.")

    max_cluster_id = -1
    for cluster_id_str, members in raw_clusters.items():
        cluster_id = int(cluster_id_str)
        member_list = [int(m) for m in members]
        if not member_list:
            raise ValueError(f"Cluster {cluster_id} has no members.")

        representative_layer_idx = member_list[0]
        rep_block = model.transformer.h[representative_layer_idx]
        rep_fc_conv = rep_block.mlp.c_fc.conv
        rep_proj_conv = rep_block.mlp.c_proj.conv

        for member_idx in member_list:
            block = model.transformer.h[member_idx]
            block.mlp.c_fc.conv = rep_fc_conv
            block.mlp.c_proj.conv = rep_proj_conv

        registry.clusters[cluster_id] = Cluster(
            cluster_id=cluster_id,
            shared_layer_idx=representative_layer_idx,
            members=member_list,
        )
        max_cluster_id = max(max_cluster_id, cluster_id)

    registry._next_id = max_cluster_id + 1
    return registry


def rebuild_compressed_model(bundle_dir: Path, model_name: str, rank: int, cluster_data: dict):
    model = GPT2LMHeadModel.from_pretrained(model_name).to(DEVICE)

    for layer in model.transformer.h:
        wrap_mlp_with_lora(layer, rank=rank)

    registry = apply_cluster_tying(model, cluster_data)

    state_dict_path = bundle_dir / "model" / "pytorch_model.bin"
    if not state_dict_path.exists():
        raise FileNotFoundError(f"Missing PyTorch checkpoint: {state_dict_path}")

    state_dict = torch.load(state_dict_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    assert_shared_ffn_ties(model, registry)
    return model


def compute_perplexity(model, tokenizer, dataset_name: str, split: str, max_length: int = 1024, stride: int = 512) -> float:
    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split)
    texts = [t for t in dataset["text"] if t.strip()]
    full_text = "\n\n".join(texts[:200])

    enc = tokenizer(full_text, return_tensors="pt")
    input_ids = enc.input_ids.to(DEVICE)

    seq_len = input_ids.size(1)
    total_nll = 0.0
    total_tokens = 0
    prev_end = 0

    for begin in tqdm(range(0, seq_len, stride), desc=f"PPL {split}", leave=False):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end

        chunk = input_ids[:, begin:end]
        labels = chunk.clone()
        labels[:, :-target_len] = -100

        with torch.no_grad():
            outputs = model(chunk, labels=labels)
            loss = outputs.loss.item()

        total_nll += loss * target_len
        total_tokens += target_len
        prev_end = end
        if end == seq_len:
            break

    return torch.exp(torch.tensor(total_nll / total_tokens)).item()


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a saved compressed model bundle, compute perplexity, and generate text.")
    parser.add_argument(
        "--bundle_dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Path to the exported model bundle directory.",
    )
    parser.add_argument(
        "--prompt",
        default="Once upon a time",
        help="Prompt to use for generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=80,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling value.",
    )
    parser.add_argument(
        "--skip_test_ppl",
        action="store_true",
        help="Skip test-set perplexity and only compute validation perplexity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)

    model, tokenizer, config_data, metadata_data, cluster_data, bundle_info = load_bundle(bundle_dir)

    print(f"Loaded bundle: {bundle_dir}")
    print("Bundle contents:")
    print(json.dumps(bundle_info, indent=2))
    if config_data:
        print("Saved config:")
        print(json.dumps(config_data, indent=2))
    if metadata_data:
        print("Saved metadata:")
        print(json.dumps(metadata_data, indent=2))
    if cluster_data:
        cluster_count = len(cluster_data.get("clusters", {}))
        print(f"Saved clusters: {cluster_count}")

    validation_ppl = compute_perplexity(model, tokenizer, "wikitext", "validation")
    print(f"Validation perplexity: {validation_ppl:.2f}")

    if not args.skip_test_ppl:
        test_ppl = compute_perplexity(model, tokenizer, "wikitext", "test")
        print(f"Test perplexity: {test_ppl:.2f}")

    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print("\nGenerated text:")
    print(generated)


if __name__ == "__main__":
    main()
