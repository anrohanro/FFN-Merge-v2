"""
FFN Clustering and Merging Pipeline for GPT-2
==============================================
Iteratively merges similar FFN layers by:
  1. Computing pairwise base distances between cluster representatives
  2. Selecting the closest pair
  3. Soft-aligning candidate cluster bases via combined loss (LM + align + anchor)
  4. Collapsing to one representative base if ΔL is within threshold
  5. Repeating until target cluster count is reached
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm.auto import tqdm
import random
import copy
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# # =============================================================================
# # CONFIG FINAL RUN
# # =============================================================================
# DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_NAME     = "gpt2"
# TRAIN_DATASET_NAME = "wikitext-2 train"
# VALIDATION_DATASET_NAME = "wikitext-2 validation"
# TEST_DATASET_NAME = "wikitext-2 test"
# UPTRAIN_DATASET_NAME = "openwebtext train[:10%]"

# BATCH_SIZE     = 8        # tiny batch
# SEQ_LEN        = 64       # shorter sequences
# CALIB_BATCHES  = 3        # minimal for distance

# ALIGN_STEPS    = 500       # just to check training loop
# RECOVERY_STEPS = 200       # quick stabilization

# LR_ALIGN       = 3e-5     # slightly higher → faster movement
# LR_RECOVERY    = 1e-5
# UPTRAIN_STEPS  = 1000       # smoke-run final global uptraining
# LR_UPTRAIN     = 1e-5
# KD_ALPHA       = 0.5
# UPTRAIN_LOG_INTERVAL = 100

# LAMBDA_MAX     = 1.0      # weaker constraint (faster convergence)
# MU             = 0.5      # lighter anchor

# LORA_RANK      = 1        # smaller → faster & less memory
# WARMUP_FRAC    = 0.2

# TARGET_CLUSTERS = 5      # only 1–2 merges (from 12 → 10)

# THRESH_EXCELLENT  = 0.2
# THRESH_ACCEPTABLE = 0.5
# THRESH_BAD        = 1.0   # very lenient → avoid rejection loops
# ALIGN_LOG_INTERVAL = 100
# RECOVERY_LOG_INTERVAL = 50

# # NUM_LAYERS = 12



# =============================================================================
# CONFIG SMOKE RUN
# =============================================================================
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME     = "gpt2"
TRAIN_DATASET_NAME = "wikitext-2 train"
VALIDATION_DATASET_NAME = "wikitext-2 validation"
TEST_DATASET_NAME = "wikitext-2 test"
UPTRAIN_DATASET_NAME = "openwebtext train[:1%]"

BATCH_SIZE     = 2        # tiny batch
SEQ_LEN        = 32       # shorter sequences
CALIB_BATCHES  = 3        # minimal for distance

ALIGN_STEPS    = 50       # just to check training loop
RECOVERY_STEPS = 20       # quick stabilization

LR_ALIGN       = 5e-5     # slightly higher → faster movement
LR_RECOVERY    = 2e-5
UPTRAIN_STEPS  = 5       # smoke-run final global uptraining
LR_UPTRAIN     = 1e-5
KD_ALPHA       = 0.5
UPTRAIN_LOG_INTERVAL = 10

LAMBDA_MAX     = 1.0      # weaker constraint (faster convergence)
MU             = 0.5      # lighter anchor

LORA_RANK      = 1        # smaller → faster & less memory
WARMUP_FRAC    = 0.2

TARGET_CLUSTERS = 5      # only 1–2 merges (from 12 → 10)

THRESH_EXCELLENT  = 0.2
THRESH_ACCEPTABLE = 0.5
THRESH_BAD        = 1.0   # very lenient → avoid rejection loops
ALIGN_LOG_INTERVAL = 200
RECOVERY_LOG_INTERVAL = 100

# NUM_LAYERS = 12


# =============================================================================
# CONFIG
# =============================================================================
# DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE     = 8
# SEQ_LEN        = 64
# CALIB_BATCHES  = 20       # batches used for distance matrix computation
# ALIGN_STEPS    = 2000     # steps for alignment training per merge round
# RECOVERY_STEPS = 300      # LM-only fine-tuning after merge
# LR_ALIGN       = 3e-5
# LR_RECOVERY    = 1e-5
# UPTRAIN_STEPS  = 2000
# LR_UPTRAIN     = 1e-5
# KD_ALPHA       = 0.5
# UPTRAIN_LOG_INTERVAL = 100
# LAMBDA_MAX     = 3.0      # weight on (align + anchor)
# MU             = 1.0      # weight on anchor inside the structural term
# LORA_RANK      = 8
# WARMUP_FRAC    = 0.3      # fraction of ALIGN_STEPS used for lambda warmup

# TARGET_CLUSTERS = 8       # stop when this many clusters remain

# # Merge acceptance thresholds
# THRESH_EXCELLENT  = 0.1
# THRESH_ACCEPTABLE = 0.3
# THRESH_BAD        = 0.5

# NUM_LAYERS = 12           # GPT-2 small


#=============================================================================
# UTILS
#=============================================================================

def count_params_total(model: nn.Module) -> int:
    return sum(p.numel() for _, p in model.named_parameters(remove_duplicate=False))


def count_params_unique(model: nn.Module) -> int:
    seen = set()
    total = 0
    for _, p in model.named_parameters(remove_duplicate=False):
        ptr = id(p)
        if ptr in seen:
            continue
        seen.add(ptr)
        total += p.numel()
    return total


def compression_ratio(unique_before: int, unique_after: int) -> float:
    return (1 - (unique_after / unique_before)) * 100


def safe_percent_delta(current: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return ((current - baseline) / baseline) * 100


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


@dataclass(frozen=True)
class PipelineConfig:
    model_name: str = MODEL_NAME
    target_clusters: int = TARGET_CLUSTERS
    rank: int = LORA_RANK
    save: bool = False


def build_output_dir_name(config: PipelineConfig) -> str:
    safe_model_name = config.model_name.replace("/", "_")
    return f"{safe_model_name}_clusters{config.target_clusters}_rank{config.rank}"


def export_final_artifacts(
    model,
    tokenizer,
    config: PipelineConfig,
    registry: "ClusterRegistry",
    log_path: Path,
    compression_percent: float,
    final_ppl: float,
) -> Path:
    output_dir = Path("outputs") / build_output_dir_name(config)
    model_dir = output_dir / "model"
    tokenizer_dir = output_dir / "tokenizer"

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)

    config_payload = {
        "model_name": config.model_name,
        "target_clusters": config.target_clusters,
        "rank": config.rank,
        "num_clusters_final": registry.num_clusters(),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    cluster_payload = {
        "clusters": {
            str(cluster.cluster_id): cluster.members
            for cluster in sorted(registry.clusters.values(), key=lambda c: c.cluster_id)
        }
    }
    (output_dir / "clusters.json").write_text(
        json.dumps(cluster_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    metadata_payload = {
        "log_file": str(log_path),
        "compression_percent": compression_percent,
        "final_ppl": final_ppl,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    return output_dir


class RunLogger:
    def __init__(self, config: PipelineConfig, device: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{timestamp}"
        self.run_dir = Path("logs") / self.run_id
        self.tables_dir = self.run_dir / "tables"
        self.plots_dir = self.run_dir / "plots"
        self.json_path = self.run_dir / "run.json"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.data: Dict[str, Any] = {
            "run_id": self.run_id,
            "status": "running",
            "last_completed_phase": "initializing",
            "experiment": {
                "model": config.model_name,
                "device": device,
                "timestamp": timestamp,
                "runtime_inputs": asdict(config),
                "datasets": {
                    "train": TRAIN_DATASET_NAME,
                    "validation": VALIDATION_DATASET_NAME,
                    "test": TEST_DATASET_NAME,
                    "uptraining": UPTRAIN_DATASET_NAME,
                },
                "hyperparameters": {
                    "BATCH_SIZE": BATCH_SIZE,
                    "SEQ_LEN": SEQ_LEN,
                    "CALIB_BATCHES": CALIB_BATCHES,
                    "ALIGN_STEPS": ALIGN_STEPS,
                    "RECOVERY_STEPS": RECOVERY_STEPS,
                    "LR_ALIGN": LR_ALIGN,
                    "LR_RECOVERY": LR_RECOVERY,
                    "UPTRAIN_STEPS": UPTRAIN_STEPS,
                    "LR_UPTRAIN": LR_UPTRAIN,
                    "KD_ALPHA": KD_ALPHA,
                    "LAMBDA_MAX": LAMBDA_MAX,
                    "MU": MU,
                    "TARGET_CLUSTERS": config.target_clusters,
                    "LORA_RANK": config.rank,
                    "WARMUP_FRAC": WARMUP_FRAC,
                    "THRESH_EXCELLENT": THRESH_EXCELLENT,
                    "THRESH_ACCEPTABLE": THRESH_ACCEPTABLE,
                    "THRESH_BAD": THRESH_BAD,
                },
                "logging": {
                    "alignment_log_interval": ALIGN_LOG_INTERVAL,
                    "recovery_log_interval": RECOVERY_LOG_INTERVAL,
                    "uptraining_log_interval": UPTRAIN_LOG_INTERVAL,
                },
            },
            "baseline": {},
            "merge_history": [],
            "phase_metrics": {
                "alignment": [],
                "recovery": [],
                "uptraining": [],
            },
            "final": {},
        }

    def mark_phase(self, phase_name: str) -> None:
        self.data["last_completed_phase"] = phase_name

    def set_status(self, status: str, phase_name: str, error: Optional[str] = None) -> None:
        self.data["status"] = status
        self.data["last_completed_phase"] = phase_name
        if error is None:
            self.data.pop("error", None)
        else:
            self.data["error"] = error

    def set_baseline(self, baseline: Dict[str, Any]) -> None:
        self.data["baseline"] = sanitize_for_json(baseline)

    def append_merge(self, merge_record: Dict[str, Any]) -> None:
        self.data["merge_history"].append(sanitize_for_json(merge_record))

    def append_phase_metric(self, phase_name: str, record: Dict[str, Any]) -> None:
        self.data["phase_metrics"][phase_name].append(sanitize_for_json(record))

    def set_final(self, final_section: Dict[str, Any]) -> None:
        self.data["final"] = sanitize_for_json(final_section)

    def write_json(self) -> None:
        tmp_path = self.json_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(sanitize_for_json(self.data), indent=2), encoding="utf-8")
        os.replace(tmp_path, self.json_path)

    def write_checkpoint(self, phase_name: str, status: str = "running") -> None:
        self.set_status(status, phase_name)
        self.write_json()

    def finalize(self, status: str = "completed", phase_name: str = "complete") -> None:
        error = self.data.get("error") if status == "failed" else None
        self.set_status(status, phase_name, error=error)
        self.write_json()
        self.write_tables()
        self.write_plots()

    def _write_markdown_table(self, path: Path, headers: List[str], rows: List[List[Any]]) -> None:
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            formatted = []
            for value in row:
                if isinstance(value, float):
                    formatted.append(f"{value:.4f}")
                else:
                    formatted.append(str(value))
            lines.append("| " + " | ".join(formatted) + " |")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_tables(self) -> None:
        baseline = self.data.get("baseline", {})
        final = self.data.get("final", {})
        final_metrics = final.get("metrics", {})
        final_compression = final.get("compression", {})

        summary_rows = [
            ["Baseline", baseline.get("L_orig", ""), baseline.get("PPL_validation", ""), baseline.get("PPL_test", ""), baseline.get("params_unique", "")],
            ["Final", final_metrics.get("L_final", ""), final_metrics.get("PPL_validation", ""), final_metrics.get("PPL_test", ""), final_compression.get("params_after", "")],
        ]
        self._write_markdown_table(
            self.tables_dir / "baseline_final_summary.md",
            ["Stage", "Loss", "Validation PPL", "Test PPL", "Unique Params"],
            summary_rows,
        )

        merge_rows = []
        for entry in self.data.get("merge_history", []):
            merge_rows.append([
                entry.get("round", ""),
                entry.get("accepted", ""),
                entry.get("clusters_before", ""),
                entry.get("merge_pair", ""),
                entry.get("distance", ""),
                entry.get("L_post_align", ""),
                entry.get("PPL_post_align", ""),
                entry.get("L_final", ""),
                entry.get("compression_after", ""),
            ])
        self._write_markdown_table(
            self.tables_dir / "merge_history.md",
            ["Round", "Accepted", "Clusters Before", "Merge Pair", "Distance", "L Post Align", "PPL Post Align", "L Final", "Compression %"],
            merge_rows,
        )

        compression_rows = [[
            final_compression.get("params_before", ""),
            final_compression.get("params_after", ""),
            final_compression.get("compression_percent", ""),
        ]]
        self._write_markdown_table(
            self.tables_dir / "compression_summary.md",
            ["Params Before", "Params After", "Compression %"],
            compression_rows,
        )

    def _write_svg_line_plot(
        self,
        path: Path,
        title: str,
        x_label: str,
        y_label: str,
        series: List[Tuple[str, List[Tuple[float, float]]]],
    ) -> None:
        width = 900
        height = 520
        left = 80
        right = 40
        top = 60
        bottom = 70
        plot_width = width - left - right
        plot_height = height - top - bottom
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

        all_points = [point for _, points in series for point in points]
        if not all_points:
            svg = (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
                f'<text x="{width/2}" y="{height/2}" text-anchor="middle" font-size="24">{title}: no data</text>'
                "</svg>"
            )
            path.write_text(svg, encoding="utf-8")
            return

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if min_x == max_x:
            max_x = min_x + 1.0
        if min_y == max_y:
            max_y = min_y + 1.0

        def sx(x: float) -> float:
            return left + ((x - min_x) / (max_x - min_x)) * plot_width

        def sy(y: float) -> float:
            return top + plot_height - ((y - min_y) / (max_y - min_y)) * plot_height

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<text x="{width/2}" y="30" text-anchor="middle" font-size="24" font-family="Arial">{title}</text>',
            f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="#333" stroke-width="2"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#333" stroke-width="2"/>',
            f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-size="16" font-family="Arial">{x_label}</text>',
            f'<text x="20" y="{height/2}" text-anchor="middle" font-size="16" font-family="Arial" transform="rotate(-90 20 {height/2})">{y_label}</text>',
        ]

        for idx in range(5):
            frac = idx / 4
            gx = left + frac * plot_width
            gy = top + frac * plot_height
            x_value = min_x + frac * (max_x - min_x)
            y_value = max_y - frac * (max_y - min_y)
            parts.append(f'<line x1="{gx}" y1="{top}" x2="{gx}" y2="{top+plot_height}" stroke="#ddd" stroke-width="1"/>')
            parts.append(f'<line x1="{left}" y1="{gy}" x2="{left+plot_width}" y2="{gy}" stroke="#ddd" stroke-width="1"/>')
            parts.append(f'<text x="{gx}" y="{top+plot_height+20}" text-anchor="middle" font-size="12" font-family="Arial">{x_value:.1f}</text>')
            parts.append(f'<text x="{left-10}" y="{gy+4}" text-anchor="end" font-size="12" font-family="Arial">{y_value:.2f}</text>')

        legend_y = top
        for idx, (label, points) in enumerate(series):
            if not points:
                continue
            color = colors[idx % len(colors)]
            polyline = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline}"/>')
            for x, y in points:
                parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{color}"/>')
            legend_x = left + plot_width + 10
            parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="12" height="12" fill="{color}"/>')
            parts.append(f'<text x="{legend_x + 18}" y="{legend_y + 10}" font-size="12" font-family="Arial">{label}</text>')
            legend_y += 20

        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")

    def write_plots(self) -> None:
        align_series: Dict[int, List[Tuple[float, float]]] = {}
        for record in self.data["phase_metrics"].get("alignment", []):
            align_series.setdefault(record["round"], []).append((record["step"], record["total_loss"]))
        self._write_svg_line_plot(
            self.plots_dir / "alignment_loss_by_round.svg",
            "Alignment Loss by Round",
            "Step",
            "Total Loss",
            [(f"Round {round_id}", points) for round_id, points in sorted(align_series.items())],
        )

        recovery_series: Dict[int, List[Tuple[float, float]]] = {}
        for record in self.data["phase_metrics"].get("recovery", []):
            recovery_series.setdefault(record["round"], []).append((record["step"], record["train_loss"]))
        self._write_svg_line_plot(
            self.plots_dir / "recovery_loss_by_round.svg",
            "Recovery Loss by Round",
            "Step",
            "Train Loss",
            [(f"Round {round_id}", points) for round_id, points in sorted(recovery_series.items())],
        )

        uptraining = self.data["phase_metrics"].get("uptraining", [])
        self._write_svg_line_plot(
            self.plots_dir / "uptraining_losses.svg",
            "Uptraining Loss Curves",
            "Step",
            "Loss",
            [
                ("LM Loss", [(r["step"], r["lm_loss"]) for r in uptraining]),
                ("KD Loss", [(r["step"], r["kd_loss"]) for r in uptraining]),
                ("Total Loss", [(r["step"], r["total_loss"]) for r in uptraining]),
            ],
        )
        self._write_svg_line_plot(
            self.plots_dir / "uptraining_validation_ppl.svg",
            "Uptraining Validation PPL",
            "Step",
            "Validation PPL",
            [("Validation PPL", [(r["step"], r["val_ppl"]) for r in uptraining if "val_ppl" in r])],
        )

        accepted_merges = [m for m in self.data.get("merge_history", []) if m.get("accepted")]
        self._write_svg_line_plot(
            self.plots_dir / "compression_vs_merge.svg",
            "Compression vs Accepted Merge",
            "Accepted Merge Index",
            "Compression %",
            [("Compression", [(idx + 1, m["compression_after"]) for idx, m in enumerate(accepted_merges) if "compression_after" in m])],
        )
        self._write_svg_line_plot(
            self.plots_dir / "validation_ppl_vs_merge.svg",
            "Validation PPL vs Accepted Merge",
            "Accepted Merge Index",
            "Validation PPL",
            [("Validation PPL", [(idx + 1, m["PPL_final"]) for idx, m in enumerate(accepted_merges) if "PPL_final" in m])],
        )

# =============================================================================
# LORA
# =============================================================================
class LoRAConv1D(nn.Module):
    """Wraps a GPT-2 Conv1D with a low-rank delta: output = base(x) + x @ (A @ B)"""

    def __init__(self, conv: nn.Module, rank: int = 8):
        super().__init__()
        self.conv = conv
        in_dim  = conv.weight.shape[0]
        out_dim = conv.weight.shape[1]
        self.A  = nn.Parameter(torch.randn(in_dim, rank, device=conv.weight.device) * 0.01)
        self.B  = nn.Parameter(torch.zeros(rank, out_dim, device=conv.weight.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + torch.matmul(x, self.A @ self.B)

    def base_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using only the base conv, bypassing LoRA."""
        return self.conv(x)


def wrap_mlp_with_lora(block: nn.Module, rank: int) -> None:
    """Replace c_fc and c_proj in an MLP block with LoRA-wrapped versions."""
    if not isinstance(block.mlp.c_fc, LoRAConv1D):
        block.mlp.c_fc   = LoRAConv1D(block.mlp.c_fc,   rank)
    if not isinstance(block.mlp.c_proj, LoRAConv1D):
        block.mlp.c_proj = LoRAConv1D(block.mlp.c_proj, rank)

# =============================================================================
# MLP FORWARD HELPERS
# =============================================================================
def full_mlp_forward(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Full MLP forward (base + LoRA)."""
    fc   = block.mlp.c_fc(x)
    act  = block.mlp.act(fc)
    proj = block.mlp.c_proj(act)
    return proj


def base_mlp_forward(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Base-only MLP forward (no LoRA delta)."""
    fc   = block.mlp.c_fc.base_forward(x)
    act  = block.mlp.act(fc)
    proj = block.mlp.c_proj.base_forward(act)
    return proj


# =============================================================================
# CLUSTER REGISTRY
# =============================================================================
@dataclass
class Cluster:
    """
    Represents one cluster of GPT-2 FFN layers sharing a common base.

    shared_layer_idx : index of the layer whose Conv1D weights ARE the shared base.
                       All other members point their conv to the same nn.Parameter.
    members          : list of layer indices belonging to this cluster.
    forbidden_pairs  : other cluster IDs that have been tried and rejected.
    """
    cluster_id      : int
    shared_layer_idx: int
    members         : List[int]
    forbidden_pairs : List[int] = field(default_factory=list)


class ClusterRegistry:
    def __init__(self, num_layers: int):
        # Start: every layer is its own singleton cluster
        self.clusters: Dict[int, Cluster] = {}
        self._next_id = 0
        for i in range(num_layers):
            cid = self._next_id
            self.clusters[cid] = Cluster(
                cluster_id       = cid,
                shared_layer_idx = i,
                members          = [i],
            )
            self._next_id += 1

    def num_clusters(self) -> int:
        return len(self.clusters)

    def get_cluster_of_layer(self, layer_idx: int) -> Optional[Cluster]:
        for c in self.clusters.values():
            if layer_idx in c.members:
                return c
        return None

    def merge(self, cid_a: int, cid_b: int, new_shared_layer_idx: int) -> int:
        """Merge two clusters into one. Returns new cluster id."""
        ca = self.clusters[cid_a]
        cb = self.clusters[cid_b]
        new_members = ca.members + cb.members
        nid = self._next_id
        self.clusters[nid] = Cluster(
            cluster_id       = nid,
            shared_layer_idx = new_shared_layer_idx,
            members          = new_members,
        )
        del self.clusters[cid_a]
        del self.clusters[cid_b]
        self._next_id += 1
        return nid

    def forbid_pair(self, cid_a: int, cid_b: int) -> None:
        if cid_a in self.clusters:
            self.clusters[cid_a].forbidden_pairs.append(cid_b)
        if cid_b in self.clusters:
            self.clusters[cid_b].forbidden_pairs.append(cid_a)

    def summary(self) -> str:
        lines = []
        for cid, c in sorted(self.clusters.items()):
            lines.append(f"  Cluster {cid}: layers={c.members}, base=layer{c.shared_layer_idx}")
        return "\n".join(lines)


# =============================================================================
# DATASET / BATCHING
# =============================================================================
def build_dataset():
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return ds

def build_test_dataset():
    return load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

def build_uptraining_dataset():
    return load_dataset("openwebtext", split="train[:1%]")


def get_batch(dataset, tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    valid = []
    while len(valid) < BATCH_SIZE:
        t = random.choice(dataset["text"])
        if t and len(t.strip()) > 10:
            valid.append(t.strip())
    enc = tokenizer(
        valid,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=SEQ_LEN,
    )
    return enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)


# =============================================================================
# EVALUATION
# =============================================================================

def compute_perplexity(model, dataset, tokenizer, max_length=1024, stride=512):
    was_training = model.training
    model.eval()

    # Use a FIXED evaluation set (important)
    texts = [t for t in dataset["text"] if t.strip()]
    full_text = "\n\n".join(texts[:200])  # limit for speed but deterministic

    enc = tokenizer(full_text, return_tensors="pt")
    input_ids = enc.input_ids.to(DEVICE)

    seq_len = input_ids.size(1)

    total_nll = 0.0
    total_tokens = 0
    prev_end = 0

    for begin in tqdm(
        range(0, seq_len, stride),
        desc="Perplexity",
        leave=False,
    ):
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

    ppl = torch.exp(torch.tensor(total_nll / total_tokens)).item()
    if was_training:
        model.train()
    return ppl

def evaluate(model, dataset, tokenizer, num_batches: int = 30) -> float:
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in tqdm(
            range(num_batches),
            desc="Evaluate",
            leave=False,
        ):
            x, m = get_batch(dataset, tokenizer)
            loss = model(x, attention_mask=m, labels=x).loss
            losses.append(loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)

def build_eval_dataset():
    return load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")


def build_teacher_model(model_name: str) -> GPT2LMHeadModel:
    teacher = GPT2LMHeadModel.from_pretrained(model_name).to(DEVICE).eval()
    teacher.config.pad_token_id = teacher.config.eos_token_id
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def assert_shared_ffn_ties(model, registry: ClusterRegistry) -> None:
    """
    Ensure every merged cluster still shares the exact same Conv1D base modules.
    This protects the compression invariant during later optimization phases.
    """
    layers = model.transformer.h

    for cid, cluster in registry.clusters.items():
        rep_idx = cluster.shared_layer_idx
        rep_block = layers[rep_idx]
        ref_fc = rep_block.mlp.c_fc.conv
        ref_proj = rep_block.mlp.c_proj.conv

        for member_idx in cluster.members:
            block = layers[member_idx]
            if block.mlp.c_fc.conv is not ref_fc:
                raise RuntimeError(
                    f"Shared FFN tie broken for cluster {cid}: "
                    f"layer {member_idx} c_fc is not tied to representative layer {rep_idx}."
                )
            if block.mlp.c_proj.conv is not ref_proj:
                raise RuntimeError(
                    f"Shared FFN tie broken for cluster {cid}: "
                    f"layer {member_idx} c_proj is not tied to representative layer {rep_idx}."
                )


def kd_kl_student_teacher(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    KL(p_student || p_teacher) over non-padded tokens.
    """
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student_probs = student_log_probs.exp()

    token_kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    mask = attention_mask.to(token_kl.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (token_kl * mask).sum() / denom
# =============================================================================
# PHASE 1 — DISTANCE MATRIX
# =============================================================================

def cache_activations(
    model, dataset, tokenizer, num_batches: int = CALIB_BATCHES
) -> Dict[int, List[torch.Tensor]]:
    """
    Cache LN2 output activations for every layer across num_batches.
    Returns dict: layer_idx -> list of (B, T, C) tensors (on CPU to save memory).
    """
    model.eval()
    layers = model.transformer.h
    num_layers = len(layers)    
    cache: Dict[int, List[torch.Tensor]] = {i: [] for i in range(num_layers)}
    hooks  = []

    layers = model.transformer.h

    def make_hook(idx):
        def hook(module, inp, out):
            # inp[0] is the hidden state before LN2; we want LN2 output
            # We hook the MLP's input which is LN2(hidden_state)
            cache[idx].append(inp[0].detach().cpu())
        return hook

    for i, layer in enumerate(layers):
        h = layer.mlp.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        for _ in tqdm(
            range(num_batches),
            desc="Cache activations",
            leave=False,
        ):
            x, m = get_batch(dataset, tokenizer)
            model(x, attention_mask=m)

    for h in hooks:
        h.remove()

    model.train()
    return cache


def compute_cluster_distance(
    block_a: nn.Module,
    block_b: nn.Module,
    h_samples_a: List[torch.Tensor],
    h_samples_b: List[torch.Tensor],
) -> float:
    """
    D(A, B) = E_h || f_A^base(h) - f_B^base(h) ||²
    Averaged over activations from both clusters' member layers.
    """
    all_samples = h_samples_a + h_samples_b
    total = 0.0
    count = 0
    with torch.no_grad():
        for h_cpu in all_samples:
            h = h_cpu.to(DEVICE)
            out_a = base_mlp_forward(block_a, h)
            out_b = base_mlp_forward(block_b, h)
            total += ((out_a - out_b) ** 2).mean().item()
            count += 1
    return total / max(count, 1)


def build_distance_matrix(
    model,
    registry: ClusterRegistry,
    act_cache: Dict[int, List[torch.Tensor]],
) -> Dict[Tuple[int, int], float]:
    """
    Build pairwise distance matrix over all current clusters.
    Returns dict: (cid_a, cid_b) -> distance  (lower triangle only, a < b)
    """
    layers   = model.transformer.h
    cids     = sorted(registry.clusters.keys())
    dist_mat = {}

    for i_idx, cid_a in enumerate(cids):
        for cid_b in cids[i_idx + 1:]:
            ca = registry.clusters[cid_a]
            cb = registry.clusters[cid_b]

            block_a    = layers[ca.shared_layer_idx]
            block_b    = layers[cb.shared_layer_idx]

            # Collect activation samples from all members of each cluster
            h_a = []
            for m in ca.members:
                h_a.extend(act_cache.get(m, []))
            h_b = []
            for m in cb.members:
                h_b.extend(act_cache.get(m, []))

            d = compute_cluster_distance(block_a, block_b, h_a, h_b)
            dist_mat[(cid_a, cid_b)] = d

    return dist_mat


def pick_merge_candidate(
    dist_mat: Dict[Tuple[int, int], float],
    registry: ClusterRegistry,
) -> Optional[Tuple[int, int, float]]:
    """
    Return (cid_a, cid_b, distance) for the pair with smallest distance,
    excluding forbidden pairs. Returns None if no valid pair exists.
    """
    best     = None
    best_d   = float("inf")

    for (cid_a, cid_b), d in dist_mat.items():
        ca = registry.clusters.get(cid_a)
        cb = registry.clusters.get(cid_b)
        if ca is None or cb is None:
            continue
        if cid_b in ca.forbidden_pairs or cid_a in cb.forbidden_pairs:
            continue
        if d < best_d:
            best_d = d
            best   = (cid_a, cid_b, d)

    return best


# =============================================================================
# PHASE 2 — SOFT ALIGNMENT TRAINING
# =============================================================================
def alignment_training(
    model,
    dataset,
    tokenizer,
    registry: ClusterRegistry,
    cid_a: int,
    cid_b: int,
    frozen_originals: Dict[int, nn.Module],
    rep_a_layer_idx: int,
    rep_b_layer_idx: int,
    round_num: int,
    logger: Optional[RunLogger] = None,
    steps: int = ALIGN_STEPS,
) -> None:
    """
    Train two representative bases + all member LoRAs with:
        L = L_LM + λ*L_align + μ*L_anchor

    L_align  : distance between representative base outputs, averaged over
               activations from all members in both candidate clusters.
    L_anchor : full function (base+LoRA) vs frozen original for every member.
    """
    layers      = model.transformer.h
    all_members = (
        registry.clusters[cid_a].members + registry.clusters[cid_b].members
    )

    rep_a_block = layers[rep_a_layer_idx]
    rep_b_block = layers[rep_b_layer_idx]

    # ---- Collect trainable parameters ----
    params = []
    seen_param_ids = set()

    def add_params(new_params) -> None:
        for p in new_params:
            if id(p) not in seen_param_ids:
                params.append(p)
                seen_param_ids.add(id(p))

    add_params(rep_a_block.mlp.c_fc.conv.parameters())
    add_params(rep_a_block.mlp.c_proj.conv.parameters())
    add_params(rep_b_block.mlp.c_fc.conv.parameters())
    add_params(rep_b_block.mlp.c_proj.conv.parameters())

    for m in all_members:
        blk = layers[m]
        add_params([blk.mlp.c_fc.A, blk.mlp.c_fc.B,
                    blk.mlp.c_proj.A, blk.mlp.c_proj.B])

    optimizer = torch.optim.AdamW(params, lr=LR_ALIGN)
    warmup_steps = int(WARMUP_FRAC * steps)

    for step in tqdm(
        range(steps),
        desc=f"Align round {round_num}",
        leave=False,
    ):
        x, mask = get_batch(dataset, tokenizer)
        optimizer.zero_grad()

        outputs = model(
            x,
            attention_mask=mask,
            labels=x,
            output_hidden_states=True,
        )
        loss_lm = outputs.loss

        loss_align  = torch.tensor(0.0, device=DEVICE)
        loss_anchor = torch.tensor(0.0, device=DEVICE)

        for m in all_members:
            h_raw = outputs.hidden_states[m]          # before layer m
            h_ln  = layers[m].ln_2(h_raw)             # LN2 output = MLP input

            rep_a_out = base_mlp_forward(rep_a_block, h_ln)
            rep_b_out = base_mlp_forward(rep_b_block, h_ln)
            loss_align = loss_align + ((rep_a_out - rep_b_out) ** 2).mean()

            with torch.no_grad():
                orig_out = full_mlp_forward(frozen_originals[m], h_ln.detach())

            full_out = full_mlp_forward(layers[m], h_ln)
            loss_anchor = loss_anchor + ((full_out - orig_out.detach()) ** 2).mean()

        loss_align  = loss_align  / len(all_members)
        loss_anchor = loss_anchor / len(all_members)

        # Lambda warmup schedule
        lam = LAMBDA_MAX * min(1.0, step / max(warmup_steps, 1))

        loss = loss_lm + lam * loss_align + MU * loss_anchor
        loss.backward()
        optimizer.step()

        should_log = step % ALIGN_LOG_INTERVAL == 0 or step == steps - 1
        if should_log:
            if logger is not None:
                logger.append_phase_metric("alignment", {
                    "round": round_num,
                    "step": step,
                    "lm_loss": loss_lm.item(),
                    "align_loss": loss_align.item(),
                    "anchor_loss": loss_anchor.item(),
                    "total_loss": loss.item(),
                    "lambda": lam,
                })
            lora_norms = []
            for m in all_members:
                n = (layers[m].mlp.c_fc.A.norm().item() +
                     layers[m].mlp.c_fc.B.norm().item())
                lora_norms.append(round(n, 4))
            rep_a_norm = (
                rep_a_block.mlp.c_fc.conv.weight.norm().item() +
                rep_a_block.mlp.c_proj.conv.weight.norm().item()
            )
            rep_b_norm = (
                rep_b_block.mlp.c_fc.conv.weight.norm().item() +
                rep_b_block.mlp.c_proj.conv.weight.norm().item()
            )
            print(
                f"  [align {step:4d}] "
                f"LM={loss_lm.item():.4f}  "
                f"Align={loss_align.item():.4f}  "
                f"Anchor={loss_anchor.item():.4f}  "
                f"Collapse={loss_align.item():.4f}  "
                f"RepNorms=({rep_a_norm:.4f}, {rep_b_norm:.4f})  "
                f"LoRA_norms={lora_norms}"
            )


def collapse_pair_to_representative(
    model,
    registry: ClusterRegistry,
    cid_a: int,
    cid_b: int,
    rank: int,
) -> Tuple[int, int]:
    """
    Collapse two soft-aligned clusters into one shared base by choosing an
    existing representative base and repointing every member to it.

    Returns:
        representative_cluster_id, representative_layer_idx
    """
    layers = model.transformer.h
    ca = registry.clusters[cid_a]
    cb = registry.clusters[cid_b]

    if len(ca.members) >= len(cb.members):
        rep_cluster_id = cid_a
        rep_layer_idx = ca.shared_layer_idx
    else:
        rep_cluster_id = cid_b
        rep_layer_idx = cb.shared_layer_idx

    rep_block = layers[rep_layer_idx]
    rep_fc_conv = rep_block.mlp.c_fc.conv
    rep_proj_conv = rep_block.mlp.c_proj.conv

    all_members = ca.members + cb.members
    for m in all_members:
        wrap_mlp_with_lora(layers[m], rank=rank)
        layers[m].mlp.c_fc.conv = rep_fc_conv
        layers[m].mlp.c_proj.conv = rep_proj_conv

    return rep_cluster_id, rep_layer_idx


# =============================================================================
# PHASE 5 — RECOVERY FINE-TUNING
# =============================================================================
def recovery_finetune(
    model,
    dataset,
    tokenizer,
    round_num: int,
    logger: Optional[RunLogger] = None,
    steps: int = RECOVERY_STEPS,
) -> None:
    """Short LM-only fine-tuning pass to let the whole model settle."""
    # Collect all parameters that have requires_grad
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR_RECOVERY)

    for step in tqdm(
        range(steps),
        desc=f"Recovery round {round_num}",
        leave=False,
    ):
        x, mask = get_batch(dataset, tokenizer)
        optimizer.zero_grad()
        loss = model(x, attention_mask=mask, labels=x).loss
        loss.backward()
        optimizer.step()

        should_log = step % RECOVERY_LOG_INTERVAL == 0 or step == steps - 1
        if should_log:
            if logger is not None:
                logger.append_phase_metric("recovery", {
                    "round": round_num,
                    "step": step,
                    "train_loss": loss.item(),
                })
            print(f"  [recovery {step:3d}] LM={loss.item():.4f}")


def uptraining_phase(
    model,
    teacher,
    dataset,
    eval_dataset,
    tokenizer,
    registry: ClusterRegistry,
    logger: Optional[RunLogger] = None,
    steps: int = UPTRAIN_STEPS,
) -> None:
    """
    Final global uptraining with LM loss + teacher-guided distillation.
    All student parameters that require grad remain trainable.
    Shared FFN references are audited throughout the phase.
    """
    model.train()
    teacher.eval()
    assert_shared_ffn_ties(model, registry)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR_UPTRAIN)

    for step in tqdm(
        range(steps),
        desc="Uptraining",
        leave=False,
    ):
        x, mask = get_batch(dataset, tokenizer)
        optimizer.zero_grad()

        student_outputs = model(
            x,
            attention_mask=mask,
            labels=x,
        )
        loss_lm = student_outputs.loss

        with torch.no_grad():
            teacher_outputs = teacher(
                x,
                attention_mask=mask,
            )

        loss_kd = kd_kl_student_teacher(
            student_outputs.logits,
            teacher_outputs.logits,
            mask,
        )
        loss = loss_lm + KD_ALPHA * loss_kd
        loss.backward()
        optimizer.step()

        if step % UPTRAIN_LOG_INTERVAL == 0 or step == steps - 1:
            assert_shared_ffn_ties(model, registry)
            val_ppl = compute_perplexity(model, eval_dataset, tokenizer)
            if logger is not None:
                logger.append_phase_metric("uptraining", {
                    "step": step,
                    "lm_loss": loss_lm.item(),
                    "kd_loss": loss_kd.item(),
                    "total_loss": loss.item(),
                    "val_ppl": val_ppl,
                })
            print(
                f"  [uptrain {step:4d}] "
                f"LM={loss_lm.item():.4f}  "
                f"KD={loss_kd.item():.4f}  "
                f"Total={loss.item():.4f}  "
                f"ValPPL={val_ppl:.2f}"
            )

    assert_shared_ffn_ties(model, registry)


# =============================================================================
# SNAPSHOT / RESTORE UTILITIES
# =============================================================================
def snapshot_model_state(model) -> dict:
    """Deep-copy the full model state dict for rollback."""
    return copy.deepcopy(model.state_dict())


def restore_model_state(model, state: dict) -> None:
    model.load_state_dict(state)


# =============================================================================
# PHASE 0 — INITIAL SETUP
# =============================================================================
def phase0_setup(model_name: str = MODEL_NAME, rank: int = LORA_RANK):
    
    print("=" * 60)
    print("PHASE 0 — Loading model and computing baseline")
    print("=" * 60)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name).to(DEVICE)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.train()
    NUM_LAYERS = len(model.transformer.h)

    before_total  = count_params_total(model)
    before_unique = count_params_unique(model)

    print(f"Params (total)  : {before_total:,}")
    print(f"Params (unique) : {before_unique:,}")
    
    dataset = build_dataset()

    # Wrap every layer with LoRA from the start so forward_base always works
    for layer in model.transformer.h:
        wrap_mlp_with_lora(layer, rank=rank)

    # Frozen originals: deepcopy of each MLP *after* LoRA wrapping
    # but with zero LoRA (A small random, B zero) — effectively just the base.
    # We store the full block so we can call full_mlp_forward on it.
    frozen_originals: Dict[int, nn.Module] = {}
    for i, layer in enumerate(model.transformer.h):
        snap = copy.deepcopy(layer).to(DEVICE).eval()
        for p in snap.parameters():
            p.requires_grad = False
        frozen_originals[i] = snap

    L_orig = evaluate(model, dataset, tokenizer)
    print(f"  Baseline L_orig = {L_orig:.4f}")

    eval_dataset = build_eval_dataset()
    test_dataset = build_test_dataset()

    PPL_base = compute_perplexity(model, eval_dataset, tokenizer)
    PPL_base_test = compute_perplexity(model, test_dataset, tokenizer)

    print(f"  Baseline PPL (correct) = {PPL_base:.2f}")
    print(f"  Baseline Test PPL      = {PPL_base_test:.2f}")
    
    registry = ClusterRegistry(NUM_LAYERS)
    print("  Initial clusters:")
    print(registry.summary())

    # return model, tokenizer, dataset, frozen_originals, registry, L_orig
    # return model, tokenizer, dataset, eval_dataset, frozen_originals, registry, L_orig, PPL_base, before_unique
    return (
        model,
        tokenizer,
        dataset,
        eval_dataset,
        test_dataset,
        frozen_originals,
        registry,
        L_orig,
        PPL_base,
        PPL_base_test,
        before_total,
        before_unique,
    )

# =============================================================================
# MAIN PIPELINE LOOP
# =============================================================================
def run_pipeline(
    model_name: str = MODEL_NAME,
    target_clusters: int = TARGET_CLUSTERS,
    rank: int = LORA_RANK,
    save: bool = False,
):
    config = PipelineConfig(
        model_name=model_name,
        target_clusters=target_clusters,
        rank=rank,
        save=save,
    )
    logger = RunLogger(config, DEVICE)
    merge_history = []

    try:
        (
            model,
            tokenizer,
            dataset,
            eval_dataset,
            test_dataset,
            frozen_originals,
            registry,
            L_orig,
            PPL_base,
            PPL_base_test,
            before_total,
            before_unique,
        ) = phase0_setup(model_name=config.model_name, rank=config.rank)

        logger.set_baseline({
            "L_orig": L_orig,
            "PPL_validation": PPL_base,
            "PPL_test": PPL_base_test,
            "params_total": before_total,
            "params_unique": before_unique,
            "initial_clusters": registry.num_clusters(),
        })
        logger.write_checkpoint("phase0_baseline")

        round_num = 0
        merge_rounds_total = max(registry.num_clusters() - config.target_clusters, 0)

        merge_progress = tqdm(
            total=merge_rounds_total,
            desc="Merge rounds",
            leave=True,
        )
        try:
            while registry.num_clusters() > config.target_clusters:
                round_num += 1
                clusters_before = registry.num_clusters()
                merge_progress.update(1)
                merge_progress.set_postfix({
                    "clusters": clusters_before,
                    "target": config.target_clusters,
                })
                print(f"\n{'='*60}")
                print(f"MERGE ROUND {round_num}  |  clusters={clusters_before}")
                print(f"{'='*60}")

                print("Phase 1 — Caching activations …")
                act_cache = cache_activations(model, dataset, tokenizer)

                print("Phase 1 — Building distance matrix …")
                dist_mat = build_distance_matrix(model, registry, act_cache)
                logger.mark_phase(f"round_{round_num}_distance_matrix")
                logger.write_json()

                sorted_pairs = sorted(dist_mat.items(), key=lambda kv: kv[1])
                print("  Top-5 closest pairs:")
                for (ca, cb), d in sorted_pairs[:5]:
                    print(f"    clusters ({ca},{cb})  D={d:.6f}")

                candidate = pick_merge_candidate(dist_mat, registry)
                if candidate is None:
                    print("  No valid merge candidates remain. Stopping.")
                    logger.mark_phase("no_valid_merge_candidates")
                    logger.write_json()
                    break

                cid_a, cid_b, dist = candidate
                members_a = list(registry.clusters[cid_a].members)
                members_b = list(registry.clusters[cid_b].members)
                print(f"\n  → Selected: cluster {cid_a} ∪ cluster {cid_b}  (D={dist:.6f})")
                print(f"    Members A={members_a}  Members B={members_b}")

                pre_merge_state = snapshot_model_state(model)
                pre_merge_registry = copy.deepcopy(registry)

                rep_a_layer_idx = registry.clusters[cid_a].shared_layer_idx
                rep_b_layer_idx = registry.clusters[cid_b].shared_layer_idx
                all_members = members_a + members_b

                print("\nPhase 2 — Soft alignment setup …")
                print(f"  Representative A: layer {rep_a_layer_idx}")
                print(f"  Representative B: layer {rep_b_layer_idx}")
                print(f"  Active members   : {all_members}")

                print(f"\nPhase 3 — Alignment training ({ALIGN_STEPS} steps) …")
                alignment_training(
                    model, dataset, tokenizer,
                    registry, cid_a, cid_b,
                    frozen_originals, rep_a_layer_idx, rep_b_layer_idx,
                    round_num=round_num,
                    logger=logger,
                )
                logger.write_checkpoint(f"round_{round_num}_alignment")

                print("\nPhase 4 — Evaluating post-alignment …")
                L_post_align = evaluate(model, dataset, tokenizer)
                delta_align = L_post_align - L_orig
                print(f"  L_post_align={L_post_align:.4f}  ΔL={delta_align:+.4f}")

                PPL_post_align = compute_perplexity(model, eval_dataset, tokenizer)
                delta_ppl_align = PPL_post_align - PPL_base
                rel_align = safe_percent_delta(PPL_post_align, PPL_base)
                print(f"  PPL_post_align={PPL_post_align:.2f}  ΔPPL={delta_ppl_align:+.2f}  %Δ={rel_align:+.2f}%")

                merge_record = {
                    "round": round_num,
                    "clusters_before": clusters_before,
                    "merge_pair": [cid_a, cid_b],
                    "distance": dist,
                    "members_A": members_a,
                    "members_B": members_b,
                    "representative_candidates": {
                        "layer_A": rep_a_layer_idx,
                        "layer_B": rep_b_layer_idx,
                    },
                    "L_post_align": L_post_align,
                    "delta_L_align": delta_align,
                    "PPL_post_align": PPL_post_align,
                    "delta_PPL": delta_ppl_align,
                    "percent_delta_PPL": rel_align,
                }

                if delta_align > THRESH_BAD:
                    print(f"  ✗ ΔL={delta_align:.4f} > {THRESH_BAD} — REJECTING merge, rolling back.")
                    restore_model_state(model, pre_merge_state)
                    registry = pre_merge_registry
                    registry.forbid_pair(cid_a, cid_b)
                    merge_record.update({
                        "accepted": False,
                        "rejection_reason": f"delta_L_align > {THRESH_BAD}",
                        "forbidden_pair_after_reject": [cid_a, cid_b],
                    })
                    merge_history.append(merge_record)
                    logger.append_merge(merge_record)
                    logger.write_checkpoint(f"round_{round_num}_rejected")
                    continue

                print("\nPhase 5 — Collapsing to representative base …")
                representative_cluster, representative_layer_idx = collapse_pair_to_representative(
                    model, registry, cid_a, cid_b, config.rank
                )
                print(f"  Representative cluster: {representative_cluster}")
                print(f"  Representative layer  : {representative_layer_idx}")

                print("\nPhase 5 — Committing merge …")
                new_cid = registry.merge(cid_a, cid_b, representative_layer_idx)
                print(f"  New cluster {new_cid}: {registry.clusters[new_cid].members}")
                assert_shared_ffn_ties(model, registry)
                logger.write_checkpoint(f"round_{round_num}_merge_committed")

                print(f"\nPhase 6 — Recovery fine-tuning ({RECOVERY_STEPS} steps) …")
                recovery_finetune(model, dataset, tokenizer, round_num=round_num, logger=logger)
                logger.write_checkpoint(f"round_{round_num}_recovery")

                L_final = evaluate(model, dataset, tokenizer)
                delta_fin = L_final - L_orig
                PPL_final_round = compute_perplexity(model, eval_dataset, tokenizer)
                delta_ppl_final = PPL_final_round - PPL_base
                rel_final = safe_percent_delta(PPL_final_round, PPL_base)
                print(f"\n  L_final={L_final:.4f}  ΔL={delta_fin:+.4f}")

                grade = ("excellent" if delta_fin < THRESH_EXCELLENT
                         else "acceptable" if delta_fin < THRESH_ACCEPTABLE
                         else "borderline")
                print(f"  Grade: {grade}")

                print("\n  Current cluster state:")
                print(registry.summary())

                after_total = count_params_total(model)
                after_unique = count_params_unique(model)
                cr = compression_ratio(before_unique, after_unique)

                print(f"  Params total  : {after_total:,}")
                print(f"  Params unique : {after_unique:,}")
                print(f"  Compression   : {cr:.2f}%")

                merge_record.update({
                    "accepted": True,
                    "representative_cluster": representative_cluster,
                    "representative_layer": representative_layer_idx,
                    "new_cluster": new_cid,
                    "members": list(registry.clusters[new_cid].members),
                    "clusters_after": registry.num_clusters(),
                    "L_final": L_final,
                    "delta_L_final": delta_fin,
                    "PPL_final": PPL_final_round,
                    "delta_PPL_final": delta_ppl_final,
                    "percent_delta_PPL_final": rel_final,
                    "grade": grade,
                    "params_total_after": after_total,
                    "params_unique_after": after_unique,
                    "compression_after": cr,
                })

                merge_history.append(merge_record)
                logger.append_merge(merge_record)
                logger.write_checkpoint(f"round_{round_num}_logged")
        finally:
            merge_progress.close()

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"  Final cluster count : {registry.num_clusters()}")
        print(f"  L_orig              : {L_orig:.4f}")
        L_end = evaluate(model, dataset, tokenizer)
        print(f"  L_final             : {L_end:.4f}")
        print(f"  Total ΔL            : {L_end - L_orig:+.4f}")

        PPL_final = compute_perplexity(model, eval_dataset, tokenizer)
        delta_ppl = PPL_final - PPL_base
        rel = safe_percent_delta(PPL_final, PPL_base)

        print(f"  Final PPL           : {PPL_final:.2f}")
        print(f"  Total ΔPPL          : {delta_ppl:+.2f}")
        print(f"  Total %Δ            : {rel:+.2f}%")

        final_unique = count_params_unique(model)
        cr = compression_ratio(before_unique, final_unique)
        print(f"\nFinal compression: {cr:.2f}%")

        print("\nFinal cluster layout:")
        print(registry.summary())
        logger.write_checkpoint("pipeline_complete_pre_uptraining")

        print("\n" + "=" * 60)
        print("PHASE 7 — UPTRAINING")
        print("=" * 60)
        assert_shared_ffn_ties(model, registry)
        teacher = build_teacher_model(config.model_name)
        uptraining_dataset = build_uptraining_dataset()

        print(f"  Uptraining steps     : {UPTRAIN_STEPS}")
        print(f"  Uptraining LR        : {LR_UPTRAIN}")
        print(f"  KD alpha             : {KD_ALPHA}")
        print(f"  Pre-uptraining loss  : {L_end:.4f}")
        print(f"  Pre-uptraining PPL   : {PPL_final:.2f}")

        uptraining_phase(
            model,
            teacher,
            uptraining_dataset,
            eval_dataset,
            tokenizer,
            registry,
            logger=logger,
        )
        logger.write_checkpoint("phase7_uptraining")

        assert_shared_ffn_ties(model, registry)
        L_post_uptrain = evaluate(model, dataset, tokenizer)
        PPL_post_uptrain = compute_perplexity(model, eval_dataset, tokenizer)

        print("\nPost-uptraining summary:")
        print(f"  L_post_uptrain       : {L_post_uptrain:.4f}")
        print(f"  ΔL vs baseline       : {L_post_uptrain - L_orig:+.4f}")
        print(f"  ΔL vs pre-uptraining : {L_post_uptrain - L_end:+.4f}")
        print(f"  PPL_post_uptrain     : {PPL_post_uptrain:.2f}")
        print(f"  ΔPPL vs baseline     : {PPL_post_uptrain - PPL_base:+.2f}")
        print(f"  ΔPPL vs pre-uptrain  : {PPL_post_uptrain - PPL_final:+.2f}")

        print("\n" + "=" * 60)
        print("FINAL TEST SET EVALUATION")
        print("=" * 60)

        PPL_test = compute_perplexity(model, test_dataset, tokenizer)
        delta_ppl_test = PPL_test - PPL_base_test
        percent_delta_ppl_test = safe_percent_delta(PPL_test, PPL_base_test)
        final_total = count_params_total(model)
        final_unique = count_params_unique(model)
        final_compression = compression_ratio(before_unique, final_unique)

        print(f"  Test PPL (final)     : {PPL_test:.2f}")
        print(f"  Test PPL (baseline)  : {PPL_base_test:.2f}")
        print(f"  Test ΔPPL            : {delta_ppl_test:+.2f}")
        print(f"  Test %Δ              : {percent_delta_ppl_test:+.2f}%")

        logger.set_final({
            "clusters": registry.num_clusters(),
            "total_merges": sum(1 for entry in merge_history if entry.get("accepted")),
            "metrics": {
                "L_final": L_post_uptrain,
                "PPL_validation": PPL_post_uptrain,
                "PPL_test": PPL_test,
                "delta_PPL_validation": PPL_post_uptrain - PPL_base,
                "percent_delta_PPL_validation": safe_percent_delta(PPL_post_uptrain, PPL_base),
                "delta_PPL_test": delta_ppl_test,
                "percent_delta_PPL_test": percent_delta_ppl_test,
            },
            "compression": {
                "params_before": before_unique,
                "params_after": final_unique,
                "params_total_before": before_total,
                "params_total_after": final_total,
                "compression_percent": final_compression,
            },
            "cluster_layout": registry.summary(),
        })

        print("\nMerge history:")
        for entry in merge_history:
            print(" ", json.dumps(entry, indent=2))

        logger.finalize(status="completed", phase_name="complete")
        output_dir = None
        if config.save:
            output_dir = export_final_artifacts(
                model=model,
                tokenizer=tokenizer,
                config=config,
                registry=registry,
                log_path=logger.json_path,
                compression_percent=final_compression,
                final_ppl=PPL_post_uptrain,
            )
            print(f"\nSaved final artifacts to: {output_dir}")
        return model, registry, merge_history, logger.json_path, output_dir
    except Exception as exc:
        logger.set_status("failed", logger.data.get("last_completed_phase", "failed"), error=str(exc))
        logger.finalize(status="failed", phase_name=logger.data.get("last_completed_phase", "failed"))
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the FFN clustering and merging pipeline with configurable inputs.",
    )
    parser.add_argument(
        "--model_name",
        default=MODEL_NAME,
        help="Hugging Face model name to load, e.g. gpt2 or gpt2-medium.",
    )
    parser.add_argument(
        "--target_clusters",
        type=int,
        default=TARGET_CLUSTERS,
        help="Desired number of clusters after merging.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=LORA_RANK,
        help="LoRA rank used for parameter-efficient adaptation.",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help="Set to 1 to save the final model, tokenizer, config, clusters, and metadata.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.target_clusters <= 0:
        raise ValueError("--target_clusters must be a positive integer.")
    if args.rank <= 0:
        raise ValueError("--rank must be a positive integer.")
    if args.save not in (0, 1):
        raise ValueError("--save must be 0 or 1.")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    args = parse_args()
    validate_args(args)
    model, registry, history, log_path, output_dir = run_pipeline(
        model_name=args.model_name,
        target_clusters=args.target_clusters,
        rank=args.rank,
        save=bool(args.save),
    )
    print(f"\nRun log saved to: {log_path}")
    if output_dir is not None:
        print(f"Final model bundle saved to: {output_dir}")
