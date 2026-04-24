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

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import random
import copy
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# =============================================================================
# CONFIG
# =============================================================================
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE     = 8
SEQ_LEN        = 64
CALIB_BATCHES  = 20       # batches used for distance matrix computation
ALIGN_STEPS    = 2000     # steps for alignment training per merge round
RECOVERY_STEPS = 300      # LM-only fine-tuning after merge
LR_ALIGN       = 3e-5
LR_RECOVERY    = 1e-5
LAMBDA_MAX     = 3.0      # weight on (align + anchor)
MU             = 1.0      # weight on anchor inside the structural term
LORA_RANK      = 8
WARMUP_FRAC    = 0.3      # fraction of ALIGN_STEPS used for lambda warmup

TARGET_CLUSTERS = 8       # stop when this many clusters remain

# Merge acceptance thresholds
THRESH_EXCELLENT  = 0.1
THRESH_ACCEPTABLE = 0.3
THRESH_BAD        = 0.5

NUM_LAYERS = 12           # GPT-2 small

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


def wrap_mlp_with_lora(block: nn.Module, rank: int = LORA_RANK) -> None:
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
def evaluate(model, dataset, tokenizer, num_batches: int = 30) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            x, m = get_batch(dataset, tokenizer)
            loss = model(x, attention_mask=m, labels=x).loss
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


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
    cache: Dict[int, List[torch.Tensor]] = {i: [] for i in range(NUM_LAYERS)}
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
        for _ in range(num_batches):
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

    for step in range(steps):
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

        if step % 200 == 0:
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
        wrap_mlp_with_lora(layers[m])
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
    steps: int = RECOVERY_STEPS,
) -> None:
    """Short LM-only fine-tuning pass to let the whole model settle."""
    # Collect all parameters that have requires_grad
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR_RECOVERY)

    for step in range(steps):
        x, mask = get_batch(dataset, tokenizer)
        optimizer.zero_grad()
        loss = model(x, attention_mask=mask, labels=x).loss
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"  [recovery {step:3d}] LM={loss.item():.4f}")


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
def phase0_setup():
    print("=" * 60)
    print("PHASE 0 — Loading model and computing baseline")
    print("=" * 60)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.train()

    dataset = build_dataset()

    # Wrap every layer with LoRA from the start so forward_base always works
    for layer in model.transformer.h:
        wrap_mlp_with_lora(layer)

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

    registry = ClusterRegistry(NUM_LAYERS)
    print("  Initial clusters:")
    print(registry.summary())

    return model, tokenizer, dataset, frozen_originals, registry, L_orig


# =============================================================================
# MAIN PIPELINE LOOP
# =============================================================================
def run_pipeline():
    model, tokenizer, dataset, frozen_originals, registry, L_orig = phase0_setup()

    merge_history = []  # list of dicts for bookkeeping

    round_num = 0

    while registry.num_clusters() > TARGET_CLUSTERS:
        round_num += 1
        print(f"\n{'='*60}")
        print(f"MERGE ROUND {round_num}  |  clusters={registry.num_clusters()}")
        print(f"{'='*60}")

        # ------------------------------------------------------------------ #
        # PHASE 1 — Build distance matrix
        # ------------------------------------------------------------------ #
        print("Phase 1 — Caching activations …")
        act_cache = cache_activations(model, dataset, tokenizer)

        print("Phase 1 — Building distance matrix …")
        dist_mat  = build_distance_matrix(model, registry, act_cache)

        # Pretty-print a few distances
        sorted_pairs = sorted(dist_mat.items(), key=lambda kv: kv[1])
        print("  Top-5 closest pairs:")
        for (ca, cb), d in sorted_pairs[:5]:
            print(f"    clusters ({ca},{cb})  D={d:.6f}")

        candidate = pick_merge_candidate(dist_mat, registry)
        if candidate is None:
            print("  No valid merge candidates remain. Stopping.")
            break

        cid_a, cid_b, dist = candidate
        print(f"\n  → Selected: cluster {cid_a} ∪ cluster {cid_b}  (D={dist:.6f})")
        print(f"    Members A={registry.clusters[cid_a].members}  "
              f"Members B={registry.clusters[cid_b].members}")

        # ------------------------------------------------------------------ #
        # Snapshot for rollback
        # ------------------------------------------------------------------ #
        pre_merge_state    = snapshot_model_state(model)
        pre_merge_registry = copy.deepcopy(registry)

        rep_a_layer_idx = registry.clusters[cid_a].shared_layer_idx
        rep_b_layer_idx = registry.clusters[cid_b].shared_layer_idx
        all_members = registry.clusters[cid_a].members + registry.clusters[cid_b].members

        # ------------------------------------------------------------------ #
        # PHASE 2 — Soft alignment with two live representative bases
        # ------------------------------------------------------------------ #
        print("\nPhase 2 — Soft alignment setup …")
        print(f"  Representative A: layer {rep_a_layer_idx}")
        print(f"  Representative B: layer {rep_b_layer_idx}")
        print(f"  Active members   : {all_members}")

        # ------------------------------------------------------------------ #
        # PHASE 3 — Alignment training
        # ------------------------------------------------------------------ #
        print(f"\nPhase 3 — Alignment training ({ALIGN_STEPS} steps) …")
        alignment_training(
            model, dataset, tokenizer,
            registry, cid_a, cid_b,
            frozen_originals, rep_a_layer_idx, rep_b_layer_idx,
        )

        # ------------------------------------------------------------------ #
        # PHASE 4 — Evaluate before committing
        # ------------------------------------------------------------------ #
        print("\nPhase 4 — Evaluating post-alignment …")
        L_post_align = evaluate(model, dataset, tokenizer)
        delta_align  = L_post_align - L_orig
        print(f"  L_post_align={L_post_align:.4f}  ΔL={delta_align:+.4f}")

        if delta_align > THRESH_BAD:
            print(f"  ✗ ΔL={delta_align:.4f} > {THRESH_BAD} — REJECTING merge, rolling back.")
            restore_model_state(model, pre_merge_state)
            # Restore registry
            registry = pre_merge_registry
            registry.forbid_pair(cid_a, cid_b)
            merge_history.append({
                "round": round_num,
                "cid_a": cid_a, "cid_b": cid_b,
                "dist": dist,
                "delta_L_post_align": delta_align,
                "accepted": False,
            })
            continue

        # ------------------------------------------------------------------ #
        # PHASE 5 — Collapse to one representative base and commit merge
        # ------------------------------------------------------------------ #
        print("\nPhase 5 — Collapsing to representative base …")
        representative_cluster, representative_layer_idx = collapse_pair_to_representative(
            model, registry, cid_a, cid_b
        )
        print(f"  Representative cluster: {representative_cluster}")
        print(f"  Representative layer  : {representative_layer_idx}")

        print("\nPhase 5 — Committing merge …")
        new_cid = registry.merge(cid_a, cid_b, representative_layer_idx)
        print(f"  New cluster {new_cid}: {registry.clusters[new_cid].members}")

        # ------------------------------------------------------------------ #
        # PHASE 6 — Recovery fine-tuning
        # ------------------------------------------------------------------ #
        print(f"\nPhase 6 — Recovery fine-tuning ({RECOVERY_STEPS} steps) …")
        recovery_finetune(model, dataset, tokenizer)

        L_final   = evaluate(model, dataset, tokenizer)
        delta_fin = L_final - L_orig
        print(f"\n  L_final={L_final:.4f}  ΔL={delta_fin:+.4f}")

        grade = ("excellent" if delta_fin < THRESH_EXCELLENT
                 else "acceptable" if delta_fin < THRESH_ACCEPTABLE
                 else "borderline")
        print(f"  Grade: {grade}")

        merge_history.append({
            "round": round_num,
            "cid_a": cid_a, "cid_b": cid_b,
            "dist": dist,
            "delta_L_post_align": delta_align,
            "delta_L_final": delta_fin,
            "accepted": True,
            "grade": grade,
            "representative_cluster": representative_cluster,
            "representative_layer_idx": representative_layer_idx,
            "new_cluster": new_cid,
            "members": registry.clusters[new_cid].members,
        })

        print("\n  Current cluster state:")
        print(registry.summary())

    # ------------------------------------------------------------------ #
    # DONE
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Final cluster count : {registry.num_clusters()}")
    print(f"  L_orig              : {L_orig:.4f}")
    L_end = evaluate(model, dataset, tokenizer)
    print(f"  L_final             : {L_end:.4f}")
    print(f"  Total ΔL            : {L_end - L_orig:+.4f}")
    print("\nFinal cluster layout:")
    print(registry.summary())

    print("\nMerge history:")
    for entry in merge_history:
        print(" ", json.dumps(entry, indent=2))

    return model, registry, merge_history


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    model, registry, history = run_pipeline()
