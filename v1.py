import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Logging
# -------------------------
log_file = "gravity_functional.jsonl"

def log(data):
    with open(log_file, "a") as f:
        f.write(json.dumps(data) + "\n")

# -------------------------
# Model
# -------------------------
model_name = "gpt2"

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# Dataset
# -------------------------
dataset = load_dataset(
    "wikitext",
    "wikitext-2-raw-v1",
    split="train[:1%]"
)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# -------------------------
# FFN layers
# -------------------------
ffn_layers = [block.mlp.c_fc for block in model.transformer.h]

# -------------------------
# Metrics (important only)
# -------------------------
def compute_metrics(ffn_layers):
    weights = [layer.weight.data for layer in ffn_layers]
    n = len(weights)

    # pairwise distance (parameter space just for monitoring)
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dists.append(torch.norm(weights[i] - weights[j]).item())

    return {
        "avg_weight_distance": sum(dists) / len(dists)
    }

# -------------------------
# Functional gravity
# -------------------------
def apply_gravity(ffn_layers, hidden_states, lambda_=1e-3, eps=1e-6, clip_val=1e-3):
    n = len(ffn_layers)

    # flatten hidden states
    h = hidden_states.reshape(-1, hidden_states.size(-1))

    for i in range(n):
        total_update = torch.zeros_like(ffn_layers[i].weight)

        for j in range(n):
            if i == j:
                continue

            ffn_i = ffn_layers[i]
            ffn_j = ffn_layers[j]

            # forward
            out_i = ffn_i(h)
            out_j = ffn_j(h).detach()

            # functional distance
            d_ij = torch.mean((out_i - out_j) ** 2)

            # gradient wrt Wi
            grad_Wi = torch.autograd.grad(
                d_ij,
                ffn_i.weight,
                retain_graph=True,
                create_graph=False
            )[0]

            direction = -grad_Wi
            direction = direction / (torch.norm(direction) + eps)

            sim = 1.0 / (d_ij.detach() + eps)

            force = sim * direction
            total_update += force

        total_update = torch.clamp(total_update, -clip_val, clip_val)
        ffn_layers[i].weight.data += lambda_ * total_update

# -------------------------
# Optimizer
# -------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# -------------------------
# Training loop
# -------------------------
model.train()

max_steps = 100

for step, batch in enumerate(loader):

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
        output_hidden_states=True
    )

    loss = outputs.loss
    hidden_states = outputs.hidden_states[-1]  # IMPORTANT: no detach

    loss.backward()

    # -------------------------
    # Logging (before updates)
    # -------------------------
    if step % 10 == 0:
        metrics = compute_metrics(ffn_layers)

        log_data = {
            "step": step,
            "loss": loss.item(),
            **metrics
        }

        log(log_data)

        print(f"Step {step} | Loss {loss.item():.4f} | "
              f"WeightDist {metrics['avg_weight_distance']:.4f}")

    # -------------------------
    # Standard update
    # -------------------------
    optimizer.step()

    # -------------------------
    # Functional gravity
    # -------------------------
    apply_gravity(ffn_layers, hidden_states)

    optimizer.zero_grad()

    if step > max_steps:
        break