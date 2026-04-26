# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer,AutoModelForCausalLM, AutoTokenizer
# from datasets import load_dataset
# from tqdm import tqdm
# import numpy as np

# # # 1. Load model + tokenizer
# model_name = "gpt2"  # or "gpt2-medium" for faster eval
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)
# model.eval()



# # from pathlib import Path
# # from inference import load_bundle   # adjust import

# # BUNDLE_DIR = Path("/data/user7/c2/outputs/gpt2_clusters9_rank8")

# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model, tokenizer, _, _, _, _ = load_bundle(BUNDLE_DIR)
# # model = model.to(DEVICE)
# # model.eval()





# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)


# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model.to(device)

# # GPT-2 fix (no pad token)
# tokenizer.pad_token = tokenizer.eos_token

# # 2. Load dataset (validation split)
# dataset = load_dataset("hellaswag", split="validation[:1000]")  
# # limit for speed (remove [:1000] for full eval)



# def score_choice(context, ending):
#     # tokenize separately
#     context_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
#     ending_ids = tokenizer(" " + ending, return_tensors="pt").input_ids.to(device)

#     input_ids = torch.cat([context_ids, ending_ids], dim=1)

#     # mask: only compute loss on ending
#     labels = input_ids.clone()
#     labels[:, :context_ids.shape[1]] = -100  # ignore context

#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, labels=labels)
#         loss = outputs.loss

#     return -loss.item()



# # 4. Evaluation loop
# correct = 0
# total = 0
# failures = []
# for example in tqdm(dataset):
#     context = example["ctx"]
#     endings = example["endings"]
#     label = int(example["label"])

#     scores = []
#     for ending in endings:
#         score = score_choice(context, ending)
#         scores.append(score)

#     pred = np.argmax(scores)

#     if pred == label:
#         correct += 1
#     else:
#         # store failure case
#         failures.append({
#             "context": context,
#             "endings": endings,
#             "pred": pred,
#             "label": label
#         })

#     total += 1
    
# accuracy = correct / total
# print(f"\nZero-shot Accuracy (HellaSwag): {accuracy:.4f}")

# print("\n--- Showing 10 Failure Cases ---\n")

# for i, fail in enumerate(failures[:10]):
#     print(f"\nExample {i+1}")
#     print("Context:")
#     print(fail["context"])

#     print("\nChoices:")
#     for j, ending in enumerate(fail["endings"]):
#         marker = ""
#         if j == fail["label"]:
#             marker += " (CORRECT)"
#         if j == fail["pred"]:
#             marker += " (PREDICTED)"
#         print(f"{j}: {ending}{marker}")












import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from scipy import stats

# 1. Load model + tokenizer
# model_name = "gpt2"
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)
# model.eval()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)






from pathlib import Path
from inference import load_bundle   # adjust import

BUNDLE_DIR = Path("/data/user7/c2/outputs/gpt2_clusters9_rank12")
print(f"Loading compressed model from: {BUNDLE_DIR}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, _, _, _, _ = load_bundle(BUNDLE_DIR)
model = model.to(device)
model.eval()

# GPT-2 fix (no pad token)
tokenizer.pad_token = tokenizer.eos_token

# 2. Load dataset (validation split)
dataset = load_dataset("hellaswag", split="validation[:1000]")


def score_choice(context, ending):
    context_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
    ending_ids = tokenizer(" " + ending, return_tensors="pt").input_ids.to(device)

    input_ids = torch.cat([context_ids, ending_ids], dim=1)

    labels = input_ids.clone()
    labels[:, :context_ids.shape[1]] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

    return -loss.item()


def wilson_confidence_interval(correct, total, confidence=0.95):
    """
    Wilson score interval — more accurate than normal approximation,
    especially for small n or extreme proportions.
    """
    if total == 0:
        return 0.0, 0.0

    z = stats.norm.ppf(1 - (1 - confidence) / 2)  # z = 1.96 for 95%
    p_hat = correct / total
    n = total

    center = (p_hat + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / (1 + z**2 / n)

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


def normal_approximation_ci(correct, total, confidence=0.95):
    """
    Standard normal approximation interval (Wald interval).
    Works well when n is large and p is not too extreme.
    """
    if total == 0:
        return 0.0, 0.0

    p_hat = correct / total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    margin = z * np.sqrt(p_hat * (1 - p_hat) / total)

    lower = max(0.0, p_hat - margin)
    upper = min(1.0, p_hat + margin)
    return lower, upper


# 4. Evaluation loop
correct = 0
total = 0
failures = []
correct_flags = []  # track per-example results for CI

for example in tqdm(dataset):
    context = example["ctx"]
    endings = example["endings"]
    label = int(example["label"])

    scores = [score_choice(context, ending) for ending in endings]
    pred = np.argmax(scores)

    is_correct = int(pred == label)
    correct_flags.append(is_correct)

    if is_correct:
        correct += 1
    else:
        failures.append({
            "context": context,
            "endings": endings,
            "pred": pred,
            "label": label
        })

    total += 1


# ── Results ──────────────────────────────────────────────────────────────────
accuracy = correct / total

wilson_lower, wilson_upper = wilson_confidence_interval(correct, total, confidence=0.95)
wald_lower, wald_upper = normal_approximation_ci(correct, total, confidence=0.95)

# Standard error and margin of error
std_err = np.std(correct_flags, ddof=1) / np.sqrt(total)  # SEM from binary outcomes
z_95 = stats.norm.ppf(0.975)

print(f"\n{'='*55}")
print(f"  Zero-shot Evaluation on HellaSwag (n={total})")
print(f"{'='*55}")
print(f"  Accuracy              : {accuracy:.4f}  ({correct}/{total})")
print(f"  Std Error (SEM)       : ±{std_err:.4f}")
print(f"  95% CI  [Wilson]      : [{wilson_lower:.4f}, {wilson_upper:.4f}]")
print(f"  95% CI  [Wald/Normal] : [{wald_lower:.4f}, {wald_upper:.4f}]")
print(f"{'='*55}\n")


# print("\n--- Showing 10 Failure Cases ---\n")
# for i, fail in enumerate(failures[:10]):
#     print(f"\nExample {i+1}")
#     print("Context:")
#     print(fail["context"])
#     print("\nChoices:")
#     for j, ending in enumerate(fail["endings"]):
#         marker = ""
#         if j == fail["label"]:
#             marker += " (CORRECT)"
#         if j == fail["pred"]:
#             marker += " (PREDICTED)"
#         print(f"  {j}: {ending}{marker}")