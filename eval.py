import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from benchmark_runner import run_model_benchmarks


MODEL_NAME = "gpt2"
# BENCHMARKS = ["SQuAD", "WMT-14 En-Fr", "CNN/DailyMail"]
BENCHMARKS = ["SQuAD"]

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

results = run_model_benchmarks(
    model=model,
    tokenizer=tokenizer,
    benchmarks=BENCHMARKS,
    device=device,
)

print(results)
