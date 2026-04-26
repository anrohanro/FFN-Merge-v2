import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm


# MODEL_GRID = {
#     "gpt2": [4, 6, 10],
#     "gpt2-medium": [8, 12, 20],
#     "gpt2-xl": [16, 24, 40],
# }
# RANK_GRID = [6, 9, 12]




MODEL_GRID = {
    "gpt2": [8, 9, 10]
}
RANK_GRID = [6, 9, 12]



ROOT_DIR = Path(__file__).resolve().parent
PIPELINE_SCRIPT = ROOT_DIR / "v5.py"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"


def build_runs() -> list[dict]:
    runs = []
    for model_name, cluster_values in MODEL_GRID.items():
        for target_clusters in cluster_values:
            for rank in RANK_GRID:
                runs.append({
                    "model_name": model_name,
                    "target_clusters": target_clusters,
                    "rank": rank,
                    "save": 1,
                })
    return runs


def run_name(run: dict) -> str:
    safe_model_name = run["model_name"].replace("/", "_")
    return f"{safe_model_name}_clusters{run['target_clusters']}_rank{run['rank']}"


def main() -> None:
    runs = build_runs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = EXPERIMENTS_DIR / f"grid_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = experiment_dir / "manifest.json"
    results_path = experiment_dir / "results.jsonl"

    manifest = {
        "created_at": timestamp,
        "total_runs": len(runs),
        "pipeline_script": str(PIPELINE_SCRIPT),
        "runs": runs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    print(f"Experiment manifest: {manifest_path}")
    print(f"Results log: {results_path}")
    print(f"Total runs: {len(runs)}")

    with results_path.open("a", encoding="utf-8") as results_file:
        progress = tqdm(runs, desc="Compression experiments", unit="run")
        for run in progress:
            name = run_name(run)
            progress.set_postfix({
                "model": run["model_name"],
                "clusters": run["target_clusters"],
                "rank": run["rank"],
            })

            cmd = [
                sys.executable,
                str(PIPELINE_SCRIPT),
                "--model_name", run["model_name"],
                "--target_clusters", str(run["target_clusters"]),
                "--rank", str(run["rank"]),
                "--save", str(run["save"]),
            ]

            print("\n" + "=" * 80)
            print(f"RUN START: {name}")
            print(" ".join(cmd))
            print("=" * 80)

            started_at = datetime.now().isoformat()
            completed = subprocess.run(
                cmd,
                cwd=ROOT_DIR,
                env=env,
                check=False,
            )
            finished_at = datetime.now().isoformat()

            result = {
                "name": name,
                "started_at": started_at,
                "finished_at": finished_at,
                "returncode": completed.returncode,
                **run,
            }
            results_file.write(json.dumps(result) + "\n")
            results_file.flush()

            if completed.returncode == 0:
                print(f"RUN COMPLETE: {name}")
            else:
                print(f"RUN FAILED: {name} (return code {completed.returncode})")
                raise subprocess.CalledProcessError(completed.returncode, cmd)

    print("\nAll compression experiments completed successfully.")
    print(f"Manifest: {manifest_path}")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
