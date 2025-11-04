import neptune
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Read API token from file (file is ignored by git)
token_file = script_dir / 'neptune_api_token.txt'
with open(token_file, 'r') as f:
    api_token = f.read().strip()

run = neptune.init_run(
    project="ECE381V-Deep-Learning/ECE381V-Term-Project-ViT-YOLO-Hybrid-Model",
    api_token=api_token,
)  # your credentials

params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params

for epoch in range(10):
    run["train/loss"].append(0.9 ** epoch)

run["eval/f1_score"] = 0.66

run.stop()