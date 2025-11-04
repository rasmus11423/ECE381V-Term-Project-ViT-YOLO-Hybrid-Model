import neptune

run = neptune.init_run(
    project="ECE381V-Deep-Learning/ECE381V-Term-Project-ViT-YOLO-Hybrid-Model",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NmUxNTU0Ni1jMDFkLTRmNjctOGJmOS0yNGNlOTc4MmVhY2UifQ==",
)  # your credentials

params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params

for epoch in range(10):
    run["train/loss"].append(0.9 ** epoch)

run["eval/f1_score"] = 0.66

run.stop()