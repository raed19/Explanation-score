import argparse, yaml, torch, numpy as np
from tqdm import tqdm
from data_prep import make_loaders
from models.custom_mobilenet import CustomMobileNet
from explainability.lime_utils import explain_prediction
from explainability.hand_mask import generate_hand_mask
from explainability.qmex import compute_explainability_score

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = make_loaders(
        cfg["data"]["train_csv"], cfg["data"]["test_csv"], cfg["data"]["img_size"],
        cfg["data"]["batch_size"], cfg["data"]["val_split"], cfg["data"]["seed"]
    )
    model = CustomMobileNet(cfg["train"]["num_classes"]).to(device)
    model.load_state_dict(torch.load(cfg["paths"]["model_out"], map_location=device))
    model.eval()

    lime_scores = []
    for images, labels in tqdm(test_loader, desc="Evaluating Q-MEX (LIME + MediaPipe)"):
        images, labels = images.to(device), labels.to(device)
        for i in range(images.size(0)):
            img_chw = images[i].detach().cpu().numpy()          # (1,H,W)
            label = int(labels[i].item())
            _, lime_mask = explain_prediction(img_chw, label, model, device)
            hand_mask = generate_hand_mask(img_chw)
            score = compute_explainability_score(lime_mask, hand_mask)
            lime_scores.append(score)

    print(f"Q-MEX (LIME) — Avg: {np.mean(lime_scores):.4f} | Std: {np.std(lime_scores):.4f} | N={len(lime_scores)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
