import argparse, yaml, os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_prep import make_loaders
from models.custom_mobilenet import CustomMobileNet
from utils.common import ensure_dir

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss / max(len(loader),1), correct / max(total,1)

def main(cfg):
    ensure_dir(os.path.dirname(cfg["paths"]["model_out"]))
    ensure_dir(cfg["paths"]["plots_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = make_loaders(
        cfg["data"]["train_csv"], cfg["data"]["test_csv"], cfg["data"]["img_size"],
        cfg["data"]["batch_size"], cfg["data"]["val_split"], cfg["data"]["seed"]
    )

    model = CustomMobileNet(cfg["train"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    best_val, patience, counter = float("inf"), cfg["train"]["patience"], 0
    tr_accs, va_accs, tr_losses, va_losses = [], [], [], []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running, correct, total = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['train']['epochs']}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss = running / max(len(train_loader),1)
        train_acc = correct / max(total,1)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        tr_losses.append(train_loss); va_losses.append(val_loss)
        tr_accs.append(train_acc);   va_accs.append(val_acc)

        print(f"Epoch {epoch}: Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}")

        if val_loss < best_val:
            best_val, counter = val_loss, 0
            torch.save(model.state_dict(), cfg["paths"]["model_out"])
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break

    # plots
    import numpy as np
    epochs = np.arange(1, len(tr_accs) + 1)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, tr_accs, label="Train Acc")
    plt.plot(epochs, va_accs, label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(epochs, tr_losses, label="Train Loss")
    plt.plot(epochs, va_losses, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["paths"]["plots_dir"], "training_curves.png"), dpi=200)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
