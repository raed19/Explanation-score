import argparse, os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image

def preprocess_images(x_flat: torch.Tensor, img_size: int) -> torch.Tensor:
    x = x_flat.view(-1, 1, 28, 28)  # (N,1,28,28)
    tfm = transforms.Compose([transforms.Resize((img_size, img_size)),
                              transforms.ToTensor()])
    out = torch.empty((x.size(0), 1, img_size, img_size), dtype=torch.float32)
    for i in range(x.size(0)):
        img_np = x[i, 0].numpy().astype("uint8")
        pil = Image.fromarray(img_np, mode="L")
        out[i] = tfm(pil)
    return out

def make_loaders(train_csv, test_csv, img_size, batch_size, val_split, seed):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    x_train_full = train_df.drop(columns=["label"]).values.astype(np.float32)
    y_train_full = train_df["label"].values.astype(np.int64)
    x_test = test_df.drop(columns=["label"]).values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.int64)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=val_split, random_state=seed, stratify=y_train_full
    )

    x_train = torch.from_numpy(x_train)
    x_val   = torch.from_numpy(x_val)
    x_test  = torch.from_numpy(x_test)
    y_train = torch.from_numpy(y_train)
    y_val   = torch.from_numpy(y_val)
    y_test  = torch.from_numpy(y_test)

    X_train = preprocess_images(x_train, img_size)
    X_val   = preprocess_images(x_val, img_size)
    X_test  = preprocess_images(x_test, img_size)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train_loader, val_loader, test_loader = make_loaders(
        args.train_csv, args.test_csv, args.img_size, args.batch_size, args.val_split, args.seed
    )
