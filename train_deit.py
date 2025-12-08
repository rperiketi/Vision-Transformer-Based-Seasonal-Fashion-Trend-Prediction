import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import timm
from timm.loss import LabelSmoothingCrossEntropy

# ---------- CONFIG ----------
NUM_CLASSES = 6
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-4
WEIGHT_DECAY = 0.05
SMOOTHING = 0.1

DATA_ROOT = Path("/content/drive/MyDrive/Data_seasons")
IMG_CSV = Path("/content/drive/MyDrive/Data_labels/img_labels.csv")
ARTIFACTS = Path("/content/drive/MyDrive/Trend_artifacts")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- DATASET ----------
class ClothingDataset(Dataset):
    def __init__(self, df, root_dir: Path, transform):
        self.df = df.reset_index(drop=True)
        self.root = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row["image_path"]

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print("BAD IMAGE:", img_path)
            raise

        y = int(row["pattern_id"])
        img = self.transform(img)
        return img, y


def get_transforms(train=True):
    if train:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
            T.RandomErasing(p=0.25, scale=(0.02, 0.2), value=0),
        ])
    else:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])


# ---------- MODEL ----------
def build_model():
    model = timm.create_model(
        "deit_small_patch16_224",
        pretrained=True,
        num_classes=NUM_CLASSES,
    )
    return model


def train_one_epoch(model, loader, opt, crit):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()

        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = crit(logits, y)

        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return loss_sum / total, correct / total


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IMG_CSV)

    # keep seasons 1–10
    df = df[df["season_id"] <= 10].copy()

    # drop unlabeled
    df = df.dropna(subset=["pattern_id"])
    df = df[df["pattern_id"] != ""]
    df["pattern_id"] = df["pattern_id"].astype(int)

    train_idx, val_idx = train_test_split(
        df.index, test_size=0.1, stratify=df["pattern_id"], random_state=42
    )

    df_train = df.loc[train_idx]
    df_val = df.loc[val_idx]

    t_train = get_transforms(train=True)
    t_val = get_transforms(train=False)

    train_ds = ClothingDataset(df_train, DATA_ROOT, t_train)
    val_ds = ClothingDataset(df_val, DATA_ROOT, t_val)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = LabelSmoothingCrossEntropy(smoothing=SMOOTHING)

    best_val = math.inf
    best_path = ARTIFACTS / "deit_fashion.pt"

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, opt, crit)
        va_loss, va_acc = evaluate(model, val_dl, nn.CrossEntropyLoss())

        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)

    print("Saved best model to", best_path)


if __name__ == "__main__":
    main()
