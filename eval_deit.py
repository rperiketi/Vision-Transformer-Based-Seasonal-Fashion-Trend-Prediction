from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

NUM_CLASSES = 6
IMG_SIZE = 224

DATA_ROOT = Path("/content/drive/MyDrive/Data_seasons")
IMG_CSV = Path("/content/drive/MyDrive/Data_labels/img_labels.csv")
ARTIFACTS = Path("/content/drive/MyDrive/Trend_artifacts")
MODEL_PATH = ARTIFACTS / "deit_fashion.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["stripes", "floral", "polka", "geometric", "solid", "gingham"]


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
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        y = int(row["pattern_id"])
        return img, y


def get_transforms():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])


def build_model():
    model = timm.create_model(
        "deit_small_patch16_224",
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    return model


def main():
    df = pd.read_csv(IMG_CSV)
    df = df[df["season_id"] <= 10].copy()
    df = df.dropna(subset=["pattern_id"])
    df = df[df["pattern_id"] != ""]
    df["pattern_id"] = df["pattern_id"].astype(int)

    train_idx, val_idx = train_test_split(
        df.index, test_size=0.1, stratify=df["pattern_id"], random_state=42
    )
    df_val = df.loc[val_idx].copy()
    print(f"Validation set size: {len(df_val)} images")

    tfm = get_transforms()
    val_ds = ClothingDataset(df_val, DATA_ROOT, tfm)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_y = []
    all_pred = []

    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(1).cpu().numpy()
            all_pred.append(preds)
            all_y.append(y.numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)

    print("=== Classification report (validation set) ===")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(xticks_rotation=45)
    plt.title("DeiT Pattern Classifier – Confusion Matrix (Validation)")
    plt.tight_layout()

    out_path = ARTIFACTS / "confusion_matrix_val.png"
    plt.savefig(out_path, dpi=200)
    
    print("Saved confusion matrix to", out_path)


if __name__ == "__main__":
    main()
