import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
import timm
import numpy as np

DATA_ROOT = Path("/content/drive/MyDrive/Data_seasons")
ARTIFACTS = Path("/content/drive/MyDrive/Trend_artifacts")
MODEL_PATH = ARTIFACTS / "deit_fashion.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
NUM_CLASSES = 6

def load_model():
    model = timm.create_model(
        "deit_small_patch16_224",
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

tfm = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

def get_image_paths(season_dir):
    return [p for p in season_dir.iterdir() if p.is_file()]

def extract_probabilities(model, img_paths):
    probs = []

    for p in img_paths:
        try:
            img = Image.open(p).convert("RGB")
        except:
            continue
        x = tfm(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            prob = F.softmax(logits, dim=1).cpu().numpy()[0]
        probs.append(prob)

    if len(probs) == 0:
        return np.zeros(NUM_CLASSES)

    return np.mean(probs, axis=0)

def main():
    model = load_model()

    seasons = sorted(DATA_ROOT.iterdir())
    rows = []

    for season_dir in seasons:
        if not season_dir.is_dir():
            continue

        season_id = int(str(season_dir.name).replace("season_", ""))
        img_paths = get_image_paths(season_dir)

        mean_probs = extract_probabilities(model, img_paths)

        row = {"season_id": season_id}
        for i in range(NUM_CLASSES):
            row[f"h_{i}"] = mean_probs[i]

        rows.append(row)
        print(f"Processed Season {season_id} with {len(img_paths)} images")

    df = pd.DataFrame(rows).sort_values("season_id")
    out_path = ARTIFACTS / "season_features.csv"
    df.to_csv(out_path, index=False)
    print("Wrote", out_path)

if __name__ == "__main__":
    main()
