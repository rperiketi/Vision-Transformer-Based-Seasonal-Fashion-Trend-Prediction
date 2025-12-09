import os, random, shutil
from pathlib import Path
import pandas as pd

# RAW images
RAW_ROOT = Path("/content/drive/MyDrive/Data")

DATA_ROOT = Path("/content/drive/MyDrive/Data_seasons")
LABELS_ROOT = Path("/content/drive/MyDrive/Data_labels")

NUM_SEASONS = 11
TRAIN_SEASONS = list(range(1, 11))  # 1..10
TEST_SEASON = 11                    # 11

PATTERN_MAP = {
    "stripes": 0,
    "floral": 1,
    "polka": 2,
    "geometric": 3,
    "solid": 4,
    "gingham": 5,
}

# allowed image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif"}

random.seed(27)

def main():
    DATA_ROOT.mkdir(exist_ok=True)
    LABELS_ROOT.mkdir(exist_ok=True)

    rows = []

    for pattern_name, pattern_id in PATTERN_MAP.items():
        src_dir = RAW_ROOT / pattern_name
        if not src_dir.exists():
            print(f"WARNING: {src_dir} does not exist, skipping.")
            continue

        images = [
            p for p in src_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]
        random.shuffle(images)

        n = len(images)
        print(f"{pattern_name}: {n} valid images")
        if n == 0:
            print(f"No images in {src_dir}, skipping.")
            continue

        # 10% -> season 11, 90% -> seasons 1–10
        n_test = max(1, int(0.1 * n))
        test_imgs = images[:n_test]
        train_imgs = images[n_test:]

        per_season = max(1, len(train_imgs) // len(TRAIN_SEASONS))
        idx = 0

        # Distribute train images across seasons 1–10
        for season in TRAIN_SEASONS:
            season_dir = DATA_ROOT / f"season_{season:02d}"
            season_dir.mkdir(parents=True, exist_ok=True)
            for _ in range(per_season):
                if idx >= len(train_imgs):
                    break
                src = train_imgs[idx]
                dst = season_dir / src.name
                shutil.copy2(src, dst)
                rel_path = f"season_{season:02d}/{src.name}"
                rows.append({
                    "image_path": rel_path,
                    "season_id": season,
                    "pattern_id": pattern_id,
                })
                idx += 1

        # Leftover training images -> random training seasons
        while idx < len(train_imgs):
            src = train_imgs[idx]
            season = random.choice(TRAIN_SEASONS)
            season_dir = DATA_ROOT / f"season_{season:02d}"
            season_dir.mkdir(parents=True, exist_ok=True)
            dst = season_dir / src.name
            shutil.copy2(src, dst)
            rel_path = f"season_{season:02d}/{src.name}"
            rows.append({
                "image_path": rel_path,
                "season_id": season,
                "pattern_id": pattern_id,
            })
            idx += 1

        # Season 11 -> unlabeled patterns
        season11_dir = DATA_ROOT / f"season_{TEST_SEASON:02d}"
        season11_dir.mkdir(parents=True, exist_ok=True)
        for src in test_imgs:
            dst = season11_dir / src.name
            shutil.copy2(src, dst)
            rel_path = f"season_{TEST_SEASON:02d}/{src.name}"
            rows.append({
                "image_path": rel_path,
                "season_id": TEST_SEASON,
                "pattern_id": "",  # unlabeled for season 11
            })

    df = pd.DataFrame(rows)
    df.to_csv(LABELS_ROOT / "img_labels.csv", index=False)
    print("Wrote", LABELS_ROOT / "img_labels.csv")

    trend_path = LABELS_ROOT / "season_trend_labels.csv"
    if not trend_path.exists():
        df2 = pd.DataFrame({
            "season_id": list(range(1, 11)),
            "trend_pattern_id": ["" for _ in range(10)],
        })
        df2.to_csv(trend_path, index=False)
        print("Wrote", trend_path, "(fill trend_pattern_id for seasons 1–10).")

if __name__ == "__main__":
    main()
