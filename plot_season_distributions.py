from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FEATURE_PATH = Path("/content/drive/MyDrive/Trend_artifacts/season_features.csv")
TREND_LABELS = Path("/content/drive/MyDrive/Data_labels/season_trend_labels.csv")
ARTIFACTS = Path("/content/drive/MyDrive/Trend_artifacts")

CLASS_NAMES = ["stripes", "floral", "polka", "geometric", "solid", "gingham"]

def main():
    feats = pd.read_csv(FEATURE_PATH)
    trends = pd.read_csv(TREND_LABELS)

    df = feats.merge(trends, on="season_id", how="left")

    pattern_cols = [f"h_{k}" for k in range(len(CLASS_NAMES))]

    num_seasons = df["season_id"].nunique()
    fig, axes = plt.subplots(num_seasons, 1, figsize=(8, 2.5 * num_seasons), sharex=True)

    if num_seasons == 1:
        axes = [axes]

    for i, (season_id, row) in enumerate(df.sort_values("season_id").iterrows()):
        ax = axes[i]
        h = [row[c] for c in pattern_cols]
        ax.bar(CLASS_NAMES, h)
        ax.set_ylim(0, 1.0)
        title = f"Season {int(row['season_id'])}"

        if not pd.isna(row.get("trend_pattern_id", np.nan)):
            trend_id = int(row["trend_pattern_id"])
            title += f"  (trend label: {CLASS_NAMES[trend_id]})"
        else:
            title += "  (Season 11 – no label)"

        ax.set_title(title)
        ax.set_ylabel("Share")

    plt.xticks(rotation=45)
    plt.tight_layout()
    out_path = ARTIFACTS / "season_pattern_distributions.png"
    plt.savefig(out_path, dpi=200)
    print("Saved season pattern distribution plot to", out_path)


if __name__ == "__main__":
    main()
