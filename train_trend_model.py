%%writefile train_trend_model_logreg.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURE_PATH = Path("/content/drive/MyDrive/Trend_artifacts/season_features.csv")
TREND_LABELS = Path("/content/drive/MyDrive/Data_labels/season_trend_labels.csv")
ARTIFACTS = Path("/content/drive/MyDrive/Trend_artifacts")

CLASS_NAMES = ["stripes", "floral", "polka", "geometric", "solid", "gingham"]

def smooth_probs(raw_proba, alpha=0.5):
    """
    Smooth model probabilities by mixing them with a uniform prior:
        p' = (p + alpha/K) / (1 + alpha)
    Prevents extreme 0/1 probabilities.
    """
    K = len(raw_proba)
    return (raw_proba + alpha / K) / (1 + alpha)

def main():
    df_feat = pd.read_csv(FEATURE_PATH)
    df_labels = pd.read_csv(TREND_LABELS)

    # Merge labels + features
    df = df_feat.merge(df_labels, on="season_id", how="left")

    # Training = Seasons 1–10
    train_df = df[df["season_id"] <= 10].copy()

    # Testing = Season 11
    test_df = df[df["season_id"] == 11].copy()

    if len(test_df) != 1:
        print("ERROR: Season 11 row missing or duplicated.")
        print(test_df)
        return

    # Split features/targets
    drop_cols = ["season_id", "trend_pattern_id"]
    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df["trend_pattern_id"].astype(int)

    X_test = test_df.drop(columns=drop_cols, errors="ignore")

    # Logistic Regression + Scaling
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            multi_class="auto",
            max_iter=5000,    # ensure convergence
            C=1.0,            # moderate regularization
            random_state=42,
        ))
    ])

    # Train on seasons 1–10
    clf.fit(X_train, y_train)

    # Predict Season 11
    raw_proba = clf.predict_proba(X_test)[0]
    smoothed_proba = smooth_probs(raw_proba, alpha=0.5)
    pred_id = int(np.argmax(smoothed_proba))
    pred_name = CLASS_NAMES[pred_id]

    print("=== SEASON 11 TREND PREDICTION (Logistic Regression) ===")
    print("Raw probabilities:     ", np.round(raw_proba, 3))
    print("Smoothed probabilities:", np.round(smoothed_proba, 3))
    print(f"Predicted trend: {pred_id} ({pred_name})")

    # Save prediction
    season11_id = int(test_df["season_id"].iloc[0])
    out_df = pd.DataFrame({
        "season_id": [season11_id],
        "pred_trend_id": [pred_id],
        "pred_trend_name": [pred_name],
    })
    out_path = ARTIFACTS / "season11_trend_prediction_logreg.csv"
    out_df.to_csv(out_path, index=False)
    print("Saved Season 11 prediction to", out_path)

if __name__ == "__main__":
    main()
