%%writefile train_trend_model.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

FEATURE_PATH = Path("/content/drive/MyDrive/Trend_artifacts/season_features.csv")
TREND_LABELS = Path("/content/drive/MyDrive/Data_labels/season_trend_labels.csv")
ARTIFACTS = Path("/content/drive/MyDrive/Trend_artifacts")

CLASS_NAMES = ["stripes", "floral", "polka", "geometric", "solid", "gingham"]

def main():
    # Load season-level features
    df = pd.read_csv(FEATURE_PATH)

    # Load trend labels (ground truth for seasons 1–10)
    labels = pd.read_csv(TREND_LABELS)

    # Merge on season_id
    m = df.merge(labels, on="season_id", how="left")

    # Separate Season 11 for final prediction
    train_df = m[m["season_id"] <= 10].copy()
    test_df  = m[m["season_id"] == 11].copy()

    drop_cols = ["season_id", "trend_pattern_id"]
    X = train_df.drop(columns=drop_cols, errors="ignore")
    y = train_df["trend_pattern_id"].astype(int)

    X_test = test_df.drop(columns=drop_cols, errors="ignore")

    # --- LOOCV with Gaussian Naive Bayes ---
    preds = []
    trues = []
    season_ids = []

    for leave_out in range(1, 11):   # seasons 1..10
        train_fold = train_df[train_df["season_id"] != leave_out]
        test_fold  = train_df[train_df["season_id"] == leave_out]

        X_tr = train_fold.drop(columns=drop_cols, errors="ignore")
        y_tr = train_fold["trend_pattern_id"].astype(int)
        X_te = test_fold.drop(columns=drop_cols, errors="ignore")
        y_te = test_fold["trend_pattern_id"].astype(int)

        clf = GaussianNB()
        clf.fit(X_tr, y_tr)

        pred = int(clf.predict(X_te)[0])
        preds.append(pred)
        trues.append(int(y_te.values[0]))
        season_ids.append(int(test_fold["season_id"].iloc[0]))

    acc = accuracy_score(trues, preds)
    print("=== LOOCV RESULTS (Seasons 1–10, GaussianNB) ===")
    print(f"LOOCV accuracy: {acc:.3f}")

    loocv_df = pd.DataFrame({
        "season_id": season_ids,
        "true_trend_id": trues,
        "true_trend_name": [CLASS_NAMES[t] for t in trues],
        "pred_trend_id": preds,
        "pred_trend_name": [CLASS_NAMES[p] for p in preds],
    }).sort_values("season_id")
    print("\nLOOCV true vs predicted:")
    print(loocv_df)

    loocv_path = ARTIFACTS / "loocv_trend_predictions_nb.csv"
    loocv_df.to_csv(loocv_path, index=False)
    print("Saved LOOCV predictions to", loocv_path)

    # --- Train final NB model on all seasons 1–10 ---
    clf_final = GaussianNB()
    clf_final.fit(X, y)

    # Predict Season 11's dominant trend
    if len(X_test) == 1:
        proba = clf_final.predict_proba(X_test)[0]
        pred11 = int(np.argmax(proba))

        print("\n=== FINAL PREDICTION FOR SEASON 11 (GaussianNB) ===")
        print("Predicted trend class:", pred11, f"({CLASS_NAMES[pred11]})")
        print("Class probabilities:", np.round(proba, 3))

        # Save Season 11 prediction
        s11 = int(test_df["season_id"].iloc[0])
        s11_df = pd.DataFrame({
            "season_id": [s11],
            "pred_trend_id": [pred11],
            "pred_trend_name": [CLASS_NAMES[pred11]],
        })
        s11_path = ARTIFACTS / "season11_trend_prediction_nb.csv"
        s11_df.to_csv(s11_path, index=False)
        print("Saved Season 11 prediction to", s11_path)
    else:
        print("ERROR: Season 11 features missing!")

if __name__ == "__main__":
    main()
