import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

FEATURE_PATH = Path("/content/drive/MyDrive/Trend_artifacts/season_features.csv")
TREND_LABELS = Path("/content/drive/MyDrive/Data_labels/season_trend_labels.csv")

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

    # Drop unwanted columns
    drop_cols = ["season_id", "trend_pattern_id"]
    X = train_df.drop(columns=drop_cols, errors="ignore")
    y = train_df["trend_pattern_id"].astype(int)

    X_test = test_df.drop(columns=drop_cols, errors="ignore")

    # LOOCV (Leave-One-Out Cross-Validation)
    preds = []
    trues = []

    for leave_out in range(1, 11):   # seasons 1..10
        train_fold = train_df[train_df["season_id"] != leave_out]
        test_fold  = train_df[train_df["season_id"] == leave_out]

        X_tr = train_fold.drop(columns=drop_cols, errors="ignore")
        y_tr = train_fold["trend_pattern_id"].astype(int)
        X_te = test_fold.drop(columns=drop_cols, errors="ignore")
        y_te = test_fold["trend_pattern_id"].astype(int)

        clf = LogisticRegression(max_iter=500, multi_class="multinomial")
        clf.fit(X_tr, y_tr)

        pred = clf.predict(X_te)[0]
        preds.append(pred)
        trues.append(int(y_te.values[0]))

    # Report LOOCV performance
    print("=== LOOCV RESULTS (Seasons 1–10) ===")
    print("Accuracy:", accuracy_score(trues, preds))
    print(classification_report(trues, preds))

    # Train final model on all seasons 1–10
    clf_final = LogisticRegression(max_iter=500, multi_class="multinomial")
    clf_final.fit(X, y)

    # Predict Season 11's dominant trend
    if len(X_test) == 1:
        pred11 = clf_final.predict(X_test)[0]
        proba = clf_final.predict_proba(X_test)[0]

        print("\n=== FINAL PREDICTION FOR SEASON 11 ===")
        print("Predicted trend class:", pred11)
        print("Class probabilities:", np.round(proba, 3))
    else:
        print("ERROR: Season 11 features missing!")

if __name__ == "__main__":
    main()
