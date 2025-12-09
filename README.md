VISION TRANSFORMER - BASED SEASONAL FASHION TREND PREDICTION
Using DeiT + Season-Level Analytics for Automated Fashion Forecasting

**Overview**

This project builds a fully automated pipeline that predicts seasonal fashion trends using only raw clothing images. It combines:

     Data-Efficient Vision Transformer (DeiT) for image-level pattern classification

     Season-level aggregation + Logistic Regression for trend forecasting

This system is designed to be accurate, interpretable, and scalable, making it ideal for fashion brands, retail analytics teams, and e-commerce platforms.


**Key Features**

DeiT-based pattern recognition (Stripes, Floral, Polka, Geometric, Solid, Gingham)
Season signature vectors for interpretable forecasting
Trend prediction model using logistic regression
Automated dataset builder for season-wise distribution
Visualization tools for patterns, confusion matrices, and seasonal analytics
Works effectively even with small datasets


System Architecture
1. Dataset Builder (build_dataset.py)

Organizes raw images into 11 seasons
Generates metadata files:
     img_labels.csv (image + pattern + season)
     season_trend_labels.csv (trend labels template)

2. Pattern Classifier (DeiT)

Model: deit_small_patch16_224
Heavy augmentation (RandAugment, Random Erasing)
Output:
     Predicted class
     Softmax probability vector (6D)

3. Season-Level Feature Extraction (extract_features.py)

Averages softmax probabilities of all images per season

Produces a 6-dimensional season signature vector
