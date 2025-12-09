#**VISION TRANSFORMER - BASED SEASONAL FASHION TREND PREDICTION**

Using DeiT + Season-Level Analytics for Automated Fashion Forecasting

**OVERVIEW**

This project builds a fully automated pipeline that predicts seasonal fashion trends using only raw clothing images. It combines:

> Data-Efficient Vision Transformer (DeiT) for image-level pattern classification

> Season-level aggregation + Logistic Regression for trend forecasting

This system is designed to be accurate, interpretable, and scalable, making it ideal for fashion brands, retail analytics teams, and e-commerce platforms.


##**KEY FEATURES**

DeiT-based pattern recognition (Stripes, Floral, Polka, Geometric, Solid, Gingham)
Season signature vectors for interpretable forecasting
Trend prediction model using logistic regression
Automated dataset builder for season-wise distribution
Visualization tools for patterns, confusion matrices, and seasonal analytics
Works effectively even with small datasets


##**SYSTEM ARCHITECTURE**

1. Dataset Builder (build_dataset.py)

     >Organizes raw images into 11 seasons

     >Generates metadata files:
            > img_labels.csv (image + pattern + season)
            > season_trend_labels.csv (trend labels template)

2. Pattern Classifier (DeiT)

Model: deit_small_patch16_224

Heavy augmentation (RandAugment, Random Erasing)

Output:

       > Predicted class
       > Softmax probability vector (6D)

3. Season-Level Feature Extraction (extract_features.py)

Averages softmax probabilities of all images per season

Produces a 6-dimensional season signature vector

          h = [h0, h1, h2, h3, h4, h5]


4. Trend Prediction (train_trend_model.py)

Multinomial Logistic Regression

Predicts trend for Season 11


##**PROJECT STRUCTURE**

     ├── data/
     │   ├── raw_images/
     │   ├── seasons/
     │   ├── img_labels.csv
     │   ├── season_trend_labels.csv
     │   └── season_features.csv
     │
     ├── src/
     │   ├── build_dataset.py
     │   ├── train_pattern_model.py
     │   ├── extract_features.py
     │   ├── train_trend_model.py
     │   ├── visualize_results.py
     │   └── utils/
     │
     ├── models/
     │   └── deit_best_model.pth
     │
     ├── results/
     │   ├── confusion_matrix.png
     │   ├── season_distribution.png
     │   └── predictions.csv
     │
     └── README.md


##**RESULTS**

Pattern Classifier (DeiT)
  > Accuracy: ~87%
 
  > Strong generalization across all 6 pattern categories

  > Attention maps show correct focus on textures & fabric regions

Season Trend Forecasting - Season 11 Prediction:
  > Predicted Trend: Polka

  > Most confident pattern in signature
> 



###**INSTALLATION**

     git clone <repo-url>
     cd fashion-trend-prediction
     pip install -r requirements.txt


     
###**USAGE**
1.Build dataset
          
          python src/build_dataset.py

2.Train pattern Classification Model
         
          python src/train_pattern_model.py

3.Extract season level features
          
          python src/extract_features.py

4.Train trend prediction model
         
          python src/train_trend_model.py

5.visualize results
         
          python src/visualize_results.py




###**USE CASES**

Design teams: Detect emerging patterns early

Retail planning: Forecast demand for patterned products

E-commerce tagging: Automated categorization

Market analytics: Season-by-season pattern distribution insights



###**LIMITATIONS**

Performance depends on pattern variety and image quality



###**FUTURE WORK**

Add temporal modeling (Transformers/TCN for seasons)

Real-time forecasting dashboard

Multi-attribute prediction (color, style, texture)

Multilingual/global fashion trend analysis



###**TEAM MEMBERS**

Usha S Vuchidi

Saanvi Joginipally

Renusri Periketi


**Acknowledgment**

AI tools were used to support drafting and debugging, but all analysis, decisions, and conclusions are original work by the team.
























