# Moroccan Telecom Churn Prediction Pipeline Run

### Operational Decision Threshold Applied: **0.4**

| Model               |   ROC-AUC |   PR-AUC |   Accuracy |   Precision (Churn) |   Recall (Churn) |   F1-Score (Churn) |
|:--------------------|----------:|---------:|-----------:|--------------------:|-----------------:|-------------------:|
| Logistic Regression |    0.9695 |   0.8458 |     0.8921 |              0.5706 |           0.9789 |             0.7209 |
| Optimized XGBoost   |    0.9962 |   0.9845 |     0.9865 |              0.9479 |           0.9579 |             0.9529 |
| Optimized LightGBM  |    0.9945 |   0.9848 |     0.9835 |              0.92   |           0.9684 |             0.9436 |
| Optimized CatBoost  |    0.9972 |   0.9897 |     0.994  |              0.9892 |           0.9684 |             0.9787 |