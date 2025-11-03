# SmartStreet — Demand Prediction Project

**Author:** Vrinda Sharma  
**Last updated:** 2025-11-03  
**License:** MIT

---

## Project overview

SmartStreet is a lightweight machine learning pipeline that predicts street-food demand levels (High / Medium / Low) using dataset features such as preparation time, cook time, ingredient count and categorical attributes (diet, flavor profile, course, region). The project trains several classical models plus a small feedforward neural network and provides simple EDA visualizations and accuracy comparisons.

---

## Features

- Data cleaning and basic feature engineering (`total_time`, `num_ingredients`)
- Label encoding and standard scaling for numeric features
- Model training and evaluation for:
  - Logistic Regression
  - Decision Tree
  - Gaussian Naive Bayes
  - SVM
  - ANN (Keras)
- EDA visualizations: demand distribution, correlation heatmap, boxplots, scatter plots
- Accuracy comparison and confusion matrix visualizations

---

## Files

- `minorproject4.py` — main script that runs the full pipeline (EDA, model training, and evaluation).
- `indian_food.csv` — dataset required by the script (not included; place it in the same folder).
- `LICENSE` — MIT license (already added).
- `README.md` — this file.

---

## Requirements

Tested with Python 3.8+. Main Python packages used:

