# Introvert vs Extrovert (IVE)

A data science project exploring personality prediction and segmentation using classical machine learning. The included Jupyter notebook analyzes features from a tabular dataset and builds models to classify individuals as Introvert vs Extrovert, with additional exploratory clustering and visualization work.

This repository is structured and documented for employers and collaborators to quickly understand the problem, approach, and how to reproduce the results.

## Overview
- Problem: Predict personality type (Introvert vs Extrovert) from tabular features; explore structure via clustering and dimensionality reduction.
- Techniques: Data cleaning/imputation, encoding/scaling, model training and evaluation, PCA for visualization, optional unsupervised clustering.
- Models featured: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, and XGBoost.
- Outputs: Evaluation metrics (accuracy, classification report, confusion matrices) and illustrative plots (matplotlib, seaborn, plotly).

## Repository Contents
- `IntrovertVsExtrovert.ipynb` — Main analysis notebook with EDA, preprocessing, modeling, evaluation, and submission generation.
- `README.md` — Project overview, setup, and usage instructions.
- `requirements.txt` — Python dependencies for reproducibility.

## Data
The notebook expects the following files in the project root:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

These files are not included in the repository. If you are reproducing the analysis, place these CSVs in the repository root. They appear to originate from a Kaggle-style dataset (based on `sample_submission.csv` usage).

## Environment Setup
Recommended: use a virtual environment (venv or conda).

Using Python venv:
1. `python -m venv .venv`
2. Activate: Windows PowerShell: `.\.venv\Scripts\Activate.ps1`; macOS/Linux: `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. (Optional) `pip install jupyterlab`

## Running the Notebook
1. Ensure `train.csv`, `test.csv`, and `sample_submission.csv` are present in the project root.
2. Launch Jupyter:
   - `jupyter lab` or `jupyter notebook`
3. Open `IntrovertVsExtrovert.ipynb` and run cells top-to-bottom.

## Approach Summary
- EDA: Inspect distributions, correlations, and potential data issues.
- Preprocessing: Impute missing values, encode categoricals, and scale numeric features.
- Modeling: Evaluate several classifiers (LogReg, Tree/Forest/GBM, SVM, KNN, XGBoost) using `train_test_split` and standard metrics.
- Model Selection: Compare results via accuracy and classification reports; inspect confusion matrices.
- Visualization: PCA for dimensionality reduction; 2D/3D plots using matplotlib/seaborn/plotly; optional KMeans clustering.
- Submission: Generate and save predicted labels matching the expected submission format.

## Notes on Reproducibility
- Random states are set where applicable in the notebook to support reproducibility.
- Results can vary with library versions; see `requirements.txt` to align your environment.

## Contact
Author: Zhi Zheng  
GitHub: https://github.com/ZhiZheng0889

If you have questions or would like to discuss the work, feel free to open an issue or reach out.
