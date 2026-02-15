This repository is created as a submission artifact for ML Assignment 2. This repository is also linked to the streamlit cloud for deployment of various classification models performance comparison.

# ML Classification Models – Fetal Health Classification

## Problem Statement
Fetal health classification aims to classify fetal health in 3 categories as **normal(1), suspect(2), or pathological(3)** fetal states from **cardiotocography (CTG)** measurements. This would help medical professionals detect risk at early period and prioritize cure.

Deploy multiple machine learning classification models on a given dataset. Develop an interactive application using the streamlit library to view the results, test data, and evaluation metrics. Deploy the application on the streamlit community cloud. Provide the links to access the deployed application.

## Dataset Description
- Data Source        : Fetal Health Classification Dataset
- Number of Instances: >2000
- Features           : >20 numeric features, cardiotocography
- Task               : Multiclass classification - 'fetal_health' in {1,2,3}
- Data Split         : 80-20 train-test split with stratification

## Models Used and Metrics
Six models trained on the same dataset using consistent preprocessing steps:
| ML Model Name        | Accuracy   | AUC       | Precision | Recall    | F1        | MCC       |
|----------------------|------------|-----------|-----------|-----------|-----------|-----------|
| XGBoost              | 0.93896714 | 0.98102147| 0.93700067| 0.93896714| 0.93709917| 0.82887644|
| RandomForest         | 0.92488263 | 0.97927623| 0.92164375| 0.92488263| 0.92170391| 0.78748866|
| DecisionTree         | 0.90140845 | 0.85800072| 0.89831599| 0.90140845| 0.89941113| 0.72516674|
| LogisticRegression   | 0.88497653 | 0.96139452| 0.88929313| 0.88497653| 0.88547066| 0.68343920|
| KNN                  | 0.87089202 | 0.93932048| 0.86147318| 0.87089202| 0.86290544| 0.61909073|
| GaussianNB           | 0.80985915 | 0.87590070| 0.86105199| 0.80985915| 0.82536721| 0.57368060|

### Observations (Model Performance)
| ML Model Name        |            Observation about model performance                       |
|----------------------|----------------------------------------------------------------------|
| XGBoost              | Best performing model in terms of overall metrics. It performed the best in terms of accuracy, auc_score, f1_score, and mcc_score|
| RandomForest         | Very strong performance and almost as close as XGBoost|
| DecisionTree         | Simple model design but heavier computations, resulting in slowing down model & reduced AUC/MCC|
| LogisticRegression   | Competitive linear model, good AUC but lower MCC than decision tree|
| KNN                  | Moderate performance, but slightly underperforms compared to Logistic/Tree|
| GaussianNB           | Lowest accuracy due to independence assumptions in the features|

A comparison table is generated to `model/artifacts/metrics_summary.csv` and also shown in the app under "All Models".

## Repository Structure
```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
|-- Data/
|   |-- feature_names.json
|   |-- fetal_health.csv
|   |-- test_sample.csv
│-- model/
│   │-- train_models.py
│   │-- __init__.py
│   └-- artifacts/  # created after training; contains models in .joblib form and metrics_summary.csv
```

## How to Run (Locally)
1) Create a Python environment and install dependencies:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Train models and generate artifacts + sample test CSV:
```powershell
python model/train_models.py
```

3) Launch the Streamlit app:
```powershell
streamlit run app.py
```

Open the URL shown (typically http://localhost:8501).

## Streamlit App Features
- Dataset upload option (CSV) – Free tier note: upload only test data.
- Sample dataset download option.
- Model selection dropdown – choose one model or "All Models" or any number of models for comparison.
- Display of evaluation metrics – Accuracy, AUC, Precision, Recall, F1, MCC when ground truth is provided.
- Confusion matrix and classification report for selected model.
- Sample data toggle – quickly try using the prepared test split.
- Per-class accuracy of each model.
- Preview top misclassifications done by each model.
- Option to download the predictions made by each model as .zip file.

CSV expectations:
- Columns must include exactly the model feature columns. If `fetal_health` column is present (1/2/3), the app computes metrics; otherwise it shows predictions only.

## Reproducibility
- Deterministic `random_state=42` for splits and applicable models.
- Preprocessing and model parameters are encapsulated in saved pipelines under `model/artifacts/*.joblib`.

## Acknowledgements
- UCI Machine Learning Repository – Fetal health dataset
