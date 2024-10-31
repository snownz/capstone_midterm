
# README: Model Evaluation Notebook

This notebook is designed to evaluate various machine learning models on a processed dataset using specific configurations. The objective is to assess the models' performance across different metrics (Accuracy, Precision, Recall, F1) to determine the most suitable model for identifying high-risk cases. The following instructions will guide you through using this notebook effectively.

---

## Prerequisites

Ensure you have the following libraries installed:
- `numpy`
- `pandas`
- `pickle`
- `scikit-learn`
- `torch`
- `torch-optimizer`
- `tqdm`
- `xgboost`

You can install any missing packages with:
```bash
pip install numpy pandas pickle scikit-learn torch torch-optimizer tqdm xgboost
```

---

## Notebook Structure

The notebook is structured as follows:
1. **Data Loading and Preprocessing**: Load the dataset, preprocess it, and split it into training and test sets using custom data processing functions.
2. **Model Training and Prediction**: Train a selected model and make predictions.
3. **Evaluation Metrics**: Calculate and display metrics (Accuracy, Precision, Recall, F1) for model evaluation.
4. **Risk Analysis**: Analyze the number of high-risk cases predicted as low risk and vice versa.

---

## How to Use This Notebook

### Step 1: Load Required Libraries and Modules
Run the following code to import the necessary libraries and the custom functions:

```python
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from midterm_lib import DataProcessing, HibriModel
import warnings
warnings.filterwarnings("ignore")
```

### Step 2: Define the Scenario and Load Data
Specify the scenario and load the dataset:

```python
scenario_name = 'scenario_2'
dataset_file = f'./{scenario_name}/risk-train-processed.csv'
dataset = pd.read_csv(dataset_file)
dataset.replace('?', None, inplace=True)
```

### Step 3: Preprocess Data
Initialize data processing and split the data into training and testing sets:

```python
data_processing = DataProcessing(scenario_name, dataset, one_hot_encoding=True)
(id_train, id_test), x_train, x_test, y_train, y_test = data_processing.get_train_test(test_size=0.2, target='CLASS')
```

### Step 4: Initialize and Train Model
Choose a model and make predictions. Available models include:
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- XGBoost
- Support Vector Classifier
- Ensemble

```python
model = HibriModel(scenario_name)
df_pred = model.predict((id_test, x_test[0], x_test[1]), selected_model='Ensemble')
```

### Step 5: Map and Evaluate Predictions
Map predictions to the actual target, convert classes, and print evaluation metrics:

```python
df_pred['target'] = df_pred['ORDER_ID'].map(dataset.set_index('ORDER_ID')['CLASS'])
df_pred['target'] = df_pred['target'].apply(lambda x: 1 if x == 'yes' else 0)
df_pred['predicted_class'] = df_pred['predicted'].apply(lambda x: round(x))

print(f'Accuracy: {accuracy_score(df_pred["target"], df_pred["predicted_class"])}')
print(f'Precision: {precision_score(df_pred["target"], df_pred["predicted_class"])}')
print(f'Recall: {recall_score(df_pred["target"], df_pred["predicted_class"])}')
print(f'F1: {f1_score(df_pred["target"], df_pred["predicted_class"])}')

print(classification_report(df_pred["target"], df_pred["predicted_class"]))
```

### Step 6: Perform Risk Analysis
Analyze the number of high-risk cases predicted as low risk and vice versa:

```python
print(f"Num High Risk predicted as Low Risk: {len(df_pred[(df_pred['predicted_class'] == 0) & (df_pred['target'] == 1)])}")
print(f"Num High Risk predicted as High Risk: {len(df_pred[(df_pred['predicted_class'] == 1) & (df_pred['target'] == 1)])}\n")

print(f"Num Low Risk predicted as Low Risk: {len(df_pred[(df_pred['predicted_class'] == 0) & (df_pred['target'] == 0)])}")
print(f"Num Low Risk predicted as High Risk: {len(df_pred[(df_pred['predicted_class'] == 1) & (df_pred['target'] == 0)])}\n")
```

### Step 7: Interpret Results
Use the results from the evaluation metrics and risk analysis to interpret the model's effectiveness in classifying high-risk cases accurately.

---

## Conclusion

This notebook provides a comprehensive approach to evaluate various machine learning models on a risk-based classification dataset. Follow the steps carefully to load, preprocess, train, and evaluate models. Use the results summary and risk analysis to select the best model for deployment.

For further assistance or questions, please refer to the comments within each cell in the notebook.