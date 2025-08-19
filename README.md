
# Documentation: Ovarian Cancer Stage Prediction Project

## 1. Project Overview

This project aims to build a machine learning classifier to predict whether a patient's ovarian cancer is early stage (Stage I-II) or late stage (Stage III-IV). The prediction is based on clinical, pathology, exposure, family history, and follow-up data from the TCGA-OV dataset.

The project involves the following key steps:

1.  **Data Loading and Merging:** Loading and merging five different data files into a single dataset.
2.  **Data Cleaning and Preprocessing:** Handling missing values, encoding categorical features, and selecting a relevant set of features for modeling.
3.  **Model Training and Evaluation:** Training and evaluating two machine learning models (Logistic Regression and Random Forest) to predict the cancer stage.
4.  **Analysis and Reporting:** Analyzing the model results, identifying key predictive features, and summarizing the findings in a report.

## 2. Project Structure and File Descriptions

This project is organized into a series of Python scripts that perform specific tasks. The scripts are designed to be run in a specific order to ensure the correct workflow. Here is a description of the files in this project:

### Data Files

*   `Data - clinical.project-tcga-ov.2025-08-09/`: This directory contains the raw data files provided for the assignment.
    *   `clinical.tsv`: Demographic and clinical variables.
    *   `pathology_detail.tsv`: Tumor and histological details.
    *   `exposure.tsv`: Lifestyle/environmental risk factors.
    *   `family_history.tsv`: Family cancer history.
    *   `follow_up.tsv`: Follow-up clinical outcomes.
*   `merged_data.csv`: A merged dataset containing data from all five raw data files.
*   `data_with_target.csv`: The merged dataset with the addition of the binary target variable (`target`).
*   `processed_data.csv`: The final, cleaned, and preprocessed dataset used for model training.
*   `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`: The split data for training and testing the models.

### Scripts

Here is a description of the Python scripts created to perform the analysis. The scripts are numbered in the order they should be run.

1.  **`load_data_3.py`**: This script loads the five raw TSV files, merges them into a single dataframe, and saves the merged dataframe as `merged_data.csv`.
2.  **`define_target_2.py`**: This script loads the `merged_data.csv` file, creates the binary target variable (`target`) based on the `diagnoses.figo_stage` column, and saves the resulting dataframe as `data_with_target.csv`.
3.  **`feature_selection_5.py`**: This script performs feature selection, handles missing values, and encodes categorical features. It takes the `data_with_target.csv` file as input and produces the `processed_data.csv` file.
4.  **`split_data.py`**: This script splits the `processed_data.csv` file into training and testing sets and saves them as `X_train.csv`, `X_test.csv`, `y_train.csv`, and `y_test.csv`.
5.  **`train_models.py`**: This script trains the Logistic Regression and Random Forest models on the training data and saves the trained models as `logistic_regression_model.joblib` and `random_forest_model.joblib`.
6.  **`evaluate_models.py`**: This script evaluates the trained models on the test data, computes various performance metrics, and generates plots for the confusion matrix and ROC curve.
7.  **`final_analysis_3.py`**: This script performs the final analysis, including identifying the most important features from the Random Forest model and generating EDA plots to explore clinical patterns.

### Model Files

*   `logistic_regression_model.joblib`: The saved Logistic Regression model.
*   `random_forest_model.joblib`: The saved Random Forest model.

### Output Files

*   `report.txt`: A text file containing the final report summarizing the project's findings.
*   `confusion_matrices.png`: A plot of the confusion matrices for both models.
*   `roc_curve.png`: A plot of the ROC curves for both models.
*   `feature_importance.png`: A plot of the top 15 most important features from the Random Forest model.
*   `eda_plots.png`: A plot of the exploratory data analysis (EDA) for the top features.

## 3. How to Run the Project

To reproduce the results of this project, you need to run the Python scripts in the following order:

```bash
python3 load_data_3.py
python3 define_target_2.py
python3 feature_selection_5.py
python3 split_data.py
python3 train_models.py
python3 evaluate_models.py
python3 final_analysis_3.py
```

After running all the scripts, the output files (plots, models, and the report) will be generated in the project directory.
