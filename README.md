# In-Depth Analysis of Ovarian Cancer Stage Prediction

This project provides a detailed walkthrough of building a machine learning classifier to predict the stage of ovarian cancer. The goal is to distinguish between early-stage (I-II) and late-stage (III-IV) cancer using a rich dataset from TCGA-OV. This document elaborates on the methodology and the reasoning behind each step in the `Cancer_Prediction.ipynb` notebook.

## Methodological Breakdown

The core logic of the project is to transform raw, multi-modal clinical data into a clean, structured format suitable for machine learning, and then to build, evaluate, and interpret predictive models.

### 1. Data Consolidation and Preparation

**The Challenge**: The patient data is fragmented across five separate files (`clinical.tsv`, `pathology_detail.tsv`, etc.), each describing a different aspect of the patient's case. To build a holistic model, we need a single, unified view of each patient.

**Our Approach**: 
- We performed a series of **outer merges** using `pandas`. An outer merge was specifically chosen to ensure that all information for every patient (`cases.case_id`) was retained, even if a patient's record was not present in all five files. This prevents data loss.
- During the merge, we identified and removed redundant columns (like `project.project_id`) that were common across files to avoid duplicated data and keep the resulting dataset clean.

### 2. Defining the Clinical Outcome (Target Variable)

**The Challenge**: The raw data contains a detailed cancer stage (`diagnoses.figo_stage`) with many sub-stages (e.g., 'Stage IA', 'Stage IIIC'). For a classification model, we need a clear, binary outcome.

**Our Approach**:
- We engineered a new **target variable** to represent our clinical question: "Is the cancer early-stage or late-stage?"
- We wrote a function to map the detailed FIGO stages into two categories:
  - **0 (Early Stage)**: Includes Stages I and II and all their sub-stages.
  - **1 (Late Stage)**: Includes Stages III and IV and all their sub-stages.
- Records with ambiguous or missing stage information (marked as `'--'`) were removed, as they could not be reliably classified and would add noise to the model.

### 3. Curating a High-Impact Feature Set

**The Challenge**: The merged dataset contains over 500 features, many of which are sparse, irrelevant, or redundant. Using all features would likely lead to poor model performance (due to the curse of dimensionality) and make the model difficult to interpret.

**Our Approach**:
- **Feature Selection**: We manually selected a small subset of clinically relevant features (e.g., `demographic.age_at_index`, `diagnoses.tumor_grade`). This is a domain-knowledge-driven approach to focus the model on variables that are most likely to be predictive.
- **Handling Missing Values**: Machine learning models cannot process missing data (`NaN`s). We imputed missing values as follows:
  - For **numerical features** (like age), we used the **median**. The median is more robust to outliers than the mean, making it a safer choice.
  - For **categorical features** (like race), we used the **mode** (the most frequent value).
- **Categorical Feature Encoding**: Models require all input to be numeric. We used **one-hot encoding** (`pd.get_dummies`) to convert categorical features into a numerical format. This method creates new binary (0/1) columns for each category, avoiding the incorrect assumption of an ordinal relationship between categories (e.g., that 'white' is "less than" 'asian'). We used `drop_first=True` to prevent multicollinearity among the newly created features.

### 4. Model Development and Validation Strategy

**The Challenge**: We need to train a model that not only performs well on the data it has seen but also generalizes to new, unseen patient data.

**Our Approach**:
- **Train-Test Split**: We split the data into a training set (80%) and a testing set (20%). The model is trained only on the training data, and its final performance is evaluated on the completely unseen test data.
- **Stratification**: When splitting, we used `stratify=y`. This ensures that the proportion of early-stage and late-stage cases is the same in both the training and testing sets. This is critical for imbalanced datasets, as it prevents a scenario where the test set accidentally contains a very different distribution of outcomes than the training set.
- **Choice of Models**: We trained two distinct models to compare their performance:
  - **Logistic Regression**: A simple, interpretable linear model that provides a good baseline.
  - **Random Forest**: A powerful, non-linear ensemble model that can capture complex interactions between features.
- **Cross-Validation**: Before final training, we used 5-fold cross-validation to get a more reliable estimate of how each model would perform on unseen data. This helps in tuning the model and ensuring its stability.

### 5. In-Depth Model Performance Evaluation

**The Challenge**: A single metric like accuracy can be misleading, especially for imbalanced datasets. We need a comprehensive set of metrics to understand the models' strengths and weaknesses.

**Our Approach**: We evaluated the models using:
- **Accuracy**: The overall percentage of correct predictions.
- **Precision**: Of the patients we predicted to have late-stage cancer, how many actually did? (Measures the cost of a false positive).
- **Recall (Sensitivity)**: Of all the patients who truly had late-stage cancer, how many did our model identify? (Measures the cost of a false negative, which is often critical in medicine).
- **F1-Score**: The harmonic mean of precision and recall, providing a single score that balances both concerns.
- **ROC-AUC**: Measures the model's ability to distinguish between early and late-stage cancer across all possible classification thresholds.

### 6. Uncovering Predictive Clinical Factors

**The Challenge**: A prediction is useful, but understanding *why* the prediction was made is crucial for clinical insight and trust.

**Our Approach**:
- **Feature Importance Analysis**: We extracted the feature importances from the Random Forest model. This analysis ranks the features based on their contribution to the model's predictive power.
- **Visualization**: We created plots to visualize these importances, highlighting the top clinical factors (e.g., `diagnoses.age_at_diagnosis`, `diagnoses.tumor_grade`) that drive the prediction of cancer stage.

## How to Run

1.  **Prerequisites**: Ensure you have Python 3 and the required libraries installed.
    ```bash
    pip install pandas numpy scikit-learn joblib matplotlib seaborn
    ```
2.  **Jupyter Notebook**: Open and run the `Cancer_Prediction.ipynb` notebook in a Jupyter environment.