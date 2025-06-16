# XGBoost Churn Predictor

This project implements an XGBoost classifier to predict player churn, defined as "early churn / low CLV". The `xgboost_churn_predictor.py` script encapsulates the entire machine learning pipeline from data preparation and feature engineering to model training, evaluation, and serialization.

## Features

The `XGBoostChurnPredictor` class provides the following functionalities:

*   **Data Preparation**: Handles raw data by creating new features and processing categorical variables using one-hot encoding.
*   **Target Variable Creation**: Defines and creates the binary target variable for churn classification.
*   **Model Training**: Trains an XGBoost classifier, including handling class imbalance.
*   **Prediction**: Predicts churn probabilities for new data.
*   **Evaluation**: Evaluates the trained model using ROC-AUC score, classification report, and confusion matrix.
*   **Serialization**: Saves the trained model to a file and loads it for later use.

## Data Preparation

The `_prepare_data` method performs the following steps:

1.  **Feature Engineering (Durations)**:
    *   Calculates `reg_ftd_duration`: Duration in days between registration date (`reg_date`) and first-time deposit date (`ftd_date`).
    *   Calculates `reg_qp_duration`: Duration in days between registration date (`reg_date`) and qualification player date (`qp_date`).
2.  **Feature Engineering (Months Active)**:
    *   Calculates `months_active`: Number of active months for a player, referencing `ftd_date` or `reg_date` if `ftd_date` is missing. Clipped at a minimum of 0.
3.  **Handling Missing Values**:
    *   Fills missing monetary values (`total_handle`, `total_deposit`, `total_ngr`) with 0.
    *   Fills missing duration values (`reg_ftd_duration`, `reg_qp_duration`) with -1.
4.  **Categorical Feature Encoding**:
    *   Applies one-hot encoding to `brand_id` and `player_reg_product` columns.
    *   Ensures consistency in columns between training and inference by aligning one-hot encoded features.

## Target Variable

The binary target variable, `early_churn_low_clv`, is created by the `_create_target_variable` method.
A player is considered to have churned (target = 1) if:
*   `total_handle` < 100
*   AND `months_active` < 3

## Model Training

The `train` method orchestrates the model training process:

1.  Prepares the training data using `_prepare_data`.
2.  Creates the target variable using `_create_target_variable`.
3.  Defines the feature set, including base numerical features and one-hot encoded categorical features.
4.  Calculates `scale_pos_weight` to handle class imbalance in the target variable and sets this parameter for the XGBoost model.
5.  Trains the `xgb.XGBoostClassifier` model.

## Evaluation

The `evaluate` method assesses the model's performance on a test dataset:

1.  Prepares the test data.
2.  Creates the true target variable for the test set.
3.  Predicts churn probabilities for the test data.
4.  Converts probabilities to binary predictions based on a specified threshold (default is 0.5).
5.  Calculates and prints:
    *   ROC-AUC Score
    *   Classification Report (precision, recall, F1-score for each class)
    *   Confusion Matrix

## Usage

### Prerequisites

*   Python 3.x
*   Required libraries (install using `pip install -r requirements.txt` from the root directory):
    *   `pandas`
    *   `openpyxl` (for reading Excel files)
    *   `scikit-learn`
    *   `xgboost`
    *   `joblib`
    *   `pyyaml` (though not directly used in `xgboost_churn_predictor.py`, it's in `requirements.txt`)

### Running the Script

The script `xgboost_churn_predictor.py` can be run directly from the `src` directory:

```bash
python xgboost_churn_predictor.py
```

This will:
1.  Load data from `../data/sample_data__technical_assessment.xlsx`.
2.  Split the data into training and testing sets.
3.  Instantiate the `XGBoostChurnPredictor` (with example hyperparameters `n_estimators=150`, `max_depth=4`).
4.  Train the model on the training data.
5.  Evaluate the model on the test data.
6.  Save the trained model to `../models/xgboost_churn_model.joblib`.
7.  Load the saved model.
8.  Make predictions on a sample of 5 users from the test set and print the results.

### Training a New Model

To train a new model with your data:

```python
from xgboost_churn_predictor import XGBoostChurnPredictor
import pandas as pd

# Load your training data into a pandas DataFrame
# Ensure date columns ('activity_month', 'reg_date', 'ftd_date', 'qp_date') are parsed as datetime objects
# df_train = pd.read_csv("your_training_data.csv", parse_dates=['activity_month', 'reg_date', 'ftd_date', 'qp_date'])
# Example with provided data structure:
# df = pd.read_excel("path_to_your_data.xlsx", sheet_name="Sheet1")
# date_cols = ["activity_month", "reg_date", "ftd_date", "qp_date"]
# for col in date_cols:
#     df[col] = pd.to_datetime(df[col], errors="coerce")
# df_train, _ = train_test_split(df, test_size=0.2, random_state=42) # Assuming you split your data

# Instantiate the predictor with desired XGBoost hyperparameters
predictor = XGBoostChurnPredictor(n_estimators=100, max_depth=5, learning_rate=0.1)

# Train the model
predictor.train(df_train)

# Save the model
predictor.save("../models/my_custom_churn_model.joblib")
```

### Loading a Pre-trained Model and Making Predictions

```python
from xgboost_churn_predictor import XGBoostChurnPredictor
import pandas as pd

# Load a pre-trained model
loaded_predictor = XGBoostChurnPredictor.load("../models/xgboost_churn_model.joblib") # Or your custom model path

# Load new data for inference (ensure it has the same structure and date columns are parsed)
# df_inference = pd.read_csv("your_inference_data.csv", parse_dates=['activity_month', 'reg_date', 'ftd_date', 'qp_date'])
# Example with provided data structure:
# sample_data = df_test.head() # Using a sample from test data for demonstration

# Make predictions (returns probabilities)
# churn_probabilities = loaded_predictor.predict_proba(df_inference)
# print(churn_probabilities)
```

## Serialization

The model can be saved and loaded using `joblib`:

*   **Save Model**: The `save(filepath: str)` method serializes the `XGBoostChurnPredictor` instance.
    ```python
    predictor.save("../models/xgboost_churn_model.joblib")
    ```
*   **Load Model**: The static method `load(filepath: str)` deserializes the model.
    ```python
    loaded_predictor = XGBoostChurnPredictor.load("../models/xgboost_churn_model.joblib")
    ```

## Demonstration

The `if __name__ == "__main__":` block in `xgboost_churn_predictor.py` provides a demonstration of the typical workflow: loading data, splitting, training, evaluating, saving, loading, and predicting on a sample.
This serves as a practical example of how to use the `XGBoostChurnPredictor` class.
