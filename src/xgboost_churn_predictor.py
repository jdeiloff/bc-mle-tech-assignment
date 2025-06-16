import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


class XGBoostChurnPredictor:
    """
    A class to train, evaluate, and use an XGBoost Classifier to predict
    player churn, defined as 'early churn / low CLV'.

    This class encapsulates the entire pipeline from data preparation to
    model serialization, making it suitable for production environments.
    """

    def __init__(self, **xgb_params):
        """
        Initializes the XGBoostChurnPredictor.

        Args:
            **xgb_params: Hyperparameters to be passed to the XGBoostClassifier.
                          e.g., n_estimators, max_depth, learning_rate.
        """
        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "use_label_encoder": False,
            "random_state": 42,
        }
        # Override defaults with any user-provided params
        final_params = {**default_params, **xgb_params}

        self.model = xgb.XGBClassifier(**final_params)
        self._features = None
        self._categorical_cols = ["brand_id", "player_reg_product"]
        self._one_hot_encoded_cols = None

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the raw data by creating features and handling categorical variables.
        This method is designed to be applied consistently during training and inference.
        """
        proc_df = df.copy()

        # 1. Feature Engineering: Calculate durations
        proc_df["reg_ftd_duration"] = (
            proc_df["ftd_date"] - proc_df["reg_date"]
        ).dt.days
        proc_df["reg_qp_duration"] = (proc_df["qp_date"] - proc_df["reg_date"]).dt.days

        # 2. Feature Engineering: Calculate months_active (now a feature)
        reference_date = proc_df["ftd_date"].fillna(proc_df["reg_date"])
        months_active = (
            proc_df["activity_month"].dt.year - reference_date.dt.year
        ) * 12 + (proc_df["activity_month"].dt.month - reference_date.dt.month)
        proc_df["months_active"] = months_active.clip(lower=0)

        # 3. Handle missing monetary/duration values
        proc_df["total_handle"].fillna(0, inplace=True)
        proc_df["total_deposit"].fillna(0, inplace=True)
        proc_df["total_ngr"].fillna(0, inplace=True)
        proc_df["reg_ftd_duration"].fillna(-1, inplace=True)
        proc_df["reg_qp_duration"].fillna(-1, inplace=True)

        # 4. Handle Categorical Features with One-Hot Encoding
        proc_df = pd.get_dummies(
            proc_df, columns=self._categorical_cols, drop_first=True, dtype=float
        )

        # During inference, align columns with the training set
        if self._one_hot_encoded_cols is not None:
            missing_cols = set(self._one_hot_encoded_cols) - set(proc_df.columns)
            for c in missing_cols:
                proc_df[c] = 0
            # Ensure the order is the same and drop columns not seen in training
            proc_df = proc_df[self._one_hot_encoded_cols]

        return proc_df

    def _create_target_variable(self, df: pd.DataFrame) -> pd.Series:
        """
        Creates the binary target variable for churn classification.

        Target definition: 'early_churn_low_clv'
        Event (churn=1) = total_handle < 100 AND months_active < 3
        """
        churn_target = ((df["total_handle"] < 100) & (df["months_active"] < 3)).astype(
            int
        )
        return churn_target

    def train(self, df_train: pd.DataFrame):
        """
        Trains the XGBoost model on the provided dataframe.

        Args:
            df_train (pd.DataFrame): The training data.
        """
        print("Preparing training data...")
        df_prepared = self._prepare_data(df_train)

        print("Creating target variable...")
        y_train = self._create_target_variable(df_prepared)

        # Define features to be used in the model
        base_features = [
            "total_deposit",
            "total_handle",
            "total_ngr",
            "reg_ftd_duration",
            "reg_qp_duration",
            "months_active",
        ]
        ohe_features = [
            c
            for c in df_prepared.columns
            if any(cat_col in c for cat_col in self._categorical_cols)
        ]
        self._features = base_features + ohe_features
        self._one_hot_encoded_cols = (
            df_prepared.columns
        )  # Store all columns for consistency

        X_train = df_prepared[self._features]

        # Handle class imbalance
        scale_pos_weight = y_train.value_counts().get(
            0, 1
        ) / y_train.value_counts().get(1, 1)
        self.model.set_params(scale_pos_weight=scale_pos_weight)

        print(f"Training XGBoost model with scale_pos_weight={scale_pos_weight:.2f}...")
        self.model.fit(X_train, y_train)
        print("Model training complete.")

    def predict_proba(self, df_inference: pd.DataFrame) -> np.ndarray:
        """
        Predicts the churn probability for new data.

        Returns:
            np.ndarray: An array of probabilities for the positive class (churn).
        """
        if self._features is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        df_prepared = self._prepare_data(df_inference)
        X_inference = df_prepared[self._features]

        # Return probability of the positive class (class 1)
        churn_probabilities = self.model.predict_proba(X_inference)[:, 1]
        return churn_probabilities

    def evaluate(self, df_test: pd.DataFrame, threshold: float = 0.5):
        """
        Evaluates the model on test data using classification metrics.
        """
        print("\n--- Model Evaluation on Test Set ---")
        if self._features is None:
            raise RuntimeError("Model has not been trained yet.")

        df_prepared = self._prepare_data(df_test)
        y_true = self._create_target_variable(df_prepared)

        churn_probabilities = self.predict_proba(df_test)
        y_pred = (churn_probabilities > threshold).astype(int)

        # ROC-AUC Score
        auc_score = roc_auc_score(y_true, churn_probabilities)
        print(f"ROC-AUC Score: {auc_score:.4f}")

        # Classification Report
        print("\nClassification Report:")
        print(
            classification_report(y_true, y_pred, target_names=["Not Churn", "Churn"])
        )

        # Confusion Matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

        return {
            "roc_auc": auc_score,
            "report": classification_report(y_true, y_pred, output_dict=True),
        }

    def save(self, filepath: str):
        """Saves the trained model artifact to a file."""
        print(f"\nSaving model to {filepath}...")
        joblib.dump(self, filepath)
        print("Model saved.")

    @staticmethod
    def load(filepath: str):
        """Loads a model artifact from a file."""
        print(f"Loading model from {filepath}...")
        model_instance = joblib.load(filepath)
        print("Model loaded.")
        return model_instance


if __name__ == "__main__":
    print("--- XGBoost Churn Predictor Demonstration ---")
    # 1. Load data
    try:
        df = pd.read_excel(
            "./data/sample_data__technical_assessment.xlsx", sheet_name="Sheet1"
        )
        date_cols = ["activity_month", "reg_date", "ftd_date", "qp_date"]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    except FileNotFoundError:
        print("Error: 'sample_data__technical_assessment.xlsx' not found.")
        exit()

    # 2. Split data
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # 3. Instantiate and train the model
    # We can pass XGBoost specific hyperparameters here
    predictor = XGBoostChurnPredictor(n_estimators=150, max_depth=4)
    predictor.train(df_train)

    # 4. Evaluate the model
    predictor.evaluate(df_test)

    # 5. Save the model artifact
    model_path = "./models/xgboost_churn_model.joblib"
    predictor.save(model_path)

    # 6. Load the model and make a prediction on a sample
    loaded_predictor = XGBoostChurnPredictor.load(model_path)

    sample_data = df_test.head(5)
    churn_probs = loaded_predictor.predict_proba(sample_data)

    print("\n--- Prediction on a sample of 5 users ---")
    print("Sample User Account IDs:", sample_data["account_id"].values)
    print("Predicted Churn Probabilities:", np.round(churn_probs, 3))
