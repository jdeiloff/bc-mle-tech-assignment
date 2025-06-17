# BetterCollective MLE Tech Assignment

Solution proposed by [Jonathan Deiloff](https://github.com/jdeiloff).

## Repository Structure

This repository is organized to address the Senior ML Engineer take-home technical assessment. Below is a description of the key directories and files:

*   **`README.md`**: (This file) Provides an overview of the project and its structure.
*   **`requirements.txt`**: Lists the Python dependencies required to run the code in this project. Install them using `pip install -r requirements.txt`.
*   **`config/`**: This directory is intended for configuration files. (Currently empty).
*   **`data/`**: Contains the dataset used for the assessment.
    *   `sample_data__technical_assessment.xlsx`: The primary dataset for EDA, model training, and evaluation.
*   **`docs/`**: Contains supplementary documentation.
    *   `section_3_production-comments.md`: Contains answers to Section 3 of the technical assessment, focusing on model evaluation, alternatives, production, and scaling.
    *   `Senior ML Engineer Take-Home.pdf`: The original PDF document outlining the technical assessment.
*   **`models/`**: This directory is intended for storing trained model artifacts. The `xgboost_churn_predictor.py` script saves its output here (e.g., `xgboost_churn_model.joblib`).
*   **`notebooks/`**: Contains Jupyter notebooks used for exploration and initial model development.
    *   `EDA-Modelling.ipynb`: This notebook includes the Exploratory Data Analysis (EDA), initial model training, and scoring. It also contains the answers to Sections 1 and 2 of the technical assessment.
*   **`src/`**: Contains the source code for the project.
    *   `xgboost_churn_predictor.py`: The primary Python script for training, evaluating, and serializing the churn prediction model. This is the expected script to generate the final model artifact.
    *   `README.md`: Provides detailed documentation specifically for the `xgboost_churn_predictor.py` script, including usage instructions.
    *   `pyproject.toml` and `uv.lock`: Files related to Python project packaging and dependency management using [`uv`](https://docs.astral.sh/uv/).
*   **`tests/`**: This directory is intended for test scripts. (Currently empty).

## Assessment Sections

*   **Sections 1 & 2**: Answered within the `notebooks/EDA-Modelling.ipynb` notebook. This includes EDA and initial model development.
*   **Section 3**: Answered in the `docs/section_3_production-comments.md` file, covering production-related considerations.
*   **Model Artifact Generation**: The script `src/xgboost_churn_predictor.py` is the designated script for generating the final model artifact.