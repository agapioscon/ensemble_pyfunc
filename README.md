# Adaptive Text Classifier (Dual-Context Model)

This repository contains a Python project for building and deploying a text classification model that can adapt to different deployment environments:

*   **Docker:** Models are loaded directly from disk (simulating a containerized environment).
*   **Databricks/MLflow Serving:** Models are loaded from the MLflow Model Registry or a specific MLflow run.

The model is an ensemble of five TF-IDF vectorizers and five Keras DNN models, trained on the NLTK movie reviews dataset for sentiment analysis (positive/negative classification).

## Repository Structure
'''
ensemble_pyfunc/
├── requirements.txt # Project dependencies
└── src/
└── models/
├── init.py # Makes models a package
├── ensemble_model.py # PyFunc model definition (MovieReviewClassifier)
├── run_model_mlflow.py # Script to load and run the model from MLflow
├── train_model.py # Training, logging (to MLflow), and PyFunc registration
└── predict_from_disk.py # load and infrence from disk
└── predict_with_pyfunc_local.py # load and infrence from pyfunc model
'''
*   **`ensemble_model.py`:**  Defines the `MovieReviewClassifier` class, which is a custom MLflow PyFunc model.  This class handles loading the individual TF-IDF and DNN models from either a local directory (for Docker) or from MLflow artifacts (for Databricks).  It also implements the `predict` method for making predictions.

*   **`train_model.py`:** This script performs the following:
    1.  Downloads and prepares the movie reviews dataset from NLTK.
    2.  Splits the data into training and testing sets (and further splits the testing set for the ensemble).
    3.  Trains five TF-IDF vectorizers and five Keras DNN models.
    4.  Logs the trained models (as individual artifacts) to MLflow.
    5.  Logs the combined `MovieReviewClassifier` as a PyFunc model to MLflow, referencing the individual model artifacts.
    6.  Registers the PyFunc model in the MLflow Model Registry.
    7. Saves the models to disk.

*   **`run_model_mlflow.py`:** This script demonstrates how to load and use the registered PyFunc model from MLflow, *either* from the Model Registry (recommended for production) *or* from a specific MLflow run (useful for testing).

*   **`predict_from_disk.py`:** This script loads the models directly from a specified directory on disk and makes predictions *without* using any MLflow loading functionality.  This simulates deploying the model in an environment where you have the model files directly available (e.g., a Docker container).

*   **`predict_with_pyfunc_local.py`:** This script loads the PyFunc model, models and vectorizers are loaded from disk, using the *Pyfunc* model class.

*   **`requirements.txt`:** Specifies the Python package dependencies for the project.

## Setup and Usage

1.  **Clone the Repository:**

    ```bash
    git clone <your_repository_url>
    cd <your_repository_name>
    ```

2.  **Create a Virtual Environment (Highly Recommended):**

    *   **Using `venv`:**
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate  # Linux/macOS
        .venv\Scripts\activate    # Windows
        ```

    *   **Using `conda`:**
        ```bash
        conda create -n text-classifier python=3.10
        conda activate text-classifier
        ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Train and Register the Model:**

    ```bash
    python src/models/train_model.py
    ```
    This will:
    *   Download the movie reviews dataset.
    *   Train the TF-IDF vectorizers and Keras models.
    *   Log the individual models and the combined PyFunc model to MLflow.
    *  Save the models in `trained_models` directory.
    *   Register the PyFunc model in the MLflow Model Registry as "movie_review_classifier" (you can change this name in the script).
    *   You can view the training run and the registered model in the MLflow UI (run `mlflow ui` in your terminal).

5.  **Run Predictions:**

    *   **From MLflow Model Registry (Recommended):**
        ```bash
        python src/models/run_model_mlflow.py
        ```
       *Edit the `run_model_mlflow.py` with the corect `run_id`.*
        This script loads the registered model from MLflow and makes predictions on example input data.  It demonstrates loading both from the Model Registry (using the model name and stage) and from a specific run ID.

    *  **Loading Models from Disk (for Docker or other non-MLflow deployments):**
      ```bash
        python src/models/predict_from_disk.py
        ```
       This script demonstrates how to load models from disk.

    *  **Loading Pyfunc Models from Disk (for Docker or other non-MLflow deployments):**
      ```bash
        python src/models/predict_with_pyfunc_local.py
        ```
       This script demonstrates how to load models and use the pyfunc model from disk.

## Key Features

*   **Dual-Context Loading:** The `MovieReviewClassifier` PyFunc model can load its components (TF-IDF vectorizers and Keras models) from either:
    *   A local directory: This is intended for Docker deployments, where the model files are packaged within the container.
    *   MLflow artifacts: This is for deployments using the MLflow Model Registry (e.g., on Databricks or using MLflow Serving).

*   **Ensemble Model:** The model combines multiple TF-IDF and DNN models for potentially improved performance and robustness.

*   **MLflow Integration:** The project uses MLflow for:
    *   Tracking training runs (parameters, metrics, artifacts).
    *   Logging and registering the combined model as a PyFunc model.
    *   Loading the model for serving (in the Databricks/MLflow scenario).

*   **Reproducibility:** The `requirements.txt` file and the use of a virtual environment ensure that the project's dependencies are clearly defined and can be easily reproduced.

*   **Flexibility:** The code provides examples of loading and using the model in different ways, making it adaptable to various deployment scenarios.

* **Handles Sparse Matrices** Correctly handles the sparse matrices, ensuring the indices of the sparse matrix are sorted.

## Further Development

*   **Hyperparameter Tuning:**  Experiment with different hyperparameters for the TF-IDF vectorizers and Keras models (e.g., using MLflow's hyperparameter tuning capabilities).
*   **Different Ensemble Methods:** Explore different ways of combining the predictions from the individual models (e.g., weighted averaging, stacking).
*   **Data Augmentation:** Consider using data augmentation techniques to increase the size and diversity of the training data.
*   **Different Datasets:**  Adapt the code to work with other text classification datasets.
*   **Deployment:**  Deploy the model as a REST API using MLflow Serving (either locally or on a platform like Databricks, AWS SageMaker, Azure ML, etc.).
*  **Add Tests**: Implement unit tests and integration test.
