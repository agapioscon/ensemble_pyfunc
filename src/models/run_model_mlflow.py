import mlflow
import pandas as pd

def load_and_predict_from_registry(model_name, stage="None", input_data=None):
    """
    Loads a model from the MLflow Model Registry and makes predictions.

    Args:
        model_name: The name of the registered model (e.g., "movie_review_classifier").
        stage: The stage of the model to load (e.g., "Production", "Staging", "None").
               Defaults to "None", which loads the latest version.
        input_data:  The input data (list of strings or pandas DataFrame).

    Returns:
        A list of predictions.
    """
    model_uri = f"models:/{model_name}/{stage}"
    print(f"Loading model from URI: {model_uri}")
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    if input_data is None:
       raise ValueError("input data cannot be None")
    predictions = loaded_model.predict(input_data)
    return predictions

def load_and_predict_from_run(run_id, input_data=None):
    """
    Loads a model from a specific MLflow run and makes predictions.

    Args:
        run_id: The ID of the MLflow run.
        input_data: The input data (list of strings or pandas DataFrame).

    Returns:
        A list of predictions.
    """
    model_uri = f"runs:/{run_id}/combined_model"
    print(f"Loading model from URI: {model_uri}")
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    if input_data is None:
       raise ValueError("input data cannot be None")
    predictions = loaded_model.predict(input_data)
    return predictions

if __name__ == "__main__":
    # --- Example Usage ---

    # 1. Loading from the Model Registry (Recommended)
    model_name = "the_model"  # Replace with your model name
    input_texts = [
        "This movie was absolutely fantastic!  The acting was superb.",
        "I hated this film.  It was a complete waste of time.",
        "The plot was interesting, but the pacing was a bit slow.",
        "Best movie ever!",
    ]
    # Using a list of strings:
    predictions_registry_list = load_and_predict_from_registry(model_name, stage="None", input_data=input_texts)
    print(f"Predictions (Registry, list input): {predictions_registry_list}")