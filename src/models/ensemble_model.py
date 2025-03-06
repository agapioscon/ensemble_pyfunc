import mlflow
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
import os
import pandas as pd
from scipy.sparse import isspmatrix_csr, csr_matrix

class MovieReviewClassifier(mlflow.pyfunc.PythonModel):

    def __init__(self, local_model_dir=None):
        self.local_model_dir = local_model_dir
        self.vectorizers: list[TfidfVectorizer] = []
        self.dnn_models: list[tf.keras.Model] = []

    def load_context(self, context):
        if self.local_model_dir:
            # Load from local directory (Docker scenario)
            print(f"Loading models from local directory: {self.local_model_dir}")
            for i in range(5):
                vectorizer_path = os.path.join(self.local_model_dir, f"tfidf_vectorizer_{i}.pkl")
                with open(vectorizer_path, "rb") as f:
                    self.vectorizers.append(pickle.load(f))

                dnn_model_path = os.path.join(self.local_model_dir, f"dnn_model_{i}.h5")
                self.dnn_models.append(tf.keras.models.load_model(dnn_model_path))
        else:
            # Load from MLflow artifacts (Databricks/MLflow serving scenario)
            print("Loading models from MLflow artifacts.")
            for i in range(5):
                vectorizer_path = context.artifacts[f"tfidf_vectorizer_{i}"]
                with open(vectorizer_path, "rb") as f:
                    self.vectorizers.append(pickle.load(f))

                dnn_model_path = context.artifacts[f"dnn_model_{i}"]
                self.dnn_models.append(tf.keras.models.load_model(dnn_model_path))

    def predict(self, context, model_input):
        """
        Makes predictions.  Accepts a list of strings or a pandas DataFrame.
        """
        # Input handling (accepts list of strings or pandas DataFrame)
        if isinstance(model_input, pd.DataFrame):
            if model_input.shape[1] != 1:
                raise ValueError("DataFrame input must have exactly one column containing text.")
            texts = model_input.iloc[:, 0].tolist()  # Extract text from the DataFrame
        elif isinstance(model_input, list):
            texts = model_input
        else:
            raise TypeError("model_input must be a list of strings or a pandas DataFrame.")

        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All inputs must be strings.")

        if len(texts) == 0:
            return []

        all_predictions = []
        for i in range(5):
            transformed_text = self.vectorizers[i].transform(texts)

            # --- SOLUTION: Ensure Sorted Indices ---
            if isspmatrix_csr(transformed_text):
                if not transformed_text.has_sorted_indices:
                   transformed_text.sort_indices()
            else:
                transformed_text = csr_matrix(transformed_text)
                if not transformed_text.has_sorted_indices:
                   transformed_text.sort_indices()

            predictions = self.dnn_models[i].predict(transformed_text)
            all_predictions.append(predictions)
        final_predictions = np.mean(all_predictions, axis=0)
        return final_predictions.tolist()