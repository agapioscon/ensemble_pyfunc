import mlflow
import tensorflow as tf
from ensemble_model import MovieReviewClassifier  # Import the PyFunc model class
# ... (Import your other necessary modules: train_and_log_models, prepare_data, etc.) ...
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
import os
import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
import pandas as pd


def prepare_data(num_splits=5):
    """
    Downloads and prepares the movie reviews dataset for training.

    Args:
        num_splits: The number of separate datasets to create (in our case, 5).

    Returns:
        A tuple: (texts, labels), where:
            - texts: A list of lists of text data (5 lists).
            - labels: A list of lists of corresponding labels (5 lists).
    """

    nltk.download('movie_reviews', quiet=True) # Downloads only if not exist
    nltk.download('punkt', quiet=True)

    # Load the movie reviews data
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]

    # Shuffle the data
    import random
    random.shuffle(documents)

    # Separate texts and labels
    texts_all, labels_all = zip(*documents)
    texts_all = [" ".join(words) for words in texts_all]  # Join words into sentences
    labels_all = [1 if label == 'pos' else 0 for label in labels_all] # Convert to numerical (1=positive, 0=negative)

    # Split data into training and a combined test/validation set.
    train_texts, test_val_texts, train_labels, test_val_labels = train_test_split(
        texts_all, labels_all, test_size=0.4, random_state=42, stratify=labels_all
    )  # 60% train, 40% test/val

    # Further split the test/val set into 5 separate, equal-sized sets.
    texts = []
    labels = []
    # Use train_test_split repeatedly with decreasing test_size
    remaining_texts = test_val_texts
    remaining_labels = test_val_labels
    for i in range(num_splits - 1):
      split_texts, remaining_texts, split_labels, remaining_labels = train_test_split(
        remaining_texts, remaining_labels, test_size=(1 - 1/(num_splits-i)), random_state=42, stratify=remaining_labels
        )
      texts.append(split_texts)
      labels.append(split_labels)
    texts.append(remaining_texts)
    labels.append(remaining_labels)

    # Add the training set
    for _ in range(num_splits):
      texts.append(train_texts)
      labels.append(train_labels)

    return texts[:5], labels[:5]

def train_and_log_models(texts, labels, run_name=None):

    if len(texts) != 5 or len(labels) != 5:
        raise ValueError("texts and labels must be lists of length 5")

    for i in range(len(texts)):
        if not all(isinstance(text, str) for text in texts[i]):
            raise ValueError(f"All inputs in texts[{i}] must be strings.")
        if len(texts[i]) != len(labels[i]):
            raise ValueError(f"texts[{i}] and labels[{i}] must have the same length")


    with mlflow.start_run(run_name=run_name) as run:
        for i in range(5):
            # --- 1. Train TF-IDF Vectorizer ---
            vectorizer = TfidfVectorizer(stop_words='english', min_df=2)  # Add stop word removal and min_df
            vectorizer.fit(texts[i])

            # Save the vectorizer locally (temporary, for logging)
            vectorizer_path = f"tfidf_vectorizer_{i}.pkl"
            with open(vectorizer_path, "wb") as f:
                pickle.dump(vectorizer, f)
            mlflow.log_artifact(vectorizer_path)  # Log to MLflow
            os.remove(vectorizer_path)  # Clean up the temporary file

            # --- 2. Train Keras DNN Model ---
            transformed_texts = vectorizer.transform(texts[i]).toarray()  # To dense array
            dnn_model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(transformed_texts.shape[1],)),
                tf.keras.layers.Dropout(0.5),  # Add dropout for regularization
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.5),  # Add dropout
                tf.keras.layers.Dense(1, activation='sigmoid')  # Output for binary classification
            ])
            dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            dnn_model.fit(transformed_texts, np.array(labels[i]), epochs=10, batch_size=32, validation_split=0.2) # Added validation split

            # Save the model locally (temporary)
            dnn_model_path = f"dnn_model_{i}.h5"
            dnn_model.save(dnn_model_path)
            mlflow.log_artifact(dnn_model_path)  # Log to MLflow
            os.remove(dnn_model_path)  # Clean up

            #mlflow.log_param("model_number", i)

        # --- 3. Log parameters, metrics (optional, but good practice) ---
        mlflow.log_param("num_models", 5)  # Example parameter
        # Log any overall metrics you might have, e.g., average accuracy, etc.

        print(f"Models logged with Run ID: {run.info.run_id}")
        return run

def log_pyfunc_model(run, model_name="movie_review_classifier"):
    """
    Logs the PyFunc model to MLflow, referencing the artifacts from the training run.
    """

    artifact_uri = run.info.artifact_uri
    print(artifact_uri)
    artifacts = {
        f"tfidf_vectorizer_{i}":  f"{artifact_uri}/tfidf_vectorizer_{i}.pkl"
        for i in range(5)
    }
    artifacts.update({
        f"dnn_model_{i}": f"{artifact_uri}/dnn_model_{i}.h5"
        for i in range(5)
    })
    print(artifacts)
    with mlflow.start_run(run_name="model") as run:
        mlflow.pyfunc.log_model(
        python_model=MovieReviewClassifier(),  # Use the imported class
        artifact_path="combined_model",
        artifacts=artifacts,
        registered_model_name="the_model"
        )
    

if __name__ == "__main__":
    texts, labels = prepare_data()
    run = train_and_log_models(texts, labels, run_name="Movie Reviews Training")
    log_pyfunc_model(run, model_name="movie_review_classifier")
