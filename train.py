"""
train.py
--------
Trains a Multinomial Naive Bayes text classifier that categorizes
customer support tickets into: Billing, Technical, HR, General.

Pipeline:
1. Load dataset.csv
2. Clean text (lowercase, remove punctuation, remove extra spaces)
3. Convert text to TF-IDF features
4. Split into train/test (80/20)
5. Train Multinomial Naive Bayes
6. Evaluate (accuracy, precision, recall, f1, classification report, confusion matrix)
7. Save model.pkl and vectorizer.pkl
8. Save confusion matrix plot for the README / screenshots

Run:
    python train.py
"""

import os
import re
import string

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_PATH = "dataset.csv"
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
CONFUSION_MATRIX_PATH = "confusion_matrix.png"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def clean_text(text: str) -> str:
    """
    Clean raw ticket text for vectorization.

    Steps:
        - Lowercase everything
        - Remove punctuation
        - Collapse multiple spaces into one
        - Strip leading/trailing whitespace

    Args:
        text: Raw input string.

    Returns:
        Cleaned string. Returns an empty string if input is not a string.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the ticket dataset from a CSV file and validate its structure.

    Args:
        path: Path to dataset.csv

    Returns:
        A pandas DataFrame with 'ticket' and 'category' columns.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If required columns are missing or the file is empty.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Please make sure dataset.csv "
            "is present in the project folder."
        )

    df = pd.read_csv(path)

    required_columns = {"ticket", "category"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns {required_columns}, "
            f"found {set(df.columns)} instead."
        )

    if df.empty:
        raise ValueError("Dataset is empty. Please provide valid data.")

    # Drop rows with missing values in required columns
    df = df.dropna(subset=["ticket", "category"]).reset_index(drop=True)

    return df


def train_model():
    """
    Full training pipeline: load data, clean text, vectorize, train,
    evaluate, and persist the model + vectorizer to disk.
    """
    print("=" * 60)
    print("AUTO EMAIL / TICKET CATEGORIZER - TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load dataset
    print("\n[1/8] Loading dataset...")
    df = load_dataset(DATASET_PATH)
    print(f"   -> Loaded {len(df)} rows.")
    print(f"   -> Categories: {sorted(df['category'].unique())}")

    # 2. Clean text
    print("\n[2/8] Cleaning text (lowercase, punctuation & whitespace removal)...")
    df["cleaned_ticket"] = df["ticket"].apply(clean_text)

    # Remove rows that became empty after cleaning
    df = df[df["cleaned_ticket"].str.len() > 0].reset_index(drop=True)

    # 3. TF-IDF vectorization
    print("\n[3/8] Converting text to TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 1),
    )
    X = vectorizer.fit_transform(df["cleaned_ticket"])
    y = df["category"]
    print(f"   -> TF-IDF matrix shape: {X.shape}")

    # 4. Train/test split (80/20)
    print("\n[4/8] Splitting data into train (80%) and test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   -> Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # 5. Train Multinomial Naive Bayes
    print("\n[5/8] Training Multinomial Naive Bayes classifier...")
    # A small alpha (Laplace smoothing) sharpens predicted probabilities,
    # which gives more meaningful confidence scores on a small dataset.
    model = MultinomialNB(alpha=0.05)
    model.fit(X_train, y_train)

    # 6. Evaluate
    print("\n[6/8] Evaluating model performance...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n   Accuracy : {accuracy * 100:.2f}%")
    print(f"   Precision: {precision * 100:.2f}%")
    print(f"   Recall   : {recall * 100:.2f}%")
    print(f"   F1-score : {f1 * 100:.2f}%")

    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("   Confusion Matrix:")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    # Save confusion matrix plot (useful for README screenshots)
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        plt.title("Confusion Matrix - Ticket Categorizer")
        plt.tight_layout()
        plt.savefig(CONFUSION_MATRIX_PATH)
        plt.close(fig)
        print(f"\n   -> Confusion matrix plot saved to '{CONFUSION_MATRIX_PATH}'")
    except Exception as exc:  # pragma: no cover - plotting is non-critical
        print(f"   -> Warning: could not save confusion matrix plot ({exc})")

    # 7 & 8. Save model and vectorizer
    print("\n[7/8] Saving trained model to 'model.pkl'...")
    joblib.dump(model, MODEL_PATH)

    print("[8/8] Saving TF-IDF vectorizer to 'vectorizer.pkl'...")
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("\n" + "=" * 60)
    print("Training complete! Model and vectorizer saved successfully.")
    print("Run 'streamlit run app.py' to launch the web app.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        train_model()
    except (FileNotFoundError, ValueError) as err:
        print(f"\n[ERROR] {err}")
    except Exception as err:  # pragma: no cover
        print(f"\n[UNEXPECTED ERROR] {err}")
