"""
BBC Text Categorization using TF-IDF Vectorizer + Multinomial Naive Bayes
========================================================================
This script trains a MultinomialNB classifier with TfidfVectorizer on the BBC
News dataset, prints comprehensive evaluation metrics on the test set, and provides
an interactive prompt to classify any new user-entered text.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics as sklm


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "bbc-text.csv")


def load_and_preprocess_data(data_path=DATA_PATH):
    """Load BBC news dataset and remove duplicate records."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Please ensure bbc-text.csv is in the data/ folder."
        )

    df = pd.read_csv(data_path)
    initial_count = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[*] Loaded dataset: {initial_count} total articles ({len(df)} unique after deduplication).")
    print(f"[*] Categories: {sorted(df['category'].unique())}")
    return df


def train_model(x_train, y_train):
    """
    Build and train a scikit-learn Pipeline combining:
      1. TfidfVectorizer (unigram + bigram, max 3000 features, English stop words)
      2. MultinomialNB (Laplace smoothing alpha=1.0)
    """
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                analyzer="word",
                stop_words="english",
                max_df=0.75,
                max_features=3000,
                ngram_range=(1, 2),
                sublinear_tf=False,
            ),
        ),
        ("clf", MultinomialNB(alpha=1.0)),
    ])

    print("[*] Training TF-IDF + MultinomialNB model...")
    pipeline.fit(x_train, y_train)
    return pipeline


def evaluate_model(pipeline, x_test, y_test, labels):
    """Evaluate and display model metrics on the test set."""
    y_pred = pipeline.predict(x_test)
    accuracy = pipeline.score(x_test, y_test)

    print("\n" + "=" * 60)
    print("                 MODEL EVALUATION METRICS")
    print("=" * 60)
    print(f"Test Set Accuracy: {accuracy * 100:.2f}%\n")

    print("-" * 60)
    print("Detailed Classification Report:")
    print("-" * 60)
    print(sklm.classification_report(y_test, y_pred, labels=labels, digits=4))

    print("-" * 60)
    print("Confusion Matrix:")
    print("-" * 60)
    cm = sklm.confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Actual: {lbl}" for lbl in labels],
                             columns=[f"Pred: {lbl}" for lbl in labels])
    print(cm_df.to_string())
    print("=" * 60 + "\n")


def predict_category(pipeline, text, categories):
    """Predict category and class probabilities for a single text input."""
    cleaned_text = text.strip()
    if not cleaned_text:
        return None, None

    prediction = pipeline.predict([cleaned_text])[0]
    probabilities = pipeline.predict_proba([cleaned_text])[0]

    # Map class name to predicted probability
    prob_dict = {
        cls: prob for cls, prob in zip(pipeline.classes_, probabilities)
    }
    # Sort by highest confidence
    prob_dict = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

    return prediction, prob_dict


def interactive_session(pipeline, categories):
    """Run an interactive CLI session where user can test multiple inputs."""
    print("=" * 60)
    print("                INTERACTIVE PREDICTION MODE")
    print("=" * 60)
    print("Type or paste any text/headline to classify.")
    print("Available categories: business, entertainment, politics, sport, tech")
    print("Type 'q' or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("\nEnter text > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

        if user_input.lower() in ("q", "exit", "quit"):
            print("Exiting interactive mode. Goodbye!")
            break

        if not user_input:
            print("Please enter non-empty text.")
            continue

        predicted_class, probs = predict_category(pipeline, user_input, categories)
        confidence = probs[predicted_class] * 100

        print("\n--- Prediction Result ---")
        print(f"Predicted Category : \033[1;32m{predicted_class.upper()}\033[0m (Confidence: {confidence:.2f}%)")
        print("Probability Breakdown:")
        for cat, prob in probs.items():
            bar = "#" * int(prob * 30)
            print(f"  - {cat:<14}: {prob * 100:6.2f}%  |{bar}")


def main():
    parser = argparse.ArgumentParser(
        description="BBC Text Categorization with TF-IDF and MultinomialNB"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DATA_PATH,
        help="Path to bbc-text.csv dataset file",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=500,
        help="Number of samples to reserve for test set (default: 500)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Optional: Single text input to classify immediately via command line",
    )
    args = parser.parse_args()

    # 1. Load data
    df = load_and_preprocess_data(args.data)
    categories = sorted(df["category"].unique())

    # 2. Train-test split (500 test samples matching the original notebook)
    X = df["text"].to_numpy()
    y = df["category"].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # 3. Train Pipeline
    pipeline = train_model(x_train, y_train)

    # 4. Evaluate metrics
    evaluate_model(pipeline, x_test, y_test, categories)

    # 5. Predict single argument if provided, otherwise launch interactive loop
    if args.text:
        pred, probs = predict_category(pipeline, args.text, categories)
        print(f"\nInput: \"{args.text}\"")
        print(f"Predicted Category: {pred.upper()} ({probs[pred] * 100:.2f}%)")
    else:
        interactive_session(pipeline, categories)


if __name__ == "__main__":
    main()
