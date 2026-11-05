"""
BBC Text Categorization: Model Comparison
=========================================
Compares the three core Naive Bayes & Vectorization strategies from the initial project:
  1. Binary CountVectorizer + MultinomialNB
  2. TF-IDF Vectorizer + GaussianNB
  3. TF-IDF Vectorizer + MultinomialNB

Evaluates all three side-by-side on the BBC news dataset (5 categories).
"""

import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics as sklm

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "bbc-text.csv")


def load_data(data_path=DATA_PATH):
    """Load BBC dataset and remove duplicates."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df = pd.read_csv(data_path)
    initial_len = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[*] Loaded dataset: {initial_len} rows ({len(df)} unique after deduplication)")
    return df


def build_models():
    """
    Define the three models compared in the project:
      1. Binary CountVectorizer + MultinomialNB
      2. TF-IDF Vectorizer + GaussianNB (requires dense conversion)
      3. TF-IDF Vectorizer + MultinomialNB
    """
    models = {
        "1. Binary CountVectorizer + MultinomialNB": Pipeline([
            ("vectorizer", CountVectorizer(
                binary=True,
                lowercase=True,
                stop_words="english",
                max_df=0.75,
                max_features=3000,
                ngram_range=(1, 2),
            )),
            ("clf", MultinomialNB(alpha=1.0)),
        ]),

        "2. TF-IDF Vectorizer + GaussianNB": Pipeline([
            ("vectorizer", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_df=0.75,
                max_features=3000,
                ngram_range=(1, 2),
            )),
            # GaussianNB requires dense arrays rather than sparse matrices
            ("todense", FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
            ("clf", GaussianNB()),
        ]),

        "3. TF-IDF Vectorizer + MultinomialNB": Pipeline([
            ("vectorizer", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_df=0.75,
                max_features=3000,
                ngram_range=(1, 2),
            )),
            ("clf", MultinomialNB(alpha=1.0)),
        ]),
    }
    return models


def evaluate_all(models, x_train, y_train, x_test, y_test, categories):
    """Train each model, record train time, accuracy, macro & weighted metrics."""
    summary_rows = []
    detailed_reports = {}

    print("\n" + "=" * 75)
    print("                 TRAINING & EVALUATION BENCHMARK")
    print("=" * 75)

    for name, pipeline in models.items():
        print(f"\n--> Training: {name} ...", end=" ", flush=True)
        start_time = time.time()
        pipeline.fit(x_train, y_train)
        train_time = time.time() - start_time

        # Predictions
        y_pred = pipeline.predict(x_test)
        acc = pipeline.score(x_test, y_test)

        # Metrics
        p_macro, r_macro, f1_macro, _ = sklm.precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        p_wt, r_wt, f1_wt, _ = sklm.precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )

        summary_rows.append({
            "Model Pipeline": name,
            "Accuracy": f"{acc * 100:.2f}%",
            "Macro F1": f"{f1_macro * 100:.2f}%",
            "Weighted F1": f"{f1_wt * 100:.2f}%",
            "Macro Precision": f"{p_macro * 100:.2f}%",
            "Macro Recall": f"{r_macro * 100:.2f}%",
            "Fit Time (s)": f"{train_time:.3f}s",
        })

        detailed_reports[name] = {
            "report": sklm.classification_report(y_test, y_pred, labels=categories, digits=4),
            "cm": sklm.confusion_matrix(y_test, y_pred, labels=categories),
        }
        print(f"Done! (Acc: {acc * 100:.2f}%, Time: {train_time:.2f}s)")

    # 1. Print Summary Comparison Table
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 85)
    print("                         SUMMARY COMPARISON TABLE")
    print("=" * 85)
    print(summary_df.to_string(index=False))
    print("=" * 85)

    # 2. Print Detailed Classification Reports
    for name, details in detailed_reports.items():
        print(f"\n" + "-" * 75)
        print(f"Detailed Classification Report: {name}")
        print("-" * 75)
        print(details["report"])

        cm_df = pd.DataFrame(
            details["cm"],
            index=[f"Actual: {c}" for c in categories],
            columns=[f"Pred: {c}" for c in categories],
        )
        print("Confusion Matrix:")
        print(cm_df.to_string())

    return summary_df


def main():
    # 1. Load data
    df = load_data()
    categories = sorted(df["category"].unique())
    X = df["text"].to_numpy()
    y = df["category"].to_numpy()

    # 2. Train-test split (500 test samples matching the original notebook)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=500, random_state=42, stratify=y
    )
    print(f"[*] Train set size: {len(x_train)} | Test set size: {len(x_test)}")
    print(f"[*] Categories ({len(categories)}): {categories}")

    # 3. Build & compare models
    models = build_models()
    evaluate_all(models, x_train, y_train, x_test, y_test, categories)


if __name__ == "__main__":
    main()
