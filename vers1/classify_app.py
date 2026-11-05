"""
BBC News Categorization - Native Desktop Application
===================================================
1. Prints model metrics and evaluation to the terminal at startup.
2. Launches a clean native desktop GUI for predicting text categories.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics as sklm

# Enable High-DPI scaling on Windows if available
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "bbc-text.csv")

CATEGORY_COLORS = {
    "sport": "#10b981",         # Emerald green
    "tech": "#3b82f6",          # Vivid blue
    "business": "#f59e0b",      # Warm amber
    "entertainment": "#c084fc", # Bright purple
    "politics": "#f87171",      # Coral red
    "default": "#94a3b8",       # Slate grey
}

SAMPLE_TEXTS = {
    "sport": "Arsenal secured a dramatic 3-2 victory over Chelsea in the London derby with an 89th-minute curling strike by their captain.",
    "tech": "Apple unveiled a new high-performance AI chip designed for MacBooks and smartphones, boasting a 40% reduction in power consumption.",
    "business": "Global stock markets rallied strongly as central banks indicated potential interest rate cuts following solid corporate earnings reports.",
    "entertainment": "The film festival concluded with a standing ovation as the director accepted the prestigious award for best drama picture.",
    "politics": "The Prime Minister faced heated debates in parliament regarding government budget cuts, tax reforms, and upcoming election strategies.",
}


def print_terminal_metrics():
    """Load dataset, train TF-IDF + MultinomialNB pipeline, and print metrics to terminal."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}!")

    print("\n" + "=" * 65)
    print("      BBC NEWS CATEGORIZATION: MODEL TRAINING & METRICS")
    print("=" * 65)

    df = pd.read_csv(DATA_PATH)
    raw_count = len(df)
    df.drop_duplicates(inplace=True)
    categories = sorted(df["category"].unique())

    print(f"[*] Total dataset articles: {raw_count} (Unique after deduplication: {len(df)})")
    print(f"[*] Target Categories ({len(categories)}): {', '.join(categories)}")

    X = df["text"].to_numpy()
    y = df["category"].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=500, random_state=42, stratify=y
    )
    print(f"[*] Data split: Train = {len(x_train)} samples | Test = {len(x_test)} samples")

    print("[*] Training Pipeline (TfidfVectorizer + MultinomialNB)...")
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
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    accuracy = pipeline.score(x_test, y_test)

    print("\n" + "=" * 65)
    print("                 MODEL EVALUATION METRICS")
    print("=" * 65)
    print(f"Test Set Accuracy: {accuracy * 100:.2f}%\n")

    print("-" * 65)
    print("Classification Report:")
    print("-" * 65)
    print(sklm.classification_report(y_test, y_pred, labels=categories, digits=4))

    print("-" * 65)
    print("Confusion Matrix:")
    print("-" * 65)
    cm = sklm.confusion_matrix(y_test, y_pred, labels=categories)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {cat}" for cat in categories],
        columns=[f"Pred: {cat}" for cat in categories],
    )
    print(cm_df.to_string())
    print("=" * 65)
    print("[*] Launching Native Desktop Prediction Window...\n")

    return pipeline, categories


class BBCClassifierApp:
    def __init__(self, root, pipeline, categories):
        self.root = root
        self.pipeline = pipeline
        self.categories = categories

        self.root.title("BBC News Topic Classifier - TF-IDF & MultinomialNB")
        self.root.geometry("960x720")
        self.root.minsize(820, 600)
        self.root.config(bg="#0b0f19")

        self.build_ui()

    def build_ui(self):
        """Construct the modern dark-themed interface."""
        # Main container with padding
        main_container = tk.Frame(self.root, bg="#0b0f19", padx=28, pady=24)
        main_container.pack(fill="both", expand=True)

        # 1. Header Frame
        header = tk.Frame(main_container, bg="#0b0f19")
        header.pack(fill="x", pady=(0, 16))

        badge = tk.Label(
            header,
            text="TF-IDF + Multinomial Naive Bayes",
            font=("Segoe UI", 9, "bold"),
            fg="#60a5fa",
            bg="#1e293b",
            padx=10,
            pady=3,
        )
        badge.pack(anchor="center", pady=(0, 6))

        title = tk.Label(
            header,
            text="BBC News Topic Classifier",
            font=("Segoe UI", 22, "bold"),
            fg="#f9fafb",
            bg="#0b0f19",
        )
        title.pack(anchor="center")

        subtitle = tk.Label(
            header,
            text="Categorize news articles into Business, Entertainment, Politics, Sport, or Tech.",
            font=("Segoe UI", 10),
            fg="#94a3b8",
            bg="#0b0f19",
        )
        subtitle.pack(anchor="center", pady=(2, 0))

        # 2. Main Card Frame
        card = tk.Frame(main_container, bg="#111827", bd=1, relief="solid", padx=22, pady=18)
        card.config(highlightbackground="#1f2937", highlightthickness=1)
        card.pack(fill="both", expand=True)

        # Quick Examples row
        samples_bar = tk.Frame(card, bg="#111827")
        samples_bar.pack(fill="x", pady=(0, 12))

        tk.Label(
            samples_bar,
            text="Quick Examples:",
            font=("Segoe UI", 9, "bold"),
            fg="#9ca3af",
            bg="#111827",
        ).pack(side="left", padx=(0, 8))

        sample_configs = [
            ("⚽ Sport", "sport"),
            ("💻 Tech", "tech"),
            ("📈 Business", "business"),
            ("🎬 Entertainment", "entertainment"),
            ("🏛️ Politics", "politics"),
        ]
        for label, cat in sample_configs:
            btn = tk.Button(
                samples_bar,
                text=label,
                font=("Segoe UI", 9),
                fg="#e2e8f0",
                bg="#1e293b",
                activebackground="#334155",
                activeforeground="#ffffff",
                relief="flat",
                bd=0,
                padx=10,
                pady=4,
                cursor="hand2",
                command=lambda c=cat: self.load_sample(c),
            )
            btn.pack(side="left", padx=4)

        # Text input area
        input_container = tk.Frame(card, bg="#111827")
        input_container.pack(fill="both", expand=True, pady=(0, 10))

        self.text_input = tk.Text(
            input_container,
            height=6,
            font=("Segoe UI", 11),
            bg="#0f172a",
            fg="#f8fafc",
            insertbackground="#60a5fa",
            relief="solid",
            bd=1,
            highlightthickness=1,
            highlightcolor="#3b82f6",
            highlightbackground="#334155",
            wrap="word",
            padx=12,
            pady=10,
        )
        self.text_input.pack(fill="both", expand=True)
        self.text_input.insert("1.0", SAMPLE_TEXTS["sport"])
        self.text_input.bind("<Control-Return>", lambda event: self.classify_text())

        # Action bar
        action_bar = tk.Frame(card, bg="#111827")
        action_bar.pack(fill="x", pady=(0, 14))

        tk.Label(
            action_bar,
            text="Tip: Press Ctrl + Enter to classify",
            font=("Segoe UI", 9),
            fg="#64748b",
            bg="#111827",
        ).pack(side="left")

        self.classify_btn = tk.Button(
            action_bar,
            text="Classify Article  ▶",
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg="#2563eb",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=22,
            pady=9,
            cursor="hand2",
            command=self.classify_text,
        )
        self.classify_btn.pack(side="right")

        # 3. Prediction Result Box
        self.result_frame = tk.Frame(card, bg="#161f30", bd=1, relief="solid", padx=18, pady=16)
        self.result_frame.config(highlightbackground="#243048", highlightthickness=1)
        self.result_frame.pack(fill="x")

        # Top row: Predicted Category + Confidence
        res_top = tk.Frame(self.result_frame, bg="#161f30")
        res_top.pack(fill="x", pady=(0, 10))

        tk.Label(
            res_top,
            text="PREDICTED CATEGORY:",
            font=("Segoe UI", 9, "bold"),
            fg="#94a3b8",
            bg="#161f30",
        ).pack(side="left")

        self.pred_category_label = tk.Label(
            res_top,
            text="READY",
            font=("Segoe UI", 18, "bold"),
            fg="#10b981",
            bg="#161f30",
            padx=10,
        )
        self.pred_category_label.pack(side="left")

        self.confidence_label = tk.Label(
            res_top,
            text="Confidence: --",
            font=("Segoe UI", 11, "bold"),
            fg="#f8fafc",
            bg="#0f172a",
            padx=14,
            pady=4,
        )
        self.confidence_label.pack(side="right")

        # Probability breakdown title
        tk.Label(
            self.result_frame,
            text="Probability Distribution Across Categories:",
            font=("Segoe UI", 9, "bold"),
            fg="#94a3b8",
            bg="#161f30",
        ).pack(anchor="w", pady=(6, 8))

        # Probability bars for each category
        self.cat_widgets = {}
        for cat in self.categories:
            row = tk.Frame(self.result_frame, bg="#161f30")
            row.pack(fill="x", pady=3)

            name_lbl = tk.Label(
                row,
                text=f"{cat.capitalize():<14}",
                font=("Segoe UI", 9, "bold"),
                fg="#cbd5e1",
                bg="#161f30",
                width=14,
                anchor="w",
            )
            name_lbl.pack(side="left")

            bar_canvas = tk.Canvas(row, height=12, bg="#243048", highlightthickness=0)
            bar_canvas.pack(side="left", fill="x", expand=True, padx=10)

            pct_lbl = tk.Label(
                row,
                text="0.00%",
                font=("Consolas", 9, "bold"),
                fg="#94a3b8",
                bg="#161f30",
                width=8,
                anchor="e",
            )
            pct_lbl.pack(side="right")

            self.cat_widgets[cat] = {
                "name": name_lbl,
                "canvas": bar_canvas,
                "pct": pct_lbl,
            }

    def load_sample(self, category):
        """Insert a sample text and classify it immediately."""
        sample = SAMPLE_TEXTS.get(category, "")
        self.text_input.delete("1.0", "end")
        self.text_input.insert("1.0", sample)
        self.classify_text()

    def classify_text(self):
        """Execute text classification and update the prediction card."""
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Input Required", "Please enter or paste an article text to classify.")
            return

        # Inference
        pred = self.pipeline.predict([text])[0]
        probs = self.pipeline.predict_proba([text])[0]

        prob_dict = {
            cls: float(prob) * 100 for cls, prob in zip(self.pipeline.classes_, probs)
        }
        confidence = prob_dict[pred]

        # Update prediction header
        color = CATEGORY_COLORS.get(pred, CATEGORY_COLORS["default"])
        self.pred_category_label.config(text=pred.upper(), fg=color)
        self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")

        # Update probability bars
        for cat in self.categories:
            p = prob_dict.get(cat, 0.0)
            widgets = self.cat_widgets[cat]
            widgets["pct"].config(text=f"{p:5.2f}%")

            # Draw bar fill
            canvas = widgets["canvas"]
            canvas.update_idletasks()
            w = max(canvas.winfo_width(), 100)
            canvas.delete("bar")

            bar_color = color if cat == pred else "#475569"
            fill_w = int((p / 100.0) * w)
            canvas.create_rectangle(0, 0, fill_w, 12, fill=bar_color, width=0, tags="bar")


def main():
    # 1. Train and print metrics directly on the terminal
    pipeline, categories = print_terminal_metrics()

    # 2. Launch Native Desktop UI
    root = tk.Tk()
    app = BBCClassifierApp(root, pipeline, categories)
    root.mainloop()


if __name__ == "__main__":
    main()
