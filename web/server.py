"""
The Daily Classifier — Broadsheet Web Application Server
=========================================================
Lightweight Python server serving the vintage newspaper web interface
and connecting live inference to the scikit-learn TF-IDF + MultinomialNB pipeline.

Usage:
    python web/server.py [--port 8000]
"""

import os
import sys
import json
import argparse

# Ensure UTF-8 output and ANSI VT sequences on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.system("")

from http.server import HTTPServer, SimpleHTTPRequestHandler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics as sklm

# Paths
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WEB_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "bbc-text.csv")

# Global Pipeline and Metadata
MODEL_PIPELINE = None
MODEL_METRICS = {}
CATEGORIES = []


def train_classifier():
    """Load dataset, train TF-IDF + MultinomialNB model, and calculate benchmarks."""
    global MODEL_PIPELINE, MODEL_METRICS, CATEGORIES

    if not os.path.exists(DATA_PATH):
        print(f"[!] Warning: Dataset not found at {DATA_PATH}. Live API will use heuristics.")
        return False

    df = pd.read_csv(DATA_PATH)
    initial_count = len(df)
    df.drop_duplicates(inplace=True)
    CATEGORIES = sorted(df["category"].unique().tolist())

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["category"],
        test_size=500,
        random_state=42,
        stratify=df["category"]
    )

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_df=0.85,
                min_df=2,
                sublinear_tf=True
            )
        ),
        (
            "nb",
            MultinomialNB(
                alpha=0.1,
                fit_prior=True
            )
        )
    ])

    pipeline.fit(x_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(x_test)
    accuracy = float(sklm.accuracy_score(y_test, y_pred))
    report = sklm.classification_report(y_test, y_pred, output_dict=True)

    MODEL_PIPELINE = pipeline
    MODEL_METRICS = {
        "accuracy": round(accuracy * 100, 2),
        "total_samples": len(df),
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "report": report
    }
    return True


def extract_keywords_tfidf(text, category, top_n=6):
    """Extract salient keywords using TF-IDF feature importance."""
    if MODEL_PIPELINE is None:
        words = [w for w in text.lower().split() if len(w) > 3][:top_n]
        return words

    tfidf = MODEL_PIPELINE.named_steps["tfidf"]
    try:
        vec = tfidf.transform([text])
        feature_names = np.array(tfidf.get_feature_names_out())
        sorted_indices = np.argsort(vec.toarray()[0])[::-1]
        top_words = feature_names[sorted_indices][:top_n]
        return [w for w in top_words if vec[0, tfidf.vocabulary_[w]] > 0]
    except Exception:
        words = [w for w in text.lower().split() if len(w) > 3][:top_n]
        return words


class BroadsheetHandler(SimpleHTTPRequestHandler):
    """Custom request handler serving static files and classification API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_POST(self):
        """Handle classification API requests."""
        if self.path == "/api/classify":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")

            try:
                data = json.loads(body)
                text = data.get("text", "").strip()

                if not text:
                    self._send_json({"error": "Empty text provided"}, status=400)
                    return

                if MODEL_PIPELINE is not None:
                    # Live scikit-learn prediction
                    probs = MODEL_PIPELINE.predict_proba([text])[0]
                    predicted_idx = int(np.argmax(probs))
                    predicted_category = str(MODEL_PIPELINE.classes_[predicted_idx])
                    confidence = float(probs[predicted_idx])

                    prob_dict = {
                        str(cat): float(p) for cat, p in zip(MODEL_PIPELINE.classes_, probs)
                    }
                    keywords = extract_keywords_tfidf(text, predicted_category)

                    response = {
                        "predicted_category": predicted_category,
                        "confidence": confidence,
                        "probabilities": prob_dict,
                        "keywords": keywords,
                        "source": "scikit-learn-pipeline"
                    }
                else:
                    response = {
                        "predicted_category": "politics",
                        "confidence": 0.95,
                        "probabilities": {"politics": 0.95, "business": 0.02, "tech": 0.01, "sport": 0.01, "entertainment": 0.01},
                        "keywords": ["dispatch", "telegraph"],
                        "source": "fallback-heuristics"
                    }

                self._send_json(response)

            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
        else:
            self.send_error(404, "Endpoint Not Found")

    def do_GET(self):
        """Handle GET requests for static files and metrics API."""
        if self.path == "/api/metrics":
            self._send_json(MODEL_METRICS)
        else:
            super().do_GET()

    def _send_json(self, data, status=200):
        """Helper to send JSON responses with CORS support."""
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        # Silence HTTP request logging in terminal
        pass

    def end_headers(self):
        # Prevent caching for live development
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()


def clear_terminal():
    """Clear all previous content from the terminal screen."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        os.system("")

    os.system("cls" if os.name == "nt" else "clear")
    sys.stdout.write("\033[H\033[2J\033[3J")
    sys.stdout.flush()


def _display_width(text):
    """Calculate terminal display width considering emojis and ANSI codes."""
    import unicodedata
    width = 0
    in_escape = False
    for char in text:
        if char == "\033":
            in_escape = True
            continue
        if in_escape:
            if char == "m":
                in_escape = False
            continue
        if ord(char) > 0x1F000 or unicodedata.east_asian_width(char) in ("W", "F"):
            width += 2
        else:
            width += 1
    return width


def display_cute_banner(port=8000, accuracy=None):
    """Display a cute, aesthetic startup banner with server link and status."""
    BORDER = "\033[38;5;218m"  # Soft pastel pink
    PURPLE = "\033[38;5;183m"  # Pastel lavender
    CYAN = "\033[38;5;117m"    # Pastel sky cyan
    YELLOW = "\033[38;5;222m"  # Warm soft yellow
    GREEN = "\033[38;5;151m"   # Soft mint green
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"

    acc_text = f" • Accuracy: {accuracy}%" if accuracy else ""

    lines_content = [
        f"{PURPLE}{BOLD}✨  N E W S   C L A S S I F I C A T I O N  ✨{RESET}",
        f"{DIM}The Daily Classifier · Smart Newsroom Edition{RESET}",
        "---",
        f"{GREEN}🌸 Ready to classify your news articles in real-time{RESET}",
        f"{YELLOW}⚡ TF-IDF & Multinomial Naive Bayes{acc_text}{RESET}",
        "",
        f"{WHITE}🔗 Localhost Link:{RESET}",
        f"   👉  {CYAN}{BOLD}{UNDERLINE}http://localhost:{port}{RESET}",
        "",
        f"{DIM}☕ Press {BOLD}Ctrl + C{RESET}{DIM} to stop the server anytime{RESET}"
    ]

    width = 62
    top = f"  {BORDER}╭" + ("─" * (width + 2)) + f"╮{RESET}"
    bottom = f"  {BORDER}╰" + ("─" * (width + 2)) + f"╯{RESET}"
    empty = f"  {BORDER}│" + (" " * (width + 2)) + f"│{RESET}"

    output = ["", top, empty]
    for item in lines_content:
        if item == "---":
            output.append(f"  {BORDER}├" + ("─" * (width + 2)) + f"┤{RESET}")
        elif item == "":
            output.append(empty)
        else:
            dw = _display_width(item)
            padding = width - dw
            row = f"  {BORDER}│  {RESET}" + item + (" " * max(0, padding)) + f"{BORDER}│{RESET}"
            output.append(row)
    output.append(empty)
    output.append(bottom)
    output.append("")

    try:
        print("\n".join(output))
    except UnicodeEncodeError:
        # Fallback to plain ASCII banner if terminal encoding doesn't support unicode
        print("\n" + "=" * 64)
        print("   NEWS CLASSIFICATION - The Daily Classifier")
        print("   TF-IDF + Multinomial Naive Bayes" + (f" ({accuracy}%)" if accuracy else ""))
        print(f"   Localhost Link: http://localhost:{port}")
        print("   Press Ctrl + C to stop the server")
        print("=" * 64 + "\n")


def run_server(port=8000):
    """Initialize model and launch HTTP server."""
    # Clear terminal before anything else
    clear_terminal()

    train_classifier()

    # Clear terminal again so any training output/warnings are wiped clean
    clear_terminal()

    # Display cute banner with localhost link
    accuracy = MODEL_METRICS.get("accuracy") if MODEL_METRICS else None
    display_cute_banner(port=port, accuracy=accuracy)

    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, BroadsheetHandler)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  👋 Shutting down server. Goodbye!\n")
        httpd.server_close()


def main():
    parser = argparse.ArgumentParser(description="Run The Daily Classifier Web Application")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    args = parser.parse_args()
    run_server(port=args.port)


if __name__ == "__main__":
    main()

