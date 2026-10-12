BBC Text Categorization (TF-IDF + Multinomial Naive Bayes)

This project trains a text classification pipeline using TF-IDF Vectorizer and Multinomial Naive Bayes (MultinomialNB) on the BBC News dataset.

Core Data Mining Pipeline:

1. Data Loading & Deduplication: Loads and deduplicates 2,225 BBC news articles across 5 categories (business, entertainment, politics, sport, tech).
2. Feature Extraction: Uses TfidfVectorizer with unigram and bigram features (ngram_range=(1, 2)), sublinear term frequency scaling, and English stop words removal.
3. Model Training: Trains a MultinomialNB classifier with optimal Laplace smoothing (alpha = 0.10).
4. Evaluation: Evaluates test set metrics on 500 stratified holdout samples (Accuracy ~96.8%, Precision, Recall, and F1-Score).
5. Inference: Takes text input and predicts the news category with a full probability breakdown across all 5 classes.

Project Structure:

data-mining/
├── data/
│   └── bbc-text.csv          # BBC News dataset
├── web/                      # Web application
│   ├── server.py             # Server & ML classification pipeline
│   ├── index.html            # Web interface
│   ├── styles.css            # Web interface styles
│   └── app.js                # Client logic
├── classify.py               # Standalone classification script
└── README.md                 # Project documentation

Usage Guide:

To run the web application:

From the project root:
    python web/server.py

Or from inside the web/ directory:
    python server.py

Once started, open your browser at:
    http://localhost:8000
