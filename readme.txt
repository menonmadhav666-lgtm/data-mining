BBC Text Categorization (TF-IDF + Multinomial Naive Bayes)

This project trains a text classification pipeline using TF-IDF Vectorizer and Multinomial Naive Bayes (MultinomialNB) on the BBC News dataset.

The system:

Loads and deduplicates 2,225 BBC news articles across 5 categories (business, entertainment, politics, sport, tech).
Trains the TfidfVectorizer + MultinomialNB pipeline with optimal parameters identified through hyperparameter tuning.
Evaluates test set metrics (Test Accuracy, Precision, Recall, F1-Score, and Confusion Matrix).
Takes new text inputs (interactively or via CLI arguments) and predicts the category with a full probability breakdown.
Project Structure

bbc_text_classification/
├── data/
│   └── bbc-text.csv                            # Downloaded BBC News dataset
├── classify.py                                 # Interactive & CLI Python script
└── README.md     
                              # Documentation and usage guide
QuickStart

1. Run the Python Script

To run the full evaluation and launch the interactive prediction loop:

bash
python classify.py

To classify a specific headline or article immediately from the command line:

bash
python classify.py --text "Manchester United clinched a 3-2 victory in the dying minutes of the championship."
2. Run the Jupyter Notebook

Open and run bbc_text_categorization_modified.ipynb in VS Code or Jupyter:

bash
jupyter notebook bbc_text_categorization_modified.ipynb

The notebook contains:

Step-by-step data inspection and deduplication.
Stratified train-test split.
Pipeline training (TfidfVectorizer + MultinomialNB).
Formatted metrics evaluation table and confusion matrix.
predict_article() function with probability visualizations.
Interactive input() cell to test custom texts.
