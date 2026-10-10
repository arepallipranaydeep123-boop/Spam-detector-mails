# Gmail / Email Spam Detection

A working spam classifier using TF-IDF + Naive Bayes / Logistic Regression (scikit-learn).
Trains both, picks the best by spam F1, and saves it.

## Quick start
```bash
pip install -r requirements.txt
python make_sample_data.py     # demo data (synthetic) -> data/emails.csv
python train.py                # trains + evaluates + saves models/spam_model.joblib
python predict.py "Congratulations! You won a free iPhone. Click http://bit.ly/x"
```

## Use a real dataset (recommended)
The bundled data is synthetic, so scores will look near-perfect. For realistic results,
download a real dataset and point train.py at it, e.g.:
- SMS Spam Collection (Kaggle / UCI) -> `python train.py --data spam.csv`
- Enron-Spam or SpamAssassin corpora converted to CSV with `label,text` columns

## Scan your Gmail (optional, read-only)
See the docstring at the top of `gmail_scan.py` for Google API setup, then:
```bash
python gmail_scan.py --max 20
```

## Files
- `make_sample_data.py` - builds demo CSV
- `train.py` - training + evaluation
- `predict.py` - CLI predictions
- `gmail_scan.py` - Gmail API scanner
- `models/` - saved model
