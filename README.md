# Phishing URL Detector

A fast and simple tool that uses AI (XGBoost) to detect dangerous phishing links. 

## Features

- **Fast Checks:** Analyzes links in milliseconds based on their text structure.
- **High Accuracy:** Uses machine learning (AI) for accurate predictions.
- **Two Modes:** Check links one-by-one interactively, or scan a batch from the command line.
- **Whitelist System:** Safelist known good sites (like google.com) to save time and avoid false warnings.

## Project Files

- **`detect_phishing.py`**: The main tool you run to check if URLs are safe.
- **Feature Extractors**: Scripts that pull useful traits from URLs so the AI can learn.
  - `fast_url_feature_extraction.py`: Very fast, only looks at the URL text.
- **`train_xgboost.py`**: Trains the AI model on the traits extracted from the datasets.

## How to Install

1. Make sure you have Python 3.8 or newer.
2. Clone this folder and open a terminal inside it.
3. Install the required libraries:
   ```bash
   pip install pandas numpy xgboost scikit-learn beautifulsoup4 requests colorama tldextract python-whois
   ```

## How to Use

### 1. Check a single URL (Interactive)
Run this command, then type or paste the URL you want to check:
```bash
python detect_phishing.py
```

### 2. Check Multiple URLs (Batch)
Run this command with a list of URLs you want to check. The results will be saved to a file called `detection_results.csv`:
```bash
python detect_phishing.py "http://google.com" "http://bad-link.com"
```

### 3. Re-train the AI (Optional)
If you want to train your own version of the AI model:
1. Run `python fast_url_feature_extraction.py` to pull traits from the links.
2. Run `python train_xgboost.py` to train the new model.
