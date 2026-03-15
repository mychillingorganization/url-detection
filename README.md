# Phishing URL Detector

A fast and simple tool that uses AI (XGBoost) to detect dangerous phishing links. 

## Features

- **Fast Checks:** Analyzes links in milliseconds based on their text structure.
- **High Accuracy:** Uses machine learning (AI) for accurate predictions.
- **Two Modes:** Check links one-by-one interactively, or scan a batch from the command line.
- **Optional Live DOM Mode:** Use `--with-html` to fetch and parse page HTML at detection time.
- **Whitelist System:** Safelist known good sites (like google.com) to save time and avoid false warnings.
- **Schema Compatibility:** Training and detection now handle feature-schema changes more safely (including URL+HTML feature sets).

## Project Files

- **`detect_phishing.py`**: The main tool you run to check if URLs are safe.
- **Feature Extractors**: Scripts that pull useful traits from URLs so the AI can learn.
  - `fast_url_feature_extraction.py`: Fast URL + optional HTML feature extraction without network calls.
- **`train_xgboost.py`**: Trains the AI model and normalizes labels/types for stronger dataset compatibility.

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

### 3. Check URLs with Live HTML Parsing (Optional)
Enable runtime HTML fetch + DOM feature extraction:
```bash
python detect_phishing.py --with-html "https://example.com"
```

You can also use it with multiple URLs:
```bash
python detect_phishing.py --with-html "https://example.com" "http://test.com/login"
```

### 4. Re-train the AI (Optional)
If you want to train your own version of the AI model:
1. Run `python fast_url_feature_extraction.py` to pull traits from the links.
2. Run `python train_xgboost.py` to train the new model.

### Notes on Model Compatibility
- `detect_phishing.py` now reads expected feature names from the trained model (or `feature_importance.csv` fallback) and aligns inference input automatically.
- If your model was trained with URL+HTML columns, detection still works for URL-only checks by filling unavailable HTML features with safe defaults.
- With `--with-html`, detection prints `✓ HTML fetched successfully` when fetch works, or a warning and fallback-to-default HTML feature values when fetch/parse fails.
