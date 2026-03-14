"""
Phishing URL Detection Console Application
Uses the trained XGBoost model to detect phishing URLs in real-time.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import re
from urllib.parse import urlparse
import ipaddress
from colorama import init, Fore, Style

# Initialize colorama for colored console output
init(autoreset=True)


# ==================== CONFIGURATION ====================
MODEL_FILE = "xgboost_phishing_model.json"
FEATURE_IMPORTANCE_FILE = "feature_importance.csv"
WHITELIST_FILE = "whitelist.txt"

# URL Shortening Services
SHORTENING_SERVICES = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                      r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                      r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                      r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|db\.tt|" \
                      r"qr\.ae|adf\.ly|bitly\.com|cur\.lv|ow\.ly|ity\.im|q\.gs|po\.st|bc\.vc|twitthis\.com|" \
                      r"u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|prettylinkpro\.com|scrnch\.me|" \
                      r"filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net"

# Suspicious Keywords
SUSPICIOUS_KEYWORDS = ['login', 'signin', 'account', 'verify', 'update', 'secure', 'banking',
                       'paypal', 'ebay', 'amazon', 'confirm', 'suspend', 'verify', 'luck',
                       'bonus', 'free', 'click']

# Load whitelist of known legitimate domains
def load_whitelist(filepath):
    """Load legitimate domains from whitelist file."""
    domains = set()
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    domains.add(line.lower())
    except FileNotFoundError:
        print(f"{Fore.YELLOW}⚠ Warning: Whitelist file '{filepath}' not found. Using empty whitelist.{Style.RESET_ALL}")
    return domains

LEGITIMATE_DOMAINS = load_whitelist(WHITELIST_FILE)



# ==================== FUNCTIONS ====================
def extract_domain(url):
    """Extract the base domain from a URL."""
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        
        # Remove 'www.' prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Remove port if present
        if ':' in domain:
            domain = domain.split(':')[0]
        
        return domain.lower()
    except:
        return None


def is_whitelisted_domain(url):
    """Check if the URL's domain is in the legitimate whitelist."""
    domain = extract_domain(url)
    if not domain:
        return False
    
    # Check exact match
    if domain in LEGITIMATE_DOMAINS:
        return True
    
    # Check if it's a subdomain of a whitelisted domain
    # e.g., 'mail.google.com' should match 'google.com'
    for trusted_domain in LEGITIMATE_DOMAINS:
        if domain.endswith('.' + trusted_domain):
            return True
    
    return False


def load_model(model_path):
    """Load the trained XGBoost model."""
    try:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        print(f"{Fore.GREEN}✓ Model loaded successfully from {model_path}{Style.RESET_ALL}\n")
        return model
    except Exception as e:
        print(f"{Fore.RED}✗ Error loading model: {e}{Style.RESET_ALL}")
        sys.exit(1)


def extract_features(url):
    """Extract 29 URL-only features from a single URL (no network calls)."""
    try:
        features = {}
        
        # Parse URL
        try:
            parsed = urlparse(str(url))
        except:
            from urllib.parse import ParseResult
            parsed = ParseResult('', '', '', '', '', '')
        
        # 1. Have_IP - Check if domain is IP address
        try:
            domain = parsed.netloc.split(':')[0]
            if domain:
                ipaddress.ip_address(domain)
                features['Have_IP'] = 1
            else:
                features['Have_IP'] = 0
        except:
            features['Have_IP'] = 0
        
        # 2. Have_At - '@' symbol in URL
        features['Have_At'] = 1 if '@' in str(url) else 0
        
        # 3. URL_Length - Length of URL
        features['URL_Length'] = len(str(url))
        
        # 4. URL_Depth - Number of subdirectories
        features['URL_Depth'] = len([x for x in parsed.path.split('/') if x])
        
        # 5. Redirection - '//' after protocol
        try:
            pos = str(url).rfind('//')
            features['Redirection'] = 1 if pos > 7 else 0
        except:
            features['Redirection'] = 0
        
        # 6. https_Domain - 'http' or 'https' in domain part
        features['https_Domain'] = 1 if 'http' in parsed.netloc.lower() else 0
        
        # 7. TinyURL - URL shortening service
        try:
            features['TinyURL'] = 1 if re.search(SHORTENING_SERVICES, str(url)) else 0
        except:
            features['TinyURL'] = 0
        
        # 8. Prefix_Suffix - '-' in domain
        features['Prefix_Suffix'] = 1 if '-' in parsed.netloc else 0
        
        # 9. IsHTTPS - Uses HTTPS protocol
        features['IsHTTPS'] = 1 if parsed.scheme == 'https' else 0
        
        # 10. NoOfDots - Count of '.' in URL
        features['NoOfDots'] = str(url).count('.')
        
        # 11. NoOfHyphen - Count of '-' in URL
        features['NoOfHyphen'] = str(url).count('-')
        
        # 12. NoOfUnderscore - Count of '_' in URL
        features['NoOfUnderscore'] = str(url).count('_')
        
        # 13. NoOfSlash - Count of '/' in URL
        features['NoOfSlash'] = str(url).count('/')
        
        # 14. NoOfQuestionMark - Count of '?' in URL
        features['NoOfQuestionMark'] = str(url).count('?')
        
        # 15. NoOfEquals - Count of '=' in URL
        features['NoOfEquals'] = str(url).count('=')
        
        # 16. NoOfAmpersand - Count of '&' in URL
        features['NoOfAmpersand'] = str(url).count('&')
        
        # 17. NoOfPercent - Count of '%' in URL
        features['NoOfPercent'] = str(url).count('%')
        
        # 18. NoOfDigits - Count of digits in URL
        features['NoOfDigits'] = sum(c.isdigit() for c in str(url))
        
        # 19. NoOfLetters - Count of letters in URL
        features['NoOfLetters'] = sum(c.isalpha() for c in str(url))
        
        # 20. LetterRatio - Ratio of letters to total length
        features['LetterRatio'] = features['NoOfLetters'] / max(features['URL_Length'], 1)
        
        # 21. DigitRatio - Ratio of digits to total length
        features['DigitRatio'] = features['NoOfDigits'] / max(features['URL_Length'], 1)
        
        # 22. HasSubdomain - Has subdomain (more than domain.tld)
        parts = parsed.netloc.split('.')
        features['HasSubdomain'] = 1 if len(parts) > 2 else 0
        
        # 23. SubdomainCount - Number of subdomains
        features['SubdomainCount'] = len(parts) - 2 if len(parts) > 2 else 0
        
        # 24. DomainLength - Length of domain
        features['DomainLength'] = len(parsed.netloc)
        
        # 25. PathLength - Length of path
        features['PathLength'] = len(parsed.path)
        
        # 26. QueryLength - Length of query string
        features['QueryLength'] = len(parsed.query)
        
        # 27. HasPort - URL has port number
        features['HasPort'] = 1 if ':' in parsed.netloc and parsed.netloc.split(':')[-1].isdigit() else 0
        
        # 28. SuspiciousWords - Contains suspicious keywords
        try:
            url_lower = str(url).lower()
            features['SuspiciousWords'] = 1 if any(keyword in url_lower for keyword in SUSPICIOUS_KEYWORDS) else 0
        except:
            features['SuspiciousWords'] = 0
        
        # 29. SpecialCharCount - Count of special characters
        try:
            special = '!@#$%^&*()[]{}|\\:;"\'<>,.?/~`'
            features['SpecialCharCount'] = sum(1 for c in str(url) if c in special)
        except:
            features['SpecialCharCount'] = 0
        
        return features
    except Exception as e:
        print(f"{Fore.RED}✗ Error extracting features: {e}{Style.RESET_ALL}")
        return None


def convert_feature_types(features_df):
    """Convert feature types to ensure compatibility with XGBoost."""
    # Ensure all columns are numeric
    for col in features_df.columns:
        if features_df[col].dtype == 'object':
            try:
                # Try to convert to numeric
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)
            except:
                pass
        
        # Ensure no NaN or inf values
        features_df[col] = features_df[col].replace([np.inf, -np.inf], 0).fillna(0)
    
    return features_df


def features_to_dataframe(features_dict):
    """Convert features dictionary to DataFrame with correct column order."""
    # Define expected feature order (29 features used in training)
    expected_features = [
        'Have_IP', 'Have_At', 'URL_Length', 'URL_Depth', 'Redirection',
        'https_Domain', 'TinyURL', 'Prefix_Suffix', 'IsHTTPS', 'NoOfDots',
        'NoOfHyphen', 'NoOfUnderscore', 'NoOfSlash', 'NoOfQuestionMark',
        'NoOfEquals', 'NoOfAmpersand', 'NoOfPercent', 'NoOfDigits',
        'NoOfLetters', 'LetterRatio', 'DigitRatio', 'HasSubdomain',
        'SubdomainCount', 'DomainLength', 'PathLength', 'QueryLength',
        'HasPort', 'SuspiciousWords', 'SpecialCharCount'
    ]
    
    # Create DataFrame with single row in correct order
    df = pd.DataFrame([features_dict])
    
    # Ensure all expected features exist and in correct order
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Return only expected features in correct order
    return df[expected_features]


def predict_url(model, url):
    """Predict if a URL is phishing or legitimate."""
    print(f"\n{Fore.CYAN}Analyzing URL: {url}{Style.RESET_ALL}")
    print("-" * 80)
    
    # Check whitelist first
    if is_whitelisted_domain(url):
        print(f"\n{Fore.YELLOW}PREDICTION RESULTS:{Style.RESET_ALL}")
        print("=" * 80)
        print(f"{Fore.GREEN}✓ This URL appears to be LEGITIMATE{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}ℹ️  Domain recognized as trusted (whitelisted){Style.RESET_ALL}")
        print(f"\nConfidence Scores:")
        print(f"  Legitimate: {Fore.GREEN}99.00%{Style.RESET_ALL}")
        print(f"  Phishing:   {Fore.RED}1.00%{Style.RESET_ALL}")
        print(f"\nRisk Level: {Fore.GREEN}LOW RISK{Style.RESET_ALL}")
        print("=" * 80)
        return
    
    # Extract features
    print(f"{Fore.CYAN}Extracting features... (this may take a few seconds){Style.RESET_ALL}")
    features_dict = extract_features(url)
    if features_dict is None:
        return
    
    # Convert to DataFrame using helper function
    features_df = features_to_dataframe(features_dict)
    
    # Convert feature types to ensure XGBoost compatibility
    features_df = convert_feature_types(features_df)
    
    # Make prediction
    try:
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        
        # probabilities[0] is legitimate, probabilities[1] is phishing
        legitimate_prob = probabilities[0] * 100
        phishing_prob = probabilities[1] * 100
        
        # Display results
        print(f"\n{Fore.YELLOW}PREDICTION RESULTS:{Style.RESET_ALL}")
        print("=" * 80)
        
        # prediction 0 = legitimate (benign), prediction 1 = phishing
        if prediction == 1:
            print(f"{Fore.RED}⚠️  WARNING: This URL appears to be PHISHING!{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}✓ This URL appears to be LEGITIMATE{Style.RESET_ALL}")
        
        print(f"\nConfidence Scores:")
        print(f"  Legitimate: {Fore.GREEN}{legitimate_prob:.2f}%{Style.RESET_ALL}")
        print(f"  Phishing:   {Fore.RED}{phishing_prob:.2f}%{Style.RESET_ALL}")
        
        # Risk level based on probability
        print(f"\nRisk Level: ", end="")
        if phishing_prob >= 80:
            print(f"{Fore.RED}HIGH RISK{Style.RESET_ALL}")
        elif phishing_prob >= 50:
            print(f"{Fore.YELLOW}MEDIUM RISK{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}LOW RISK{Style.RESET_ALL}")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"{Fore.RED}✗ Error making prediction: {e}{Style.RESET_ALL}")


def display_header():
    """Display application header."""
    print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'PHISHING URL DETECTION SYSTEM':^80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'Powered by XGBoost Machine Learning':^80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")


def run_interactive_mode(model):
    """Run the application in interactive mode."""
    display_header()
    
    while True:
        try:
            # Get URL from user
            print(f"\n{Fore.YELLOW}Enter a URL to check (or 'quit' to exit):{Style.RESET_ALL}")
            url = input("> ").strip()
            
            # Check for exit command
            if url.lower() in ['quit', 'exit', 'q']:
                print(f"\n{Fore.CYAN}Thank you for using Phishing Detection System!{Style.RESET_ALL}")
                break
            
            # Skip empty input
            if not url:
                continue
            
            # Add http:// if no protocol specified
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            
            # Predict
            predict_url(model, url)
            
        except KeyboardInterrupt:
            print(f"\n\n{Fore.CYAN}Thank you for using Phishing Detection System!{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"{Fore.RED}✗ Unexpected error: {e}{Style.RESET_ALL}")


def run_batch_mode(model, urls):
    """Run the application in batch mode for multiple URLs."""
    display_header()
    print(f"{Fore.YELLOW}Processing {len(urls)} URLs...{Style.RESET_ALL}\n")
    
    results = []
    for idx, url in enumerate(urls, 1):
        print(f"\n[{idx}/{len(urls)}]", end=" ")
        
        # Check whitelist first
        if is_whitelisted_domain(url):
            result = {
                'url': url,
                'prediction': 'LEGITIMATE',
                'phishing_probability': '1.00%',
                'source': 'WHITELIST'
            }
            results.append(result)
            print(f"{url}: {Fore.GREEN}LEGITIMATE (Whitelisted){Style.RESET_ALL} (1.00%)")
            continue
        
        features_dict = extract_features(url)
        if features_dict is None:
            results.append({'url': url, 'prediction': 'ERROR', 'phishing_probability': '0%', 'source': 'ERROR'})
            continue
        
        # Convert to DataFrame using helper function
        features_df = features_to_dataframe(features_dict)
        
        # Convert feature types to ensure XGBoost compatibility
        features_df = convert_feature_types(features_df)
        
        try:
            prediction = model.predict(features_df)[0]
            probabilities = model.predict_proba(features_df)[0]
            # probabilities[0] is legitimate, probabilities[1] is phishing
            phishing_prob = probabilities[1] * 100
            
            result = {
                'url': url,
                'prediction': 'PHISHING' if prediction == 1 else 'LEGITIMATE',
                'phishing_probability': f"{phishing_prob:.2f}%",
                'source': 'MODEL'
            }
            results.append(result)
            
            # prediction 0 = legitimate, prediction 1 = phishing
            status = f"{Fore.RED}PHISHING{Style.RESET_ALL}" if prediction == 1 else f"{Fore.GREEN}LEGITIMATE{Style.RESET_ALL}"
            print(f"{url}: {status} ({phishing_prob:.2f}%)")
            
        except Exception as e:
            print(f"{url}: {Fore.RED}ERROR - {e}{Style.RESET_ALL}")
            results.append({'url': url, 'prediction': 'ERROR', 'phishing_probability': '0%', 'source': 'ERROR'})
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = "detection_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n{Fore.GREEN}✓ Results saved to {output_file}{Style.RESET_ALL}")


# ==================== MAIN ====================
def main():
    """Main application entry point."""
    # Load model
    model = load_model(MODEL_FILE)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Batch mode: URLs provided as arguments
        urls = sys.argv[1:]
        run_batch_mode(model, urls)
    else:
        # Interactive mode
        run_interactive_mode(model)


if __name__ == "__main__":
    main()
