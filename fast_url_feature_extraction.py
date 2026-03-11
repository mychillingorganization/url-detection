"""
Fast URL-Only Feature Extraction
Extracts features from URLs without making any network calls (no WHOIS, no HTTP requests).
Processes 1M URLs in minutes instead of days.
"""

import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
import ipaddress
import time
import warnings
warnings.filterwarnings('ignore')


# ==================== CONFIGURATION ====================
CONFIG = {
    'input_file': 'dataset.csv',
    'output_file': 'fast_url_features.csv',
    'batch_size': 10000,  # Process in batches for memory efficiency
}

# URL Shortening Services
SHORTENING_SERVICES = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                      r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                      r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                      r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|db\.tt|" \
                      r"qr\.ae|adf\.ly|bitly\.com|cur\.lv|ow\.ly|ity\.im|q\.gs|po\.st|bc\.vc|twitthis\.com|" \
                      r"u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|prettylinkpro\.com|scrnch\.me|" \
                      r"filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net"


# ==================== FAST FEATURE EXTRACTION FUNCTIONS ====================

def extract_url_features_vectorized(urls):
    """
    Extract features from URLs using vectorized operations for speed.
    Returns a DataFrame with all extracted features.
    """
    n = len(urls)
    features = {}
    
    print(f"Extracting features from {n} URLs...")
    
    # Parse all URLs at once - with error handling for malformed URLs
    parsed_urls = []
    for url in urls:
        try:
            parsed_urls.append(urlparse(str(url)))
        except (ValueError, Exception) as e:
            # If URL parsing fails, create a dummy parsed result
            from urllib.parse import ParseResult
            parsed_urls.append(ParseResult('', '', '', '', '', ''))
    
    # 1. Domain
    features['Domain'] = [parsed.netloc.replace('www.', '') if parsed.netloc.startswith('www.') 
                         else parsed.netloc for parsed in parsed_urls]
    
    # 2. Have_IP - Check if domain is IP address
    def check_ip(netloc):
        try:
            domain = netloc.split(':')[0]  # Remove port
            if not domain:  # Empty domain
                return 0
            ipaddress.ip_address(domain)
            return 1
        except (ValueError, Exception):
            return 0
    features['Have_IP'] = [check_ip(parsed.netloc) for parsed in parsed_urls]
    
    # 3. Have_At - '@' symbol in URL
    features['Have_At'] = [1 if '@' in str(url) else 0 for url in urls]
    
    # 4. URL_Length - Length of URL
    features['URL_Length'] = [len(str(url)) for url in urls]
    
    # 5. URL_Depth - Number of subdirectories
    features['URL_Depth'] = [len([x for x in parsed.path.split('/') if x]) for parsed in parsed_urls]
    
    # 6. Redirection - '//' after protocol
    def check_redirection(url):
        try:
            pos = str(url).rfind('//')
            if pos > 7:  # Beyond https://
                return 1
            return 0
        except:
            return 0
    features['Redirection'] = [check_redirection(url) for url in urls]
    
    # 7. https_Domain - 'http' or 'https' in domain part
    features['https_Domain'] = [1 if 'http' in parsed.netloc.lower() else 0 for parsed in parsed_urls]
    
    # 8. TinyURL - URL shortening service
    def check_tiny_url(url):
        try:
            return 1 if re.search(SHORTENING_SERVICES, str(url)) else 0
        except:
            return 0
    features['TinyURL'] = [check_tiny_url(url) for url in urls]
    
    # 9. Prefix/Suffix - '-' in domain
    features['Prefix_Suffix'] = [1 if '-' in parsed.netloc else 0 for parsed in parsed_urls]
    
    # 10. IsHTTPS - Uses HTTPS protocol
    features['IsHTTPS'] = [1 if parsed.scheme == 'https' else 0 for parsed in parsed_urls]
    
    # 11. NoOfDots - Count of '.' in URL
    features['NoOfDots'] = [str(url).count('.') for url in urls]
    
    # 12. NoOfHyphen - Count of '-' in URL
    features['NoOfHyphen'] = [str(url).count('-') for url in urls]
    
    # 13. NoOfUnderscore - Count of '_' in URL
    features['NoOfUnderscore'] = [str(url).count('_') for url in urls]
    
    # 14. NoOfSlash - Count of '/' in URL
    features['NoOfSlash'] = [str(url).count('/') for url in urls]
    
    # 15. NoOfQuestionMark - Count of '?' in URL
    features['NoOfQuestionMark'] = [str(url).count('?') for url in urls]
    
    # 16. NoOfEquals - Count of '=' in URL
    features['NoOfEquals'] = [str(url).count('=') for url in urls]
    
    # 17. NoOfAmpersand - Count of '&' in URL
    features['NoOfAmpersand'] = [str(url).count('&') for url in urls]
    
    # 18. NoOfPercent - Count of '%' in URL
    features['NoOfPercent'] = [str(url).count('%') for url in urls]
    
    # 19. NoOfDigits - Count of digits in URL
    features['NoOfDigits'] = [sum(c.isdigit() for c in str(url)) for url in urls]
    
    # 20. NoOfLetters - Count of letters in URL
    features['NoOfLetters'] = [sum(c.isalpha() for c in str(url)) for url in urls]
    
    # 21. LetterRatio - Ratio of letters to total length
    features['LetterRatio'] = [features['NoOfLetters'][i] / max(features['URL_Length'][i], 1) 
                              for i in range(n)]
    
    # 22. DigitRatio - Ratio of digits to total length
    features['DigitRatio'] = [features['NoOfDigits'][i] / max(features['URL_Length'][i], 1) 
                             for i in range(n)]
    
    # 23. HasSubdomain - Has subdomain (more than domain.tld)
    def has_subdomain(netloc):
        parts = netloc.split('.')
        return 1 if len(parts) > 2 else 0
    features['HasSubdomain'] = [has_subdomain(parsed.netloc) for parsed in parsed_urls]
    
    # 24. SubdomainCount - Number of subdomains
    features['SubdomainCount'] = [len(parsed.netloc.split('.')) - 2 if len(parsed.netloc.split('.')) > 2 else 0
                                  for parsed in parsed_urls]
    
    # 25. DomainLength - Length of domain
    features['DomainLength'] = [len(parsed.netloc) for parsed in parsed_urls]
    
    # 26. PathLength - Length of path
    features['PathLength'] = [len(parsed.path) for parsed in parsed_urls]
    
    # 27. QueryLength - Length of query string
    features['QueryLength'] = [len(parsed.query) for parsed in parsed_urls]
    
    # 28. HasPort - URL has port number
    def has_port(netloc):
        return 1 if ':' in netloc and netloc.split(':')[-1].isdigit() else 0
    features['HasPort'] = [has_port(parsed.netloc) for parsed in parsed_urls]
    
    # 29. SuspiciousWords - Contains suspicious keywords
    suspicious_keywords = ['login', 'signin', 'account', 'verify', 'update', 'secure', 'banking',
                          'paypal', 'ebay', 'amazon', 'confirm', 'suspend', 'verify', 'luck',
                          'bonus', 'free', 'click']
    def has_suspicious_words(url):
        try:
            url_lower = str(url).lower()
            return 1 if any(keyword in url_lower for keyword in suspicious_keywords) else 0
        except:
            return 0
    features['SuspiciousWords'] = [has_suspicious_words(url) for url in urls]
    
    # 30. SpecialCharCount - Count of special characters
    def count_special_chars(url):
        try:
            special = '!@#$%^&*()[]{}|\\:;"\'<>,.?/~`'
            return sum(1 for c in str(url) if c in special)
        except:
            return 0
    features['SpecialCharCount'] = [count_special_chars(url) for url in urls]
    
    return pd.DataFrame(features)


# ==================== MAIN PROCESSING ====================

def process_dataset():
    """Process the dataset in batches for memory efficiency."""
    print("="*80)
    print("FAST URL-ONLY FEATURE EXTRACTION")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading dataset from: {CONFIG['input_file']}")
    start_time = time.time()
    
    # Read CSV info
    data = pd.read_csv(CONFIG['input_file'])
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"\nClass distribution:")
    print(data['type'].value_counts())
    
    total_rows = len(data)
    batch_size = CONFIG['batch_size']
    
    print(f"\n{'='*80}")
    print("EXTRACTING FEATURES")
    print(f"{'='*80}")
    print(f"Processing {total_rows} URLs in batches of {batch_size}")
    print(f"Output file: {CONFIG['output_file']}\n")
    
    # Process in batches
    all_results = []
    
    for batch_start in range(0, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        batch = data.iloc[batch_start:batch_end]
        
        # Extract features for this batch
        batch_features = extract_url_features_vectorized(batch['url'].values)
        
        # Add label
        batch_features['Label'] = batch['type'].values
        
        all_results.append(batch_features)
        
        # Progress update
        elapsed = time.time() - start_time
        processed = batch_end
        rate = processed / elapsed
        remaining = (total_rows - processed) / rate if rate > 0 else 0
        
        print(f"✓ Batch {batch_start//batch_size + 1}: Processed {processed:,}/{total_rows:,} URLs | "
              f"Rate: {rate:.0f} URLs/sec | Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s")
    
    # Combine all batches
    print(f"\n{'='*80}")
    print("COMBINING RESULTS")
    print(f"{'='*80}")
    
    final_df = pd.concat(all_results, ignore_index=True)
    
    print(f"\nFinal dataset shape: {final_df.shape}")
    print(f"Total features extracted: {len(final_df.columns) - 1}")  # Minus label column
    print(f"\nFeature columns:")
    for i, col in enumerate(final_df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Save to CSV
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    final_df.to_csv(CONFIG['output_file'], index=False)
    print(f"✓ Features saved to: {CONFIG['output_file']}")
    
    # Statistics
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETED")
    print(f"{'='*80}")
    print(f"Total URLs processed: {len(final_df):,}")
    print(f"Total features per URL: {len(final_df.columns) - 1}")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average rate: {len(final_df)/total_time:.0f} URLs/second")
    print(f"\nClass distribution in output:")
    print(final_df['Label'].value_counts())
    print(f"{'='*80}")
    
    return final_df


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    df = process_dataset()
    
    # Display sample
    print("\nSample of extracted features:")
    print(df.head(3).to_string())
