"""
Fast URL + HTML Feature Extraction
Extracts features from URLs and HTML files without making any network calls.
Processes URLs and their corresponding HTML files efficiently.
"""

import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
import ipaddress
import time
import warnings
import os
from bs4 import BeautifulSoup
from pathlib import Path
warnings.filterwarnings('ignore')


# ==================== CONFIGURATION ====================
CONFIG = {
    'input_file': 'dataset-folder/dataset/data_dropped.csv',
    'html_folder': 'dataset-folder/dataset/',  # Folder containing HTML files
    'output_file': 'fast_url_html_features.csv',
    'batch_size': 1000,  # Process in batches for memory efficiency
}

# URL Shortening Services
SHORTENING_SERVICES = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                      r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                      r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                      r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|db\.tt|" \
                      r"qr\.ae|adf\.ly|bitly\.com|cur\.lv|ow\.ly|ity\.im|q\.gs|po\.st|bc\.vc|twitthis\.com|" \
                      r"u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|prettylinkpro\.com|scrnch\.me|" \
                      r"filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net"


# ==================== HTML FEATURE EXTRACTION FUNCTIONS ====================

def extract_html_features(html_path):
    """
    Extract features from HTML file.
    Returns a dictionary with all extracted HTML features.
    """
    features = {}
    
    # Check if file exists
    if not os.path.exists(html_path):
        # Return default values for missing files
        return get_default_html_features()
    
    try:
        # Read HTML file
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 1. Form features
        forms = soup.find_all('form')
        features['NoOfForms'] = len(forms)
        
        # 2. Input fields
        inputs = soup.find_all('input')
        features['NoOfInputs'] = len(inputs)
        features['NoOfPasswordFields'] = len([i for i in inputs if i.get('type') == 'password'])
        features['NoOfEmailFields'] = len([i for i in inputs if i.get('type') == 'email' or 'email' in str(i.get('name', '')).lower()])
        features['NoOfHiddenFields'] = len([i for i in inputs if i.get('type') == 'hidden'])
        
        # 3. Link features
        links = soup.find_all('a', href=True)
        features['NoOfLinks'] = len(links)
        
        # Count external vs internal links
        external_links = 0
        internal_links = 0
        null_links = 0
        
        for link in links:
            href = link.get('href', '')
            if href.startswith('http://') or href.startswith('https://'):
                external_links += 1
            elif href.startswith('#') or href == '' or href.startswith('javascript:'):
                null_links += 1
            else:
                internal_links += 1
        
        features['NoOfExternalLinks'] = external_links
        features['NoOfInternalLinks'] = internal_links
        features['NoOfNullLinks'] = null_links
        features['ExternalLinkRatio'] = external_links / max(len(links), 1)
        
        # 4. Form action analysis
        external_form_actions = 0
        for form in forms:
            action = form.get('action', '')
            if action.startswith('http://') or action.startswith('https://'):
                external_form_actions += 1
        features['NoOfExternalFormActions'] = external_form_actions
        
        # 5. Script tags
        scripts = soup.find_all('script')
        features['NoOfScripts'] = len(scripts)
        features['NoOfInlineScripts'] = len([s for s in scripts if s.string and len(s.string.strip()) > 0])
        
        # 6. IFrame tags
        iframes = soup.find_all('iframe')
        features['NoOfIframes'] = len(iframes)
        
        # 7. Image tags
        images = soup.find_all('img')
        features['NoOfImages'] = len(images)
        
        # 8. Meta tags
        meta_tags = soup.find_all('meta')
        features['NoOfMetaTags'] = len(meta_tags)
        
        # 9. Title
        title = soup.find('title')
        features['HasTitle'] = 1 if title and title.string else 0
        features['TitleLength'] = len(title.string) if title and title.string else 0
        
        # 10. Suspicious keywords in HTML
        html_text = soup.get_text().lower()
        suspicious_keywords = ['verify', 'account', 'suspend', 'login', 'signin', 'update', 
                              'confirm', 'secure', 'banking', 'click here', 'urgent', 
                              'expired', 'verify your account', 'update your information']
        features['SuspiciousKeywordsCount'] = sum(1 for keyword in suspicious_keywords if keyword in html_text)
        
        # 11. Button tags
        buttons = soup.find_all('button')
        features['NoOfButtons'] = len(buttons)
        
        # 12. Text fields
        textareas = soup.find_all('textarea')
        features['NoOfTextareas'] = len(textareas)
        
        # 13. CSS analysis
        styles = soup.find_all('style')
        features['NoOfStyleTags'] = len(styles)
        
        # 14. External CSS
        css_links = soup.find_all('link', rel='stylesheet')
        features['NoOfExternalCSS'] = len(css_links)
        
        # 15. Obfuscation indicators
        features['HasObfuscation'] = 1 if ('display:none' in html_content.lower() or 
                                            'visibility:hidden' in html_content.lower()) else 0
        
        # 16. Pop-up indicators
        features['HasPopup'] = 1 if ('window.open' in html_content.lower() or 
                                      'alert(' in html_content.lower()) else 0
        
        # 17. Document length
        features['HTMLLength'] = len(html_content)
        features['TextLength'] = len(html_text)
        features['TextToHTMLRatio'] = len(html_text) / max(len(html_content), 1)
        
        # 18. Has favicon
        favicon = soup.find('link', rel=lambda x: x and 'icon' in x.lower()) if soup.find('link') else None
        features['HasFavicon'] = 1 if favicon else 0
        
        # 19. Social media links
        social_keywords = ['facebook', 'twitter', 'instagram', 'linkedin', 'youtube']
        features['NoOfSocialLinks'] = sum(1 for link in links 
                                          if any(social in link.get('href', '').lower() 
                                                for social in social_keywords))
        
        # 20. Email in HTML
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        features['NoOfEmailsInHTML'] = len(re.findall(email_pattern, html_content))
        
    except Exception as e:
        # If any error occurs, return default features
        return get_default_html_features()
    
    return features


def get_default_html_features():
    """Return default HTML features when file is missing or error occurs."""
    return {
        'NoOfForms': 0, 'NoOfInputs': 0, 'NoOfPasswordFields': 0, 'NoOfEmailFields': 0,
        'NoOfHiddenFields': 0, 'NoOfLinks': 0, 'NoOfExternalLinks': 0, 'NoOfInternalLinks': 0,
        'NoOfNullLinks': 0, 'ExternalLinkRatio': 0, 'NoOfExternalFormActions': 0,
        'NoOfScripts': 0, 'NoOfInlineScripts': 0, 'NoOfIframes': 0, 'NoOfImages': 0,
        'NoOfMetaTags': 0, 'HasTitle': 0, 'TitleLength': 0, 'SuspiciousKeywordsCount': 0,
        'NoOfButtons': 0, 'NoOfTextareas': 0, 'NoOfStyleTags': 0, 'NoOfExternalCSS': 0,
        'HasObfuscation': 0, 'HasPopup': 0, 'HTMLLength': 0, 'TextLength': 0,
        'TextToHTMLRatio': 0, 'HasFavicon': 0, 'NoOfSocialLinks': 0, 'NoOfEmailsInHTML': 0
    }


# ==================== URL FEATURE EXTRACTION FUNCTIONS ====================

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
    print("FAST URL + HTML FEATURE EXTRACTION")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading dataset from: {CONFIG['input_file']}")
    start_time = time.time()
    
    # Read CSV info
    data = pd.read_csv(CONFIG['input_file'])
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"\nClass distribution:")
    print(data['result'].value_counts())
    
    total_rows = len(data)
    batch_size = CONFIG['batch_size']
    
    print(f"\n{'='*80}")
    print("EXTRACTING FEATURES")
    print(f"{'='*80}")
    print(f"Processing {total_rows} URLs in batches of {batch_size}")
    print(f"Output file: {CONFIG['output_file']}\n")
    
    # Process in batches
    all_results = []
    html_folder = CONFIG['html_folder']
    
    for batch_start in range(0, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        batch = data.iloc[batch_start:batch_end]
        
        # Extract URL features for this batch
        batch_features = extract_url_features_vectorized(batch['url'].values)
        
        # Extract HTML features for this batch
        print(f"  Extracting HTML features...")
        html_features_list = []
        for idx, row in batch.iterrows():
            html_filename = row['website']
            html_path = os.path.join(html_folder, html_filename)
            html_features = extract_html_features(html_path)
            html_features_list.append(html_features)
        
        # Combine URL and HTML features
        html_features_df = pd.DataFrame(html_features_list)
        batch_features = pd.concat([batch_features, html_features_df], axis=1)
        
        # Add label
        batch_features['Label'] = batch['result'].values
        
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
