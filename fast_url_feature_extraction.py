"""
Fast URL + HTML Feature Extraction
Extracts features from URLs and HTML files without making any network calls.
Processes URLs and their corresponding HTML files efficiently.
Fast URL + HTML Feature Extraction
Extracts features from URLs and HTML files without making any network calls.
Processes URLs and their corresponding HTML files efficiently.
"""

import os
import ipaddress
import re
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from urllib.parse import ParseResult, urlparse

import pandas as pd
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')


# ==================== CONFIGURATION ====================
# Detect lxml availability once at startup; fall back to built-in html.parser
try:
    import lxml  # noqa: F401
    _HTML_PARSER = 'lxml'
except ImportError:
    _HTML_PARSER = 'html.parser'

CONFIG = {
    'input_file': 'data_dropped.csv',
    'html_folder': 'dataset-folder/dataset/',  # Folder containing HTML files
    'output_file': 'fast_url_features.csv',
    'batch_size': 5000,   # Larger batches → less ProcessPool overhead per batch
    'use_html_parallel': True,
    'html_workers': max((os.cpu_count() or 2) - 1, 1),
    'html_parser': _HTML_PARSER,
    'html_chunksize': 100, # More files per worker task → less IPC overhead
}

# URL Shortening Services
SHORTENING_SERVICES = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                      r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                      r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                      r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|db\.tt|" \
                      r"qr\.ae|adf\.ly|bitly\.com|cur\.lv|ow\.ly|ity\.im|q\.gs|po\.st|bc\.vc|twitthis\.com|" \
                      r"u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|prettylinkpro\.com|scrnch\.me|" \
                      r"filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net"

# Pre-compiled patterns — built once at import time (no recompilation overhead per call)
_RE_SHORTENING    = re.compile(SHORTENING_SERVICES)
_RE_DISPLAY_NONE  = re.compile(r'display\s*:\s*none|visibility\s*:\s*hidden', re.IGNORECASE)
_RE_EMAIL         = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
_RE_TAGS          = re.compile(r'<[^>]+'+'>')       # strips HTML tags for text-length estimate
_SUSPICIOUS_URL_KW = frozenset(['login', 'signin', 'account', 'verify', 'update', 'secure',
                                  'banking', 'paypal', 'ebay', 'amazon', 'confirm', 'suspend',
                                  'reset', 'luck', 'bonus', 'free', 'click'])
_SUSPICIOUS_HTML_KW = ('verify', 'account', 'suspend', 'login', 'signin', 'update',
                        'confirm', 'secure', 'banking', 'click here', 'urgent',
                        'expired', 'verify your account', 'update your information')
_SOCIAL_KW   = frozenset(['facebook', 'twitter', 'instagram', 'linkedin', 'youtube'])
_SPECIAL_CHARS = frozenset('!@#$%^&*()[]{}|\\:;"\'\'<>,.?/~`')


# ==================== MODULE-LEVEL URL HELPER FUNCTIONS ====================
# Defined here (not inside the hot loop) so Python doesn't recreate them per batch call.

def _check_ip(netloc):
    """Return 1 if netloc is a bare IP address, else 0."""
    try:
        domain = netloc.split(':')[0]
        if not domain:
            return 0
        ipaddress.ip_address(domain)
        return 1
    except (ValueError, Exception):
        return 0


def _has_subdomain(netloc):
    """Return 1 if netloc has more than two dot-separated labels, else 0."""
    return 1 if len(netloc.split('.')) > 2 else 0


def _has_port(netloc):
    """Return 1 if netloc contains an explicit port number, else 0."""
    return 1 if ':' in netloc and netloc.split(':')[-1].isdigit() else 0


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
        try:
            soup = BeautifulSoup(html_content, CONFIG['html_parser'])
        except Exception:
            soup = BeautifulSoup(html_content, 'html.parser')

        # ── Single-pass tree traversal (replaces 15+ separate find_all calls) ─────
        tag_counter = Counter()
        forms, inputs, link_hrefs, scripts, link_tags = [], [], [], [], []
        title_text = None

        for tag in soup.find_all(True):
            name = tag.name
            tag_counter[name] += 1
            if name == 'form':
                forms.append(tag)
            elif name == 'input':
                inputs.append(tag)
            elif name == 'a':
                href = tag.get('href')
                if href is not None:
                    link_hrefs.append(href)
            elif name == 'script':
                scripts.append(tag)
            elif name == 'link':
                link_tags.append(tag)
            elif name == 'title' and title_text is None:
                title_text = tag.string  # capture first <title> in the same pass
        # ─────────────────────────────────────────────────────────────────────────

        # 1. Form features
        features['NoOfForms'] = len(forms)

        # 2. Input fields
        features['NoOfInputs'] = len(inputs)
        features['NoOfPasswordFields'] = sum(1 for i in inputs if i.get('type') == 'password')
        features['NoOfEmailFields']    = sum(1 for i in inputs
                                             if i.get('type') == 'email'
                                             or 'email' in str(i.get('name', '')).lower())
        features['NoOfHiddenFields']   = sum(1 for i in inputs if i.get('type') == 'hidden')

        # 3. Link features
        features['NoOfLinks'] = len(link_hrefs)
        external_links = sum(1 for h in link_hrefs if h.startswith(('http://', 'https://')))
        null_links      = sum(1 for h in link_hrefs if h.startswith('#') or h == '' or h.startswith('javascript:'))
        internal_links  = len(link_hrefs) - external_links - null_links
        features['NoOfExternalLinks'] = external_links
        features['NoOfInternalLinks'] = internal_links
        features['NoOfNullLinks']     = null_links
        features['ExternalLinkRatio'] = external_links / max(len(link_hrefs), 1)

        # 4. Form action analysis
        features['NoOfExternalFormActions'] = sum(
            1 for form in forms
            if form.get('action', '').startswith(('http://', 'https://'))
        )

        # 5. Script tags
        features['NoOfScripts']       = len(scripts)
        features['NoOfInlineScripts'] = sum(1 for s in scripts if s.string and s.string.strip())

        # 6-8. Counts from the tag counter (no extra traversal needed)
        features['NoOfIframes']  = tag_counter['iframe']
        features['NoOfImages']   = tag_counter['img']
        features['NoOfMetaTags'] = tag_counter['meta']
        features['NoOfButtons']  = tag_counter['button']
        features['NoOfTextareas']= tag_counter['textarea']

        # 9. Title — already captured in the single pass above
        features['HasTitle']    = 1 if title_text else 0
        features['TitleLength'] = len(title_text) if title_text else 0

        # 10. Suspicious keywords — search raw lowercased HTML (avoids soup.get_text() tree walk)
        html_lower = html_content.lower()
        features['SuspiciousKeywordsCount'] = sum(1 for kw in _SUSPICIOUS_HTML_KW if kw in html_lower)

        # 11. CSS
        features['NoOfStyleTags'] = tag_counter['style']
        features['NoOfExternalCSS'] = sum(
            1 for lt in link_tags
            if 'stylesheet' in (
                ' '.join(lt.get('rel', [])) if isinstance(lt.get('rel'), list)
                else str(lt.get('rel', ''))
            )
        )

        # 12. Obfuscation & pop-up — use pre-compiled patterns on raw HTML
        features['HasObfuscation'] = 1 if _RE_DISPLAY_NONE.search(html_content) else 0
        features['HasPopup']       = 1 if 'window.open' in html_lower or 'alert(' in html_lower else 0

        # 13. Document length — strip tags with pre-compiled regex (faster than soup.get_text())
        text_approx = _RE_TAGS.sub(' ', html_lower)
        features['HTMLLength']      = len(html_content)
        features['TextLength']      = len(text_approx)
        features['TextToHTMLRatio'] = len(text_approx) / max(len(html_content), 1)

        # 14. Has favicon (check <link> tags already collected)
        has_favicon = False
        for lt in link_tags:
            rel = lt.get('rel', [])
            rel_str = (' '.join(rel) if isinstance(rel, list) else str(rel)).lower()
            if 'icon' in rel_str:
                has_favicon = True
                break
        features['HasFavicon'] = 1 if has_favicon else 0

        # 15. Social media links
        features['NoOfSocialLinks'] = sum(
            1 for h in link_hrefs if any(s in h.lower() for s in _SOCIAL_KW)
        )

        # 16. Email addresses in HTML — use pre-compiled pattern
        features['NoOfEmailsInHTML'] = len(_RE_EMAIL.findall(html_content))

    except Exception:
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


def resolve_dataset_schema(data):
    """Identify label and HTML columns for supported dataset layouts."""
    if 'url' not in data.columns:
        raise ValueError("Input dataset must contain a 'url' column.")

    if 'result' in data.columns:
        label_column = 'result'
    elif 'type' in data.columns:
        label_column = 'type'
    else:
        raise ValueError("Input dataset must contain either a 'result' or 'type' column.")

    html_column = 'website' if 'website' in data.columns else None
    return label_column, html_column


def extract_html_features_parallel(html_paths, max_workers, executor=None, chunksize=1):
    """Extract HTML features across multiple processes while preserving input order."""
    if not html_paths:
        return []

    if max_workers <= 1:
        return [extract_html_features(path) for path in html_paths]

    if executor is not None:
        return list(executor.map(extract_html_features, html_paths, chunksize=chunksize))

    with ProcessPoolExecutor(max_workers=max_workers) as local_executor:
        return list(local_executor.map(extract_html_features, html_paths, chunksize=chunksize))


# ==================== URL FEATURE EXTRACTION FUNCTIONS ====================

def extract_url_features_vectorized(urls):
    """
    Extract features from URLs using vectorized operations for speed.
    Returns a DataFrame with all extracted features.
    """
    n = len(urls)
    features = {}

    print(f"Extracting features from {n} URLs...")

    # Convert all URLs to str once — avoids repeated str(url) calls in every feature below
    url_strs = [str(u) for u in urls]

    # Parse all URLs at once - with error handling for malformed URLs
    parsed_urls = []
    for url in url_strs:
        try:
            parsed_urls.append(urlparse(url))
        except (ValueError, Exception):
            parsed_urls.append(ParseResult('', '', '', '', '', ''))
    
    # 1. DomainPartCount - Number of dot-separated labels in domain (replaces raw Domain string
    #    which cannot be used directly by XGBoost and would be silently dropped at training time)
    features['DomainPartCount'] = [len(parsed.netloc.split('.')) for parsed in parsed_urls]
    
    # 2. Have_IP - Check if domain is IP address
    features['Have_IP'] = [_check_ip(parsed.netloc) for parsed in parsed_urls]
    
    # 3. Have_At - '@' symbol in URL
    features['Have_At'] = [1 if '@' in s else 0 for s in url_strs]

    # 4. URL_Length - Length of URL
    features['URL_Length'] = [len(s) for s in url_strs]

    # 5. URL_Depth - Number of subdirectories
    features['URL_Depth'] = [len([x for x in parsed.path.split('/') if x]) for parsed in parsed_urls]

    # 6. Redirection - '//' after protocol
    features['Redirection'] = [1 if s.rfind('//') > 7 else 0 for s in url_strs]

    # 7. https_Domain - 'http' or 'https' in domain part
    features['https_Domain'] = [1 if 'http' in parsed.netloc.lower() else 0 for parsed in parsed_urls]

    # 8. TinyURL - URL shortening service (pre-compiled pattern)
    features['TinyURL'] = [1 if _RE_SHORTENING.search(s) else 0 for s in url_strs]
    
    # 9. Prefix/Suffix - '-' in domain
    features['Prefix_Suffix'] = [1 if '-' in parsed.netloc else 0 for parsed in parsed_urls]
    
    # 10. IsHTTPS - Uses HTTPS protocol
    features['IsHTTPS'] = [1 if parsed.scheme == 'https' else 0 for parsed in parsed_urls]
    
    # 11-18. Single-character counts — use pre-converted url_strs
    features['NoOfDots']         = [s.count('.')  for s in url_strs]
    features['NoOfHyphen']       = [s.count('-')  for s in url_strs]
    features['NoOfUnderscore']   = [s.count('_')  for s in url_strs]
    features['NoOfSlash']        = [s.count('/')  for s in url_strs]
    features['NoOfQuestionMark'] = [s.count('?')  for s in url_strs]
    features['NoOfEquals']       = [s.count('=')  for s in url_strs]
    features['NoOfAmpersand']    = [s.count('&')  for s in url_strs]
    features['NoOfPercent']      = [s.count('%')  for s in url_strs]

    # 19. NoOfDigits
    features['NoOfDigits']  = [sum(c.isdigit() for c in s) for s in url_strs]

    # 20. NoOfLetters
    features['NoOfLetters'] = [sum(c.isalpha() for c in s) for s in url_strs]
    
    # 21. LetterRatio - Ratio of letters to total length
    features['LetterRatio'] = [l / max(u, 1)
                               for l, u in zip(features['NoOfLetters'], features['URL_Length'])]

    # 22. DigitRatio - Ratio of digits to total length
    features['DigitRatio'] = [d / max(u, 1)
                              for d, u in zip(features['NoOfDigits'], features['URL_Length'])]

    # 23. HasSubdomain - Has subdomain (more than domain.tld)
    features['HasSubdomain'] = [_has_subdomain(parsed.netloc) for parsed in parsed_urls]

    # 24. SubdomainCount - Number of subdomains (reuses DomainPartCount to avoid double split)
    features['SubdomainCount'] = [max(p - 2, 0) for p in features['DomainPartCount']]
    
    # 25. DomainLength - Length of domain
    features['DomainLength'] = [len(parsed.netloc) for parsed in parsed_urls]
    
    # 26. PathLength - Length of path
    features['PathLength'] = [len(parsed.path) for parsed in parsed_urls]
    
    # 27. QueryLength - Length of query string
    features['QueryLength'] = [len(parsed.query) for parsed in parsed_urls]
    
    # 28. HasPort - URL has port number
    features['HasPort'] = [_has_port(parsed.netloc) for parsed in parsed_urls]
    
    # 29. SuspiciousWords — use module-level frozenset and pre-converted strings
    features['SuspiciousWords'] = [
        1 if any(kw in s.lower() for kw in _SUSPICIOUS_URL_KW) else 0
        for s in url_strs
    ]

    # 30. SpecialCharCount
    features['SpecialCharCount'] = [sum(1 for c in s if c in _SPECIAL_CHARS) for s in url_strs]
    
    return pd.DataFrame(features)


# ==================== MAIN PROCESSING ====================

def _already_processed_rows(output_file):
    """Return how many rows are already written to the output file (0 if it doesn't exist).

    Uses pandas to count rows so embedded newlines inside quoted CSV fields
    (e.g. in unusual URLs) don't cause miscounting.
    """
    if not os.path.exists(output_file):
        return 0
    try:
        return max(len(pd.read_csv(output_file, usecols=[0])), 0)
    except Exception:
        return 0


def process_dataset():
    """
    Process the dataset in batches.
    - Each batch is appended to the output file immediately after extraction.
    - On restart the script reads the output file, counts already-processed rows,
      and skips straight to the next unprocessed batch (checkpoint / resume).
    """
    print("=" * 80)
    print("FAST URL + HTML FEATURE EXTRACTION")
    print("=" * 80)

    # Early path validation — fail fast with a clear message
    if not os.path.exists(CONFIG['input_file']):
        raise FileNotFoundError(
            f"Input file not found: '{CONFIG['input_file']}'. "
            "Update CONFIG['input_file'] or run from the correct directory."
        )

    # Load dataset
    print(f"\nLoading dataset from: {CONFIG['input_file']}")
    start_time = time.time()

    data = pd.read_csv(CONFIG['input_file'])
    label_column, html_column = resolve_dataset_schema(data)
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"\nClass distribution:")
    print(data[label_column].value_counts())

    total_rows = len(data)
    batch_size = CONFIG['batch_size']
    output_file = CONFIG['output_file']

    # ── Checkpoint: find out how many rows are already done ──────────────────
    already_done = _already_processed_rows(output_file)
    if already_done > 0:
        print(f"\n⚡ Resuming: {already_done:,} rows already written to '{output_file}'.")
        print(f"   Skipping to row {already_done:,} ...")
    else:
        print(f"\n🆕 Starting fresh — output file will be created at '{output_file}'.")

    remaining_rows = total_rows - already_done
    if remaining_rows <= 0:
        print("\n✅ All rows already processed. Nothing to do.")
        return pd.read_csv(output_file)
    # ─────────────────────────────────────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print("EXTRACTING FEATURES")
    print(f"{'=' * 80}")
    print(f"Processing {remaining_rows:,} remaining URLs "
          f"({total_rows:,} total) in batches of {batch_size:,}")
    print(f"Output file: {output_file}\n")

    html_folder = CONFIG['html_folder']
    use_html_parallel = CONFIG['use_html_parallel'] and html_column is not None
    html_workers = CONFIG['html_workers']
    html_chunksize = CONFIG['html_chunksize']

    if use_html_parallel:
        print(f"HTML extraction mode: parallel ({html_workers} workers, chunksize={html_chunksize})")
    elif html_column is not None:
        print("HTML extraction mode: sequential")
    else:
        print("HTML extraction mode: disabled (no HTML column found)")

    executor = ProcessPoolExecutor(max_workers=html_workers) if use_html_parallel else None
    batches_done = 0

    try:
        for batch_start in range(already_done, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            batch = data.iloc[batch_start:batch_end]
            batches_done += 1

            # ── URL features ─────────────────────────────────────────────────
            batch_features = extract_url_features_vectorized(batch['url'].values)

            # ── HTML features ─────────────────────────────────────────────────
            if html_column is not None:
                print("  Extracting HTML features...")
                html_paths = [
                    os.path.join(html_folder, fn)
                    for fn in batch[html_column].tolist()
                ]
                html_features_list = extract_html_features_parallel(
                    html_paths, html_workers,
                    executor=executor, chunksize=html_chunksize,
                )
                batch_features = pd.concat(
                    [batch_features, pd.DataFrame(html_features_list)], axis=1
                )

            # ── Label ─────────────────────────────────────────────────────────
            batch_features['Label'] = batch[label_column].values

            # ── Write / append to output file ────────────────────────────────
            # Use mode='w' (overwrite) for the very first batch of a fresh run so
            # a stale header-only file from a previous interrupted run is replaced
            # cleanly.  All subsequent batches append without a header.
            is_first_fresh_batch = (already_done == 0 and batch_start == 0)
            write_mode   = 'w' if is_first_fresh_batch else 'a'
            write_header = is_first_fresh_batch
            batch_features.to_csv(output_file, mode=write_mode, index=False, header=write_header)

            # ── Progress ──────────────────────────────────────────────────────
            elapsed = time.time() - start_time
            processed_this_run = batch_end - already_done
            rate = processed_this_run / elapsed if elapsed > 0 else 0
            still_remaining = total_rows - batch_end
            eta = still_remaining / rate if rate > 0 else 0

            print(
                f"✓ Batch {batches_done}: rows {batch_start:,}–{batch_end:,} | "
                f"Total done: {batch_end:,}/{total_rows:,} | "
                f"Rate: {rate:.0f} rows/sec | "
                f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s"
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    rows_this_run = total_rows - already_done
    print(f"\n{'=' * 80}")
    print("EXTRACTION COMPLETED")
    print(f"{'=' * 80}")
    print(f"Rows processed this run : {rows_this_run:,}")
    print(f"Total rows in output    : {total_rows:,}")
    print(f"Total time              : {total_time:.2f}s ({total_time / 60:.2f} min)")
    if rows_this_run > 0:
        print(f"Average rate            : {rows_this_run / total_time:.0f} rows/sec")
    print(f"Output file             : {output_file}")
    print(f"{'=' * 80}")

    # Return the full output for convenience
    final_df = pd.read_csv(output_file)
    print(f"\nClass distribution in output:")
    print(final_df['Label'].value_counts())
    return final_df


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    df = process_dataset()
    
    # Display sample
    print("\nSample of extracted features:")
    print(df.head(3).to_string())
