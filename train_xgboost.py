import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import xgboost as xgb


# ==================== CONFIGURATION ====================
CONFIG = {
    'data_file': 'fast_url_features.csv',
    'data_file_fallbacks': ['fast_url_html_features.csv'],
    'model_file': 'xgboost_phishing_model.json',
    'feature_importance_file': 'feature_importance.csv',
    'test_size': 0.2,
    'validation_size': 0.15,
    'random_state': 42,
    'model_params': {
        'n_estimators': 2500,  # Reduced from 2455 for faster training
        'max_depth': 6,  # Reduced from 8 for less complexity (fewer features)
        'tree_method': 'hist',
        'scale_pos_weight': 1,
        'learning_rate': 0.1,  # Increased from 0.05 for faster convergence
        'min_child_weight': 1,  # Reduced from 3 for better learning
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,  # Reduced from 0.1
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }
}


# ==================== FUNCTIONS ====================
def resolve_data_file(primary_file, fallback_files):
    """Use the first available feature file so extractor output stays compatible with training."""
    candidate_files = [primary_file] + fallback_files
    for candidate_file in candidate_files:
        if os.path.exists(candidate_file):
            if candidate_file != primary_file:
                print(f"Primary feature file not found. Using fallback: {candidate_file}")
            return candidate_file

    raise FileNotFoundError(
        f"None of the configured feature files exist: {candidate_files}"
    )


def load_data(file_path):
    """Load and display basic dataset information."""
    print("Loading feature dataset...")
    data = pd.read_csv(file_path, low_memory=False)

    if 'Label' not in data.columns:
        raise ValueError(
            "Feature dataset must contain a 'Label' column. "
            "Re-run fast_url_feature_extraction.py to regenerate the training file."
        )
    
    print(f"Dataset shape: {data.shape}")
    print(f"\nClass distribution:")
    print(data['Label'].value_counts())
    
    return data


def preprocess_features(X):
    """Handle data type conversions and missing values."""
    print(f"\nFeature data types:")
    print(X.dtypes.value_counts())
    
    # Convert boolean columns to integers
    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        print(f"Converting {len(bool_cols)} boolean columns to integers")
        X[bool_cols] = X[bool_cols].astype(int)
    
    # Handle object/string columns - convert if possible, then map common booleans
    object_cols = X.select_dtypes(include=['object', 'string']).columns
    if len(object_cols) > 0:
        print(f"Found {len(object_cols)} object/string columns: {object_cols.tolist()}")
        for col in object_cols:
            if X[col].dtype == 'object' or X[col].dtype == 'string':
                unique_vals = X[col].dropna().unique()
                print(f"  {col}: unique values = {unique_vals[:5]}")

                numeric_series = pd.to_numeric(X[col], errors='coerce')
                non_null_mask = X[col].notna()
                converted_ratio = (
                    numeric_series[non_null_mask].notna().mean()
                    if non_null_mask.any() else 1.0
                )

                if converted_ratio >= 0.95:
                    X[col] = numeric_series.fillna(0)
                    print(f"  -> Converted {col} to numeric ({converted_ratio:.1%} valid)")
                    continue

                normalized = X[col].astype(str).str.strip().str.lower()
                if normalized.dropna().isin(['true', 'false', '1', '0']).all():
                    X[col] = normalized.map({'true': 1, 'false': 0, '1': 1, '0': 0}).fillna(0)
                    print(f"  -> Converted {col} from boolean-like strings to int")
                else:
                    print(f"  -> Dropping {col} (cannot safely convert)")
                    X = X.drop(col, axis=1)
    
    # Handle missing values
    if X.isnull().any().any():
        missing_count = X.isnull().sum().sum()
        print(f"\nFilling {missing_count} missing values with 0")
        X = X.fillna(0)
    
    return X


def preprocess_labels(y):
    """Normalize labels to strict binary 0/1 for XGBoost compatibility."""
    print("\nNormalizing labels...")
    y = y.copy()

    if y.dtype == 'object' or str(y.dtype).startswith('string'):
        normalized = y.astype(str).str.strip().str.lower()
        label_map = {
            'legitimate': 0, 'benign': 0, 'safe': 0, 'normal': 0, '0': 0,
            'phishing': 1, 'malicious': 1, 'unsafe': 1, 'fraud': 1, '1': 1
        }
        mapped = normalized.map(label_map)
        if mapped.notna().all():
            y = mapped.astype(int)
            print("Mapped string labels to 0/1")
        else:
            raise ValueError(
                f"Unsupported string labels found: {sorted(set(normalized.unique()) - set(label_map.keys()))}"
            )

    y = pd.to_numeric(y, errors='raise').astype(int)
    unique_labels = sorted(y.unique().tolist())
    print(f"Label values before normalization: {unique_labels}")

    if unique_labels == [-1, 1]:
        y = y.map({-1: 0, 1: 1}).astype(int)
        print("Converted labels from -1/1 to 0/1")
    elif unique_labels != [0, 1]:
        raise ValueError(f"Expected binary labels [0, 1] or [-1, 1], got: {unique_labels}")

    return y


def split_data(X, y, test_size, validation_size, random_state):
    """Split data into train, validation, and test sets."""
    print(f"\nSplitting data (train/val/test)...")
    
    # First split: train+val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train / validation
    val_ratio = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train, X_val, y_val, model_params):
    """Train XGBoost model with early stopping."""
    print("\nTraining XGBoost model with early stopping...")
    
    # Add early stopping to model parameters
    model_params_with_es = model_params.copy()
    model_params_with_es['early_stopping_rounds'] = 15
    
    model = xgb.XGBClassifier(**model_params_with_es)
    
    # Train with validation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print(f"Best iteration: {model.best_iteration}")
    print(f"Best validation score: {model.best_score:.4f}")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and print detailed metrics."""
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # F1 Score
    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"  True Negatives:  {cm[0][0]:,}")
    print(f"  False Positives: {cm[0][1]:,}")
    print(f"  False Negatives: {cm[1][0]:,}")
    print(f"  True Positives:  {cm[1][1]:,}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return y_pred, y_pred_proba


def save_model_and_features(model, feature_names, model_file, importance_file):
    """Save trained model and feature importance."""
    # Save model
    model.save_model(model_file)
    print(f"\nModel saved to: {model_file}")
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20))
    
    feature_importance.to_csv(importance_file, index=False)
    print(f"Feature importance saved to: {importance_file}")
    
    return feature_importance


# ==================== MAIN EXECUTION ====================
def main():
    """Main training pipeline."""
    # Load data
    data_file = resolve_data_file(CONFIG['data_file'], CONFIG['data_file_fallbacks'])
    data = load_data(data_file)
    
    # Separate features and target
    X = data.drop('Label', axis=1)
    y = preprocess_labels(data['Label'])
    
    # Preprocess features
    X = preprocess_features(X)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, 
        CONFIG['test_size'], 
        CONFIG['validation_size'], 
        CONFIG['random_state']
    )
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val, CONFIG['model_params'])
    
    # Evaluate model
    y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # Save model and features
    feature_importance = save_model_and_features(
        model, 
        X.columns, 
        CONFIG['model_file'], 
        CONFIG['feature_importance_file']
    )
    
    print("\n" + "="*50)
    print("Training pipeline completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()
