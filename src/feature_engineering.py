import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def extract_features(df, is_train=True, encoders=None):
    """
    Extracts leakage-free triage features available at ticket inception.
    """
    X = pd.DataFrame(index=df.index)
    
    # 1. Temporal cyclical features
    hour = df['Open_Hour'].fillna(12)
    X['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    X['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    
    dow = df['Open_DayOfWeek'].fillna(0)
    X['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
    X['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)
    
    X['is_weekend'] = df['Is_Weekend'].fillna(0).astype(int)
    X['month'] = df['Open_Month'].fillna(1).astype(int)
    
    # 2. Number of related customer interactions at opening
    X['interactions_cnt'] = df['No_of_Related_Interactions_Clean'].fillna(0).astype(float)
    X['number_cnt'] = df['number_cnt'].fillna(0.5).astype(float)
    
    # 3. Categorical encoding & frequencies
    if is_train:
        encoders = {}
        # Frequency encodings
        for col in ['CI_Name', 'CI_Subcat', 'WBS', 'KB_number']:
            freq = df[col].value_counts(normalize=True).to_dict()
            encoders[f"{col}_freq"] = freq
            X[f"{col}_freq"] = df[col].map(freq).fillna(0.0).astype(float)
            
        # Top categories for CI_Cat and Category
        top_ci_cats = df['CI_Cat'].value_counts().head(8).index.tolist()
        top_categories = df['Category'].value_counts().head(3).index.tolist()
        encoders['top_ci_cats'] = top_ci_cats
        encoders['top_categories'] = top_categories
        
        for c in top_ci_cats:
            X[f"ci_cat_{c}"] = (df['CI_Cat'] == c).astype(int)
        for c in top_categories:
            clean_c = c.replace(' ', '_')
            X[f"cat_{clean_c}"] = (df['Category'] == c).astype(int)
            
        return X, encoders
    else:
        # Transform using saved encoders
        for col in ['CI_Name', 'CI_Subcat', 'WBS', 'KB_number']:
            freq = encoders.get(f"{col}_freq", {})
            X[f"{col}_freq"] = df[col].map(freq).fillna(0.0).astype(float)
            
        top_ci_cats = encoders.get('top_ci_cats', [])
        top_categories = encoders.get('top_categories', [])
        
        for c in top_ci_cats:
            X[f"ci_cat_{c}"] = (df['CI_Cat'] == c).astype(int)
        for c in top_categories:
            clean_c = c.replace(' ', '_')
            X[f"cat_{clean_c}"] = (df['Category'] == c).astype(int)
            
        return X

if __name__ == "__main__":
    from src.config import CLEANED_DATA_PATH
    df = pd.read_csv(CLEANED_DATA_PATH)
    X, enc = extract_features(df, is_train=True)
    print("Feature matrix shape:", X.shape)
    print("Features extracted:", list(X.columns))
