# =============================================================
# data_loader.py  —  load, clean, impute, validate
# Updated to support 22-column CSV with Name and Roll No.
# =============================================================
import pandas as pd
import numpy as np
import re
from config import DATA_PATH, ORDINAL_COLS, JAIN_VARIANTS

# Total 24 elements in COLUMN_NAMES to ensure safety with extra CSV columns
# Matches: Timestamp, Name, Roll No, Gender, and 16 lifestyle features + 3 dealbreakers + 1 optional
COLUMN_NAMES = [
    "timestamp", "name", "roll_no", "gender",
    "sleep_time", "wake_time", "late_return",
    "cleanliness", "evening_pref", "guest_leaves", "convo_level",
    "study_env", "music_bother", "food_pref", "nonveg_sensitivity",
    "smoke_drink", "conflict_style", "sharing", "temp_pref", "light_pref",
    "dealbreaker_1", "dealbreaker_2", "dealbreaker_3", "optional"
]

CAT_COLS = [col for col, _ in ORDINAL_COLS] + ["food_pref", "study_env"]

def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    # 1. Read CSV
    df = pd.read_csv(path)
    
    # 2. FIXED REGEX: This pattern safely removes tags
    # The double backslash or r'\[...\]' is required to escape square brackets
    df.columns = [re.sub(r'\[.*?\]', '', str(c)).strip() for c in df.columns]

    # 3. Filter out empty rows
    df = df[df.iloc[:, 0].notna()].copy().reset_index(drop=True)
    
    # 4. Map columns safely (Supporting your new 22-column structure)
    if len(df.columns) >= len(COLUMN_NAMES):
        df = df.iloc[:, :len(COLUMN_NAMES)]
        df.columns = COLUMN_NAMES
    else:
        # Fallback for shorter CSVs
        df.columns = COLUMN_NAMES[:len(df.columns)]
        print(f"⚠ Warning: CSV has only {len(df.columns)} columns.")

    # 5. Drop non-feature columns (timestamp, optional, and dealbreakers)
    # Keeping name, roll_no, and gender for identification and split logic
    drop_cols = [c for c in ["timestamp", "optional", "dealbreaker_1", "dealbreaker_2", "dealbreaker_3"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize Jain variants from config.py
    df["food_pref"] = df["food_pref"].replace(JAIN_VARIANTS, "Jain (no root vegetables)")
    
    # Numeric conversion for scale-based features
    if "music_bother" in df.columns and df["music_bother"].dtype == object:
        df["music_bother"] = df["music_bother"].astype(str).str.extract(r"(\d)")[0]
    
    df["convo_level"]  = pd.to_numeric(df["convo_level"],  errors="coerce")
    df["music_bother"] = pd.to_numeric(df["music_bother"], errors="coerce")
    
    # Strip whitespace from identity string fields
    for col in ["name", "roll_no", "gender"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

def impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Categorical imputer
    for col in CAT_COLS:
        if col in df.columns:
            mode_res = df[col].mode()
            if not mode_res.empty:
                df[col] = df[col].fillna(mode_res[0])
                
    # Numeric imputer
    for col in ["convo_level", "music_bother"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            
    # FIXED: Added .str before .lower()
    gender_mode = df["gender"].mode()[0] if "gender" in df.columns else "Male"
    df["gender_imputed"] = df["gender"].isna() | (df["gender"].astype(str).str.lower() == "nan")
    df["gender"] = df["gender"].replace("nan", gender_mode).fillna(gender_mode)
    return df

def validate(df: pd.DataFrame) -> None:
    bad = False
    # Validate against orders defined in config.py
    for col, order in ORDINAL_COLS:
        if col not in df.columns:
            continue
        unexpected = set(df[col].dropna().unique()) - set(order)
        if unexpected:
            print(f"  ⚠  Column '{col}' has unexpected values: {unexpected}")
            bad = True
    
    if bad:
        print("❌ Validation Failed. Check your CSV vs config.py mappings.")
        raise ValueError("Fix unexpected category values before continuing.")
    print("✅ Validation passed")

def load_and_prepare(path: str = DATA_PATH) -> pd.DataFrame:
    print(f"── STAGE 1: Load from {path} ──")
    df = load_raw(path)
    df = clean(df)
    df = impute(df)
    validate(df)
    
    # Identity Logic to ensure every user has a unique student_key
    if "roll_no" not in df.columns or df["roll_no"].iloc[0] == "nan":
        df["roll_no"] = [f"TEMP_{i}" for i in range(len(df))]
    
    if "name" not in df.columns or df["name"].iloc[0] == "nan":
        df["name"] = "Student"
        
    df["student_key"] = df["roll_no"].astype(str).str.upper().str.strip()
    df["display_name"] = df["name"].astype(str) + " (" + df["student_key"] + ")"
    
    df = df.reset_index(drop=True)
    print(
        f"✅ {len(df)} responses processed | "
        f"Female={(df['gender']=='Female').sum()} | "
        f"Male={(df['gender']=='Male').sum()}"
    )
    return df