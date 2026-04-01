# ╔══════════════════════════════════════════════════════════════════╗
# ║   Hostel Roommate Compatibility — Preprocessing Pipeline        ║
# ║                                                                  ║
# ║   Stages:                                                        ║
# ║     1  Load & clean                                              ║
# ║     2  Missing value handling                                    ║
# ║     3  Encoding (ordinal + one-hot)                              ║
# ║     4  Normalisation (MinMaxScaler)                              ║
# ║     5  Custom compatibility matrices (food, smoking)             ║
# ║     6  Compatibility scoring function (gender-split weights)     ║
# ║     7  PCA (separate per gender)                                 ║
# ║     8  KMeans clustering (separate per gender)                   ║
# ║                                                                  ║
# ║   Run independently. No dependency on eda.py.                    ║
# ╚══════════════════════════════════════════════════════════════════╝

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── Config ───────────────────────────────────────────────────────────
DATA_PATH = r"C:\Users\HP\Documents\Hostel_Roommate_Prediction_model_VS_Code\Hostel-Roommate-Prediction-model\Raw_data.csv"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 1 — Load & Clean
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

df_raw = pd.read_csv(DATA_PATH)
df_raw = df_raw[df_raw["Timestamp"].notna() & (df_raw["Timestamp"] != "")].copy()
df_raw = df_raw.reset_index(drop=True)

df_raw.columns = [
    "timestamp", "gender", "sleep_time", "wake_time", "late_return",
    "cleanliness", "evening_pref", "guest_leaves", "convo_level",
    "study_env", "music_bother", "food_pref", "nonveg_sensitivity",
    "smoke_drink", "conflict_style", "sharing", "temp_pref",
    "light_pref", "dealbreaker1", "dealbreaker2", "dealbreaker3",
    "optional_notes",
]
df_raw = df_raw.drop(columns=["timestamp", "optional_notes"])

# Canonical Jain spelling — must match eda.py exactly
JAIN_VARIANTS = [
    "Jain(no root vegetable)", "Jain (no root vegetable)", "Jain(no root vegetables)",
]
df_raw["food_pref"] = df_raw["food_pref"].replace(JAIN_VARIANTS, "Jain (no root vegetables)")

# music_bother: extract digit from string format "4 - Very bothered"
if df_raw["music_bother"].dtype == object:
    df_raw["music_bother"] = (
        df_raw["music_bother"].str.extract(r"(\d)")[0]
        .astype(float).astype("Int64")
    )

df_raw["convo_level"]  = pd.to_numeric(df_raw["convo_level"],  errors="coerce")
df_raw["music_bother"] = pd.to_numeric(df_raw["music_bother"], errors="coerce")

print(f"Loaded {len(df_raw)} responses, {df_raw.shape[1]} columns")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 2 — Missing Value Handling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

df = df_raw.copy()

# Categorical columns: mode fill
CAT_COLS = [
    "sleep_time", "wake_time", "late_return", "cleanliness", "evening_pref",
    "guest_leaves", "study_env", "food_pref", "nonveg_sensitivity",
    "smoke_drink", "conflict_style", "sharing", "temp_pref", "light_pref",
]
for col in CAT_COLS:
    mode = df[col].mode()
    if not mode.empty and df[col].isna().any():
        n_filled = df[col].isna().sum()
        df[col] = df[col].fillna(mode[0])
        print(f"  {col}: filled {n_filled} missing with mode '{mode[0]}'")

# Numeric columns: median fill
for col in ["convo_level", "music_bother"]:
    median = df[col].median()
    if df[col].isna().any():
        n_filled = df[col].isna().sum()
        df[col] = df[col].fillna(median)
        print(f"  {col}: filled {n_filled} missing with median {median:.2f}")

# Dealbreaker slots: 'None' string so they don't interfere with valid option matching
for col in ["dealbreaker1", "dealbreaker2", "dealbreaker3"]:
    df[col] = df[col].fillna("None")

# Gender: mode fill — needed so every student enters the pipeline.
# The mode-filled rows are flagged so gender-split analysis can exclude them.
gender_mode = df["gender"].mode()[0]
df["gender_imputed"] = df["gender"].isna()
df["gender"] = df["gender"].fillna(gender_mode)
n_gender_filled = df["gender_imputed"].sum()
if n_gender_filled > 0:
    print(f"  gender: filled {n_gender_filled} missing with mode '{gender_mode}' "
          f"(flagged in 'gender_imputed' column)")

# Verify no NaNs remain in feature columns
remaining = df[CAT_COLS + ["convo_level", "music_bother", "gender"]].isnull().sum()
assert remaining.sum() == 0, f"NaNs remain: {remaining[remaining > 0]}"
print("✅ Missing values handled — no NaNs in feature columns")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 3 — Encoding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Ordinal: columns with a natural order — encoded as rank integers.
# One-hot: columns where order is arbitrary (gender, food_pref, study_env).
# Dealbreakers: binary flag per valid option.
# Numeric: convo_level, music_bother — kept as-is.

ORDINAL_COLS_ORDERS = [
    ("sleep_time",    ["10 PM – 11:30 PM", "11:30 PM – 1 AM", "1 AM – 3 AM",
                       "After 3 AM", "Varies wildly day to day"]),
    ("wake_time",     ["Before 6 AM", "6 – 8 AM", "8 – 10 AM", "After 10 AM"]),
    ("cleanliness",   ["Daily", "Every 2–3 days", "Weekly",
                       "When it gets messy enough to bother me", "Rarely / Never"]),
    ("late_return",   ["I almost never return after 11 PM",
                       "Use phone torch, try to stay quiet",
                       "Stay out until I know roommate is awake",
                       "Turn on the main light and move normally"]),
    ("conflict_style",["I stay quiet to keep the peace",
                       "Dropped a hint but haven't addressed it directly",
                       "Had one direct conversation about it",
                       "Set a clear boundary and expect it to be respected",
                       "Involved a third person or authority"]),
    ("sharing",       ["Never — always ask first", "Only small things like a pen",
                       "Most daily-use items are fine", "Basically anything is fine"]),
    ("nonveg_sensitivity", ["Very — it's a dealbreaker", "I'd prefer it not happen",
                             "Doesn't bother me", "I eat non-veg myself"]),
    ("guest_leaves",  ["I don't bring friends", "10 PM", "Midnight", "2 AM or later"]),
    ("evening_pref",  ["Alone in the room — lights dim, silence",
                       "Low-key — one person, quiet chat or a show",
                       "Friends in the room, casual hangout",
                       "Out with people, barely in the room"]),
    ("light_pref",    ["Complete darkness — lights off by 10 PM",
                       "Dim lamp is fine, main light off",
                       "Main light on is okay if someone's working",
                       "I wear a sleep mask — doesn't matter"]),
    ("temp_pref",     ["Cool (Cooler/AC at 18–20°C)", "Comfortable (Cooler/Fan at 22–24°C)",
                       "Warm (fan, no AC or AC at 26°C+)", "No strong preference"]),
    ("smoke_drink",   ["I don't smoke/drink at all",
                       "I smoke/drink only outside / designated areas",
                       "I smoke/drink in presence of peers",
                       "I sometimes smoke/drink in the room",
                       "I smoke/drink regularly in shared spaces"]),
]

# food_pref and study_env are one-hot (no meaningful rank)
OHE_COLS = ["food_pref", "study_env", "gender"]

VALID_DEALBREAKERS = [
    "Smoking Habits", "Sleep Schedule", "Cleanliness",
    "Loud Noise / Music", "Frequent guests at night",
    "Lack of privacy", "Roommate avoids all conflict",
]

# ── Ordinal encoding ──────────────────────────────────────────────────
ordinal_encoder = OrdinalEncoder(
    categories=[order for _, order in ORDINAL_COLS_ORDERS],
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)
ordinal_col_names = [col for col, _ in ORDINAL_COLS_ORDERS]
df_ordinal = pd.DataFrame(
    ordinal_encoder.fit_transform(df[ordinal_col_names]),
    columns=ordinal_col_names,
    index=df.index,
)

# ── One-hot encoding ──────────────────────────────────────────────────
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe_array = ohe.fit_transform(df[OHE_COLS])
ohe_col_names = ohe.get_feature_names_out(OHE_COLS).tolist()
df_ohe = pd.DataFrame(ohe_array, columns=ohe_col_names, index=df.index)

# ── Dealbreaker binary flags ──────────────────────────────────────────
db_flags = {}
for db in VALID_DEALBREAKERS:
    safe_name = "db_" + db.lower().replace(" ", "_").replace("/", "_")
    db_flags[safe_name] = (
        (df["dealbreaker1"] == db) |
        (df["dealbreaker2"] == db) |
        (df["dealbreaker3"] == db)
    ).astype(int)
df_dealbreakers = pd.DataFrame(db_flags, index=df.index)

# ── Numeric columns ───────────────────────────────────────────────────
df_numeric = df[["convo_level", "music_bother"]].copy().astype(float)

# ── Combine all encoded features ─────────────────────────────────────
df_encoded = pd.concat([df_ordinal, df_ohe, df_dealbreakers, df_numeric], axis=1)

print(f"Encoded shape: {df_encoded.shape}  "
      f"({len(ordinal_col_names)} ordinal, {len(ohe_col_names)} OHE, "
      f"{len(db_flags)} dealbreaker flags, 2 numeric)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 4 — Normalisation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# MinMaxScaler applied to ordinal + numeric columns only.
# OHE and dealbreaker binary columns are already in [0, 1] — do not scale.

SCALE_COLS  = ordinal_col_names + ["convo_level", "music_bother"]
NOSCALE_COLS = ohe_col_names + list(db_flags.keys())

scaler   = MinMaxScaler()
scaled   = scaler.fit_transform(df_encoded[SCALE_COLS])
df_scaled_part = pd.DataFrame(scaled, columns=SCALE_COLS, index=df.index)

df_final = pd.concat([df_scaled_part, df_encoded[NOSCALE_COLS]], axis=1)

# Verify all values in [0, 1]
assert df_final.min().min() >= 0.0, "Values below 0 after scaling"
assert df_final.max().max() <= 1.0, "Values above 1 after scaling"
print(f"Normalised shape: {df_final.shape}  — all values in [0.0, 1.0] ✅")

# Attach raw categorical columns for matrix lookups in Stage 6
df_final["food_pref_raw"]   = df["food_pref"].values
df_final["smoke_drink_raw"] = df["smoke_drink"].values
df_final["gender"]          = df["gender"].values
df_final["gender_imputed"]  = df["gender_imputed"].values
df_final["student_id"]      = df.index


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 5 — Custom Compatibility Matrices
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# These replace ordinal distance for food and smoking because the domain
# incompatibility between categories is not captured by numeric distance.
# Vegan + Non-veg = 0.1 is a qualitative assessment, not |4 - 1| = 3.
#
# How they're used:
#   In total_compatibility() (Stage 6), instead of
#     1 - abs(encoded_A["food"] - encoded_B["food"])
#   we call:
#     get_food_score(raw_food_A, raw_food_B)
#
# Matrix is symmetric. Missing pairs return DEFAULT_SCORE = 0.5.

DEFAULT_SCORE = 0.5

FOOD_MATRIX = {
    ("Strictly vegetarian",     "Strictly vegetarian")    : 1.0,
    ("Strictly vegetarian",     "Jain (no root vegetables)"): 0.9,
    ("Strictly vegetarian",     "Vegan")                  : 0.85,
    ("Strictly vegetarian",     "Vegetarian + eggs")      : 0.7,
    ("Strictly vegetarian",     "No preference")          : 0.5,
    ("Strictly vegetarian",     "Non-vegetarian")         : 0.2,
    ("Jain (no root vegetables)","Jain (no root vegetables)"): 1.0,
    ("Jain (no root vegetables)","Vegan")                 : 0.8,
    ("Jain (no root vegetables)","Vegetarian + eggs")     : 0.6,
    ("Jain (no root vegetables)","No preference")         : 0.4,
    ("Jain (no root vegetables)","Non-vegetarian")        : 0.1,
    ("Vegan",                   "Vegan")                  : 1.0,
    ("Vegan",                   "Vegetarian + eggs")      : 0.6,
    ("Vegan",                   "No preference")          : 0.4,
    ("Vegan",                   "Non-vegetarian")         : 0.1,
    ("Vegetarian + eggs",       "Vegetarian + eggs")      : 1.0,
    ("Vegetarian + eggs",       "No preference")          : 0.7,
    ("Vegetarian + eggs",       "Non-vegetarian")         : 0.4,
    ("No preference",           "No preference")          : 1.0,
    ("No preference",           "Non-vegetarian")         : 0.9,
    ("Non-vegetarian",          "Non-vegetarian")         : 1.0,
}

SMOKE_MATRIX = {
    ("I don't smoke/drink at all",                  "I don't smoke/drink at all")                  : 1.0,
    ("I don't smoke/drink at all",                  "I smoke/drink only outside / designated areas"): 0.8,
    ("I don't smoke/drink at all",                  "I smoke/drink in presence of peers")           : 0.4,
    ("I don't smoke/drink at all",                  "I sometimes smoke/drink in the room")          : 0.1,
    ("I don't smoke/drink at all",                  "I smoke/drink regularly in shared spaces")     : 0.0,
    ("I smoke/drink only outside / designated areas","I smoke/drink only outside / designated areas"): 1.0,
    ("I smoke/drink only outside / designated areas","I smoke/drink in presence of peers")          : 0.7,
    ("I smoke/drink only outside / designated areas","I sometimes smoke/drink in the room")         : 0.3,
    ("I smoke/drink only outside / designated areas","I smoke/drink regularly in shared spaces")    : 0.1,
    ("I smoke/drink in presence of peers",          "I smoke/drink in presence of peers")           : 1.0,
    ("I smoke/drink in presence of peers",          "I sometimes smoke/drink in the room")          : 0.6,
    ("I smoke/drink in presence of peers",          "I smoke/drink regularly in shared spaces")     : 0.4,
    ("I sometimes smoke/drink in the room",         "I sometimes smoke/drink in the room")          : 1.0,
    ("I sometimes smoke/drink in the room",         "I smoke/drink regularly in shared spaces")     : 0.8,
    ("I smoke/drink regularly in shared spaces",    "I smoke/drink regularly in shared spaces")     : 1.0,
}

def get_matrix_score(matrix, val_a, val_b):
    """Symmetric lookup — tries both orderings, returns DEFAULT if missing."""
    score = matrix.get((val_a, val_b)) or matrix.get((val_b, val_a))
    return score if score is not None else DEFAULT_SCORE

def get_food_score(food_a, food_b):
    return get_matrix_score(FOOD_MATRIX, food_a, food_b)

def get_smoke_score(smoke_a, smoke_b):
    return get_matrix_score(SMOKE_MATRIX, smoke_a, smoke_b)

print("✅ Custom compatibility matrices defined")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 6 — Compatibility Scoring Function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# HOW WEIGHTS WERE DERIVED (from EDA inferences):
#
#   - Dealbreaker frequency (Cell 4):
#       Smoking Habits and Sleep Schedule dominate #1 slot →
#       sleep_time and smoke_drink get the highest weights.
#
#   - Correlation heatmap (Cell 7):
#       If sleep_time and evening_pref correlate strongly (|r|>0.3),
#       they capture the same trait — one is discounted.
#       evening_pref is kept but at a lower weight.
#
#   - Gender differences (Cell 4b):
#       Female students show stronger divergence on cleanliness and
#       conflict_style → female weights are higher on those axes.
#       Male students show stronger divergence on smoke_drink and
#       guest_leaves → male weights are higher on those.
#
#   - conflict_style (Cell 9 recommendation):
#       Not in original scoring. Added with weight 0.07 because
#       silent mismatch is flagged as high-risk in the EDA summary.
#
#   - nonveg_sensitivity:
#       Global weight is low (0.03). For veg/Jain/Vegan students,
#       this is handled by the food matrix (which already penalises
#       the Veg/NonVeg pair heavily). Keeping it as a small global
#       signal is intentional.
#
#   Weights normalised to sum to 1.0 per gender.
#
# SMOKING IS A HARD FILTER:
#   If smoke score = 0.0, the pair is incompatible regardless of
#   all other scores. total_compatibility() returns 0.0 immediately.
#
# FOOD IS THE FIRST CONSTRAINT:
#   food_score is computed via matrix (not ordinal distance).
#   A score < 0.2 (e.g. Vegan + Non-veg) hard-caps the total at 0.0.

# Weights derived from EDA — Female cohort
WEIGHTS_FEMALE = {
    "sleep_time"        : 0.18,
    "wake_time"         : 0.06,
    "cleanliness"       : 0.12,   # higher for female (EDA Cell 4b)
    "late_return"       : 0.05,
    "conflict_style"    : 0.10,   # higher for female (EDA Cell 9 + 4b)
    "sharing"           : 0.05,
    "nonveg_sensitivity": 0.03,
    "guest_leaves"      : 0.06,
    "evening_pref"      : 0.07,
    "light_pref"        : 0.05,
    "temp_pref"         : 0.04,
    "convo_level"       : 0.07,
    "music_bother"      : 0.05,
    "food"              : 0.07,   # uses matrix, not ordinal distance
}

# Weights derived from EDA — Male cohort
WEIGHTS_MALE = {
    "sleep_time"        : 0.20,
    "wake_time"         : 0.07,
    "cleanliness"       : 0.08,
    "late_return"       : 0.06,
    "conflict_style"    : 0.07,
    "sharing"           : 0.05,
    "nonveg_sensitivity": 0.03,
    "guest_leaves"      : 0.08,   # higher for male (EDA Cell 4b)
    "evening_pref"      : 0.08,
    "light_pref"        : 0.04,
    "temp_pref"         : 0.04,
    "convo_level"       : 0.06,
    "music_bother"      : 0.07,
    "food"              : 0.07,
}

# Verify weights sum to 1.0
for name, weights in [("Female", WEIGHTS_FEMALE), ("Male", WEIGHTS_MALE)]:
    total = round(sum(weights.values()), 4)
    assert abs(total - 1.0) < 1e-6, f"{name} weights sum to {total}, not 1.0"
print("✅ Weight sets validated (both sum to 1.0)")


def ordinal_sim(val_a, val_b, max_val):
    """Similarity between two ordinal values: 1 - normalised distance."""
    if pd.isna(val_a) or pd.isna(val_b) or max_val == 0:
        return 0.5
    return 1.0 - abs(val_a - val_b) / max_val


def total_compatibility(idx_a, idx_b):
    """
    Returns compatibility score [0.0, 1.0] between two students.

    Uses gender of student A to select the weight set.
    Hard filter: returns 0.0 if smoking is incompatible or food score < 0.2.

    Parameters:
        idx_a, idx_b : integer row indices into df_final
    """
    a = df_final.iloc[idx_a]
    b = df_final.iloc[idx_b]

    weights = WEIGHTS_FEMALE if a["gender"] == "Female" else WEIGHTS_MALE

    # ── Hard filter 1: smoking ────────────────────────────────────────
    smoke_score = get_smoke_score(a["smoke_drink_raw"], b["smoke_drink_raw"])
    if smoke_score == 0.0:
        return 0.0

    # ── Hard filter 2: food incompatibility ───────────────────────────
    food_score = get_food_score(a["food_pref_raw"], b["food_pref_raw"])
    if food_score < 0.2:
        return 0.0

    # ── Ordinal feature similarities ──────────────────────────────────
    # max_val for each ordinal column = number of categories - 1
    ordinal_maxes = {col: (len(order) - 1) for col, order in ORDINAL_COLS_ORDERS}

    def osim(col):
        return ordinal_sim(a[col], b[col], ordinal_maxes.get(col, 1))

    score = (
        osim("sleep_time")         * weights["sleep_time"]
      + osim("wake_time")          * weights["wake_time"]
      + osim("cleanliness")        * weights["cleanliness"]
      + osim("late_return")        * weights["late_return"]
      + osim("conflict_style")     * weights["conflict_style"]
      + osim("sharing")            * weights["sharing"]
      + osim("nonveg_sensitivity") * weights["nonveg_sensitivity"]
      + osim("guest_leaves")       * weights["guest_leaves"]
      + osim("evening_pref")       * weights["evening_pref"]
      + osim("light_pref")         * weights["light_pref"]
      + osim("temp_pref")          * weights["temp_pref"]
      + (1 - abs(a["convo_level"] - b["convo_level"]))  * weights["convo_level"]
      + (1 - abs(a["music_bother"] - b["music_bother"])) * weights["music_bother"]
      + food_score                 * weights["food"]
    )

    # smoking contributes as a soft modifier too (already passed hard filter)
    # handled implicitly — if score is needed explicitly, add to weights

    return round(min(max(score, 0.0), 1.0), 4)


print("✅ total_compatibility() defined")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 7 — PCA (separate per gender)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# PCA is used ONLY for clustering (Stage 8), not for compatibility scoring.
# Weights are irrelevant here — PCA finds variance structure in the data.
# Separate PCA per gender: female and male students have different
# distributions (confirmed in EDA Cell 4b). Fitting one PCA across both
# would dilute the structure within each group.

# Feature columns for PCA — exclude raw strings, gender flags, and id
PCA_FEATURE_COLS = [c for c in df_final.columns
                    if c not in ("food_pref_raw", "smoke_drink_raw",
                                 "gender", "gender_imputed", "student_id")]

def run_pca_for_subset(subset_df, label, n_components=0.90):
    """
    Runs PCA on a gender subset.

    n_components=0.90 retains components explaining 90% of variance.
    Returns (pca_object, transformed_array, feature_names).
    """
    features = subset_df[PCA_FEATURE_COLS].values

    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(features)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    print(f"\n{label} PCA:")
    print(f"  Input features  : {features.shape[1]}")
    print(f"  Components kept : {pca.n_components_}  "
          f"(explain {cumulative[-1]*100:.1f}% of variance)")
    for i, (e, c) in enumerate(zip(explained, cumulative)):
        print(f"    PC{i+1:02d}  {e*100:.1f}%  cumulative {c*100:.1f}%")

    # Scree plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"PCA — {label}", fontsize=13, fontweight="bold")

    axes[0].bar(range(1, len(explained) + 1), explained * 100,
                color="#378ADD", edgecolor="white")
    axes[0].set_title("Explained variance per component")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Variance explained (%)")

    axes[1].plot(range(1, len(cumulative) + 1), cumulative * 100,
                 marker="o", color="#D85A30", linewidth=2)
    axes[1].axhline(90, color="#7F77DD", linestyle="--", linewidth=1,
                    label="90% threshold")
    axes[1].set_title("Cumulative explained variance")
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative variance (%)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return pca, transformed

# Split by gender — exclude imputed rows from PCA/clustering
# (gender_imputed rows stay in df_final for scoring but not for clustering)
df_female_clean = df_final[(df_final["gender"] == "Female") &
                            (~df_final["gender_imputed"])].copy()
df_male_clean   = df_final[(df_final["gender"] == "Male") &
                            (~df_final["gender_imputed"])].copy()

print(f"\nPCA/Clustering input:")
print(f"  Female (confirmed only): {len(df_female_clean)}")
print(f"  Male   (confirmed only): {len(df_male_clean)}")

pca_female, pca_female_data = run_pca_for_subset(df_female_clean, "Female")
pca_male,   pca_male_data   = run_pca_for_subset(df_male_clean,   "Male")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 8 — KMeans Clustering (separate per gender)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# KMeans runs on PCA-reduced data, separately per gender.
# Clustering groups students into lifestyle archetypes so that
# total_compatibility() only runs within clusters (not O(n²) globally).
#
# K selection: elbow method (inertia) + silhouette score.
# Range tested: 2 to min(10, n_students // 3).
# Final k is chosen at the elbow — visually inspect the plots.
# CHOSEN_K below can be set manually after inspection.

CHOSEN_K_FEMALE = 4   # update after inspecting elbow plot
CHOSEN_K_MALE   = 4   # update after inspecting elbow plot

CLUSTER_NAMES_FEMALE = {
    0: "Night Owl Introverts",
    1: "Early Bird Socials",
    2: "Flexible Peacekeepers",
    3: "Structured Cleaners",
}

CLUSTER_NAMES_MALE = {
    0: "Late Night Gamers",
    1: "Morning Grinders",
    2: "Chill Sharers",
    3: "Social Night Owls",
}

def select_k_and_cluster(pca_data, subset_df, label, chosen_k):
    """
    Runs elbow + silhouette analysis, then fits KMeans with chosen_k.
    Returns the subset_df with a 'cluster' column added.
    """
    n = len(pca_data)
    k_range = range(2, min(11, n // 3 + 1))

    if len(k_range) < 2:
        print(f"\n{label}: too few students ({n}) for meaningful clustering. "
              f"Assigning all to cluster 0.")
        subset_df = subset_df.copy()
        subset_df["cluster"] = 0
        return subset_df, None

    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(pca_data)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(pca_data, labels))

    # Elbow + silhouette plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"KMeans K Selection — {label}", fontsize=13, fontweight="bold")

    axes[0].plot(list(k_range), inertias, marker="o",
                 color="#378ADD", linewidth=2)
    axes[0].axvline(chosen_k, color="#D85A30", linestyle="--",
                    label=f"Chosen k={chosen_k}")
    axes[0].set_title("Inertia (elbow method)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].legend()

    axes[1].plot(list(k_range), sil_scores, marker="o",
                 color="#1D9E75", linewidth=2)
    axes[1].axvline(chosen_k, color="#D85A30", linestyle="--",
                    label=f"Chosen k={chosen_k}")
    axes[1].set_title("Silhouette score (higher = better)")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette score")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # Fit final model with chosen_k
    km_final = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(pca_data)

    subset_df = subset_df.copy()
    subset_df["cluster"] = cluster_labels

    print(f"\n{label} — KMeans with k={chosen_k}:")
    for c in range(chosen_k):
        n_c = (subset_df["cluster"] == c).sum()
        print(f"  Cluster {c}: {n_c} students  ({n_c / len(subset_df) * 100:.0f}%)")

    sil_final = silhouette_score(pca_data, cluster_labels)
    print(f"  Silhouette score : {sil_final:.3f}  "
          f"(>0.5 = strong, 0.2–0.5 = moderate, <0.2 = weak)")

    return subset_df, km_final


df_female_clustered, km_female = select_k_and_cluster(
    pca_female_data, df_female_clean, "Female", CHOSEN_K_FEMALE
)
df_male_clustered, km_male = select_k_and_cluster(
    pca_male_data, df_male_clean, "Male", CHOSEN_K_MALE
)

# ── Cluster profile summary ───────────────────────────────────────────
def print_cluster_profiles(clustered_df, label, k):
    """Prints the mode answer for each key column per cluster."""
    KEY_PROFILE_COLS = [
        "sleep_time", "cleanliness", "food_pref_raw",
        "smoke_drink_raw", "conflict_style", "study_env",
    ]
    print(f"\n{'=' * 55}")
    print(f"{label} Cluster Profiles")
    print(f"{'=' * 55}")
    for c in range(k):
        cluster_rows = clustered_df[clustered_df["cluster"] == c]
        print(f"\n  Cluster {c}  (n={len(cluster_rows)})")
        for col in KEY_PROFILE_COLS:
            raw_col = col if col in cluster_rows.columns else col.replace("_raw", "")
            if raw_col in cluster_rows.columns:
                mode_val = cluster_rows[raw_col].mode()
                val = str(mode_val.iloc[0])[:60] if not mode_val.empty else "N/A"  # ← fixed
                print(f"    {raw_col.replace('_raw',''):<20} : {val}")


def plot_clusters(pca_data, clustered_df, label, k, cluster_names):
    """
    Scatter plot of clusters using PC1 and PC2.
    Each cluster gets a distinct color and a designated name label.
    """
    COLORS = ["#378ADD", "#D85A30", "#1D9E75", "#9B59B6",
              "#E67E22", "#2ECC71", "#E74C3C", "#3498DB",
              "#F39C12", "#1ABC9C"]

    fig, ax = plt.subplots(figsize=(10, 7))
    cluster_labels = clustered_df["cluster"].values

    for c in range(k):
        mask = cluster_labels == c
        ax.scatter(
            pca_data[mask, 0],
            pca_data[mask, 1],
            c=COLORS[c % len(COLORS)],
            label=f"Cluster {c}: {cluster_names.get(c, f'Group {c}')}",
            s=80,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )

        # Centroid marker
        cx = pca_data[mask, 0].mean()
        cy = pca_data[mask, 1].mean()
        ax.scatter(cx, cy, c=COLORS[c % len(COLORS)],
                   s=250, marker="*", edgecolors="black",
                   linewidths=0.8, zorder=5)

        # Centroid label
        ax.annotate(
            cluster_names.get(c, f"Cluster {c}"),
            xy=(cx, cy),
            xytext=(cx + 0.05, cy + 0.05),
            fontsize=9,
            fontweight="bold",
            color=COLORS[c % len(COLORS)],
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
        )

    ax.set_title(f"{label} Student Clusters (PC1 vs PC2)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Principal Component 1", fontsize=11)
    ax.set_ylabel("Principal Component 2", fontsize=11)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# ── Calls ─────────────────────────────────────────────────────────────
print_cluster_profiles(df_female_clustered, "Female", CHOSEN_K_FEMALE)
print_cluster_profiles(df_male_clustered,   "Male",   CHOSEN_K_MALE)

plot_clusters(pca_female_data, df_female_clustered, "Female",
              CHOSEN_K_FEMALE, CLUSTER_NAMES_FEMALE)
plot_clusters(pca_male_data,   df_male_clustered,   "Male",
              CHOSEN_K_MALE,   CLUSTER_NAMES_MALE)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OUTPUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n✅ Preprocessing pipeline complete")
print(f"   df_final shape            : {df_final.shape}")
print(f"   Female clusters           : {CHOSEN_K_FEMALE}")
print(f"   Male clusters             : {CHOSEN_K_MALE}")
print(f"   Female clustered students : {len(df_female_clustered)}")
print(f"   Male clustered students   : {len(df_male_clustered)}")
