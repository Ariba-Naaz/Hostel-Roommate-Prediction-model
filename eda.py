# ╔══════════════════════════════════════════════════════════════════╗
# ║        Hostel Roommate Compatibility — EDA                      ║
# ║        Run independently. No dependency on preprocessing.py.    ║
# ╚══════════════════════════════════════════════════════════════════╝

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2

# %matplotlib inline

# ── Config ───────────────────────────────────────────────────────────
DATA_PATH  = r"C:\Users\HP\Documents\Hostel_Roommate_Prediction_model_VS_Code\Hostel-Roommate-Prediction-model\Raw_data.csv"
OUTPUT_DIR = "eda_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor" : "white",
    "axes.facecolor"   : "#f9f9f7",
    "axes.grid"        : True,
    "grid.color"       : "#e0e0e0",
    "grid.linewidth"   : 0.6,
    "font.size"        : 11,
    "axes.titlesize"   : 13,
    "axes.titleweight" : "bold",
    "axes.labelsize"   : 11,
})

PALETTE = ["#378ADD", "#D85A30", "#1D9E75", "#7F77DD", "#EF9F27", "#D4537E"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 1 — Shared Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# These must match the canonical values in preprocessing.py exactly.
# Any change here must be mirrored there.

SLEEP_ORDER = [
    "10 PM – 11:30 PM", "11:30 PM – 1 AM",
    "1 AM – 3 AM", "After 3 AM", "Varies wildly day to day",
]
WAKE_ORDER        = ["Before 6 AM", "6 – 8 AM", "8 – 10 AM", "After 10 AM"]
FOOD_ORDER        = [
    "Strictly vegetarian", "Jain (no root vegetables)",
    "Vegan", "Vegetarian + eggs", "No preference", "Non-vegetarian",
]
SENSITIVITY_ORDER = [
    "Very — it's a dealbreaker", "I'd prefer it not happen",
    "Doesn't bother me", "I eat non-veg myself",
]
EVENING_ORDER     = [
    "Alone in the room — lights dim, silence",
    "Low-key — one person, quiet chat or a show",
    "Friends in the room, casual hangout",
    "Out with people, barely in the room",
]
CLEANLINESS_ORDER = [
    "Daily", "Every 2–3 days", "Weekly",
    "When it gets messy enough to bother me", "Rarely / Never",
]
CONFLICT_ORDER    = [
    "I stay quiet to keep the peace",
    "Dropped a hint but haven't addressed it directly",
    "Had one direct conversation about it",
    "Set a clear boundary and expect it to be respected",
    "Involved a third person or authority",
]
STUDY_ORDER       = [
    "Complete silence — any noise breaks focus",
    "Quiet — soft background is okay",
    "I use headphones, so room noise doesn't matter",
    "I can study through most noise",
]
GUEST_ORDER       = ["I don't bring friends", "10 PM", "Midnight", "2 AM or later"]
SMOKE_ORDER       = [
    "I don't smoke/drink at all",
    "I smoke/drink only outside / designated areas",
    "I smoke/drink in presence of peers",
    "I sometimes smoke/drink in the room",
    "I smoke/drink regularly in shared spaces",
]

VALID_DEALBREAKERS = [
    "Smoking Habits", "Sleep Schedule", "Cleanliness",
    "Loud Noise / Music", "Frequent guests at night",
    "Lack of privacy", "Roommate avoids all conflict",
]

COMPARE_COLS = {
    "sleep_time"        : SLEEP_ORDER,
    "wake_time"         : WAKE_ORDER,
    "food_pref"         : FOOD_ORDER,
    "evening_pref"      : EVENING_ORDER,
    "cleanliness"       : CLEANLINESS_ORDER,
    "study_env"         : STUDY_ORDER,
    "smoke_drink"       : SMOKE_ORDER,
    "temp_pref"         : [
        "Cool (Cooler/AC at 18–20°C)", "Comfortable (Cooler/Fan at 22–24°C)",
        "Warm (fan, no AC or AC at 26°C+)", "No strong preference",
    ],
    "light_pref"        : [
        "Complete darkness — lights off by 10 PM", "Dim lamp is fine, main light off",
        "Main light on is okay if someone's working", "I wear a sleep mask — doesn't matter",
    ],
    "late_return"       : [
        "I almost never return after 11 PM", "Use phone torch, try to stay quiet",
        "Stay out until I know roommate is awake", "Turn on the main light and move normally",
    ],
    "conflict_style"    : CONFLICT_ORDER,
    "sharing"           : [
        "Never — always ask first", "Only small things like a pen",
        "Most daily-use items are fine", "Basically anything is fine",
    ],
    "nonveg_sensitivity": SENSITIVITY_ORDER,
    "guest_leaves"      : GUEST_ORDER,
}

print("✅ Cell 1 complete")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 2 — Load, Clean, Validate + Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

df = pd.read_csv(DATA_PATH)
df = df[df["Timestamp"].notna() & (df["Timestamp"] != "")].copy()
df = df.reset_index(drop=True)

df.columns = [
    "timestamp", "gender", "sleep_time", "wake_time", "late_return",
    "cleanliness", "evening_pref", "guest_leaves", "convo_level",
    "study_env", "music_bother", "food_pref", "nonveg_sensitivity",
    "smoke_drink", "conflict_style", "sharing", "temp_pref",
    "light_pref", "dealbreaker1", "dealbreaker2", "dealbreaker3",
    "optional_notes",
]
df = df.drop(columns=["timestamp", "optional_notes"])

# Normalise Jain spelling variants to canonical form
JAIN_VARIANTS = [
    "Jain(no root vegetable)", "Jain (no root vegetable)", "Jain(no root vegetables)",
]
df["food_pref"] = df["food_pref"].replace(JAIN_VARIANTS, "Jain (no root vegetables)")

# music_bother: handle "4 - Very bothered" string format
if df["music_bother"].dtype == object:
    df["music_bother"] = (
        df["music_bother"].str.extract(r"(\d)")[0]
        .astype(float).astype("Int64")
    )

# convo_level: 1–10 self-rated conversation preference
df["convo_level"] = pd.to_numeric(df["convo_level"], errors="coerce")

MUSIC_BOTHER_ORDER = sorted(df["music_bother"].dropna().unique().tolist())

# ── Missing value handling ────────────────────────────────────────────
# Categorical: fill with mode per column
CAT_COLS = [c for c in COMPARE_COLS if c in df.columns]
for col in CAT_COLS:
    mode = df[col].mode()
    if not mode.empty:
        df[col] = df[col].fillna(mode[0])

# Numeric: fill with median
for col in ["convo_level", "music_bother"]:
    df[col] = df[col].fillna(df[col].median())

# Dealbreaker slots: fill with explicit None string (not NaN) for filtering
for col in ["dealbreaker1", "dealbreaker2", "dealbreaker3"]:
    df[col] = df[col].fillna("None")

# Gender: do NOT impute — missing gender rows stay in df (full-dataset
# analysis) but are excluded from df_female / df_male to avoid
# fabricating gender data for the gender-split charts.
df_female     = df[df["gender"] == "Female"].copy()
df_male       = df[df["gender"] == "Male"].copy()
missing_gender = df["gender"].isna().sum()

# ── Validation ───────────────────────────────────────────────────────
print("=" * 55)
print("DATA VALIDATION")
print("=" * 55)
print(f"Total responses     : {len(df)}")
print(f"Female (confirmed)  : {len(df_female)}")
print(f"Male   (confirmed)  : {len(df_male)}")
print(f"Missing gender      : {missing_gender}")

remaining_missing = df.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]
print(f"\nMissing after imputation:\n"
      f"{remaining_missing.to_string() if not remaining_missing.empty else '  None ✅'}")

print("\nUnexpected category values:")
found_unexpected = False
for col, order in COMPARE_COLS.items():
    if col not in df.columns:
        continue
    bad = set(df[col].dropna().unique()) - set(order)
    if bad:
        print(f"  {col}: {bad}")
        found_unexpected = True
if not found_unexpected:
    print("  None ✅")

if found_unexpected:
    raise ValueError("Fix unexpected category values above before continuing.")

print(f"\nmusic_bother range  : {df['music_bother'].min()} – {df['music_bother'].max()}")
print(f"convo_level range   : {df['convo_level'].min()} – {df['convo_level'].max()}")


# ── Helper functions ─────────────────────────────────────────────────

def pct(num, denom):
    p = 100 * num / denom if denom else 0
    return f"{num} / {denom}  ({p:.1f}%)"

def add_bar_labels(ax, bars, fontsize=9):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.15,
                    str(int(h)), ha="center", va="bottom", fontsize=fontsize)

def plot_bars(ax, counts, title, color, rotate=30):
    bars = ax.bar(counts.index, counts.values,
                  color=color, edgecolor="white", width=0.6)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xticklabels(counts.index, rotation=rotate, ha="right")
    add_bar_labels(ax, bars)

def grouped_bar(ax, f_counts, m_counts, order, title):
    x, width = np.arange(len(order)), 0.35
    bars_f = ax.bar(x - width / 2, f_counts.values, width,
                    label="Female", color=PALETTE[4], edgecolor="white")
    bars_m = ax.bar(x + width / 2, m_counts.values, width,
                    label="Male",   color=PALETTE[0], edgecolor="white")
    add_bar_labels(ax, list(bars_f) + list(bars_m))
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [l[:18] + ".." if len(l) > 18 else l for l in order],
        rotation=30, ha="right", fontsize=8,
    )
    ax.legend(fontsize=8)

def plot_crosstab(ax, data, col1, col2, order1, order2, title):
    for col in [col1, col2]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found.")
    ct = pd.crosstab(data[col1], data[col2])
    present_rows = [o for o in order1 if o in ct.index]
    present_cols = [o for o in order2 if o in ct.columns]
    for o in order1:
        if o not in ct.index:
            print(f"  [warn] '{title}': missing row: {o}")
    for o in order2:
        if o not in ct.columns:
            print(f"  [warn] '{title}': missing col: {o}")
    ct = (ct.reindex(present_rows).reindex(present_cols, axis=1)
            .fillna(0).astype(int))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax,
                linewidths=0.4, linecolor="#e0e0e0", cbar_kws={"shrink": 0.6})
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels(
        [l.get_text()[:18] + ".." if len(l.get_text()) > 18
         else l.get_text() for l in ax.get_yticklabels()],
        rotation=0, fontsize=8)
    ax.set_xticklabels(
        [l.get_text()[:15] + ".." if len(l.get_text()) > 15
         else l.get_text() for l in ax.get_xticklabels()],
        rotation=30, ha="right", fontsize=8)

def get_dealbreaker_counts(subset_df, slots=("dealbreaker1", "dealbreaker2", "dealbreaker3")):
    combined = pd.concat([subset_df[s] for s in slots], ignore_index=True)
    combined = combined[combined.isin(VALID_DEALBREAKERS)]
    return combined.value_counts().reindex(VALID_DEALBREAKERS, fill_value=0)

def save_fig(fig, name):
    fig.savefig(f"{OUTPUT_DIR}/{name.replace(' ', '_').replace('/', '-')}.png",
                dpi=150, bbox_inches="tight")

print("\n✅ Cell 2 complete")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 3 — Sleep & Wake Distributions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sleep_counts = df["sleep_time"].value_counts().reindex(SLEEP_ORDER, fill_value=0)
wake_counts  = df["wake_time"].value_counts().reindex(WAKE_ORDER,  fill_value=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Sleep & Wake Time Distributions", fontsize=14, fontweight="bold")

for ax, counts, title, color in [
    (axes[0], sleep_counts, "When do students fall asleep?", PALETTE[0]),
    (axes[1], wake_counts,  "When do students wake up?",    PALETTE[1]),
]:
    ax.barh(counts.index, counts.values, color=color, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Number of students")
    ax.invert_yaxis()
    for i, v in enumerate(counts.values):
        ax.text(v + 0.2, i, str(v), va="center", fontsize=10)

plt.tight_layout()
save_fig(fig, "3_sleep_wake")
plt.show()

late_n = df["sleep_time"].isin(["1 AM – 3 AM", "After 3 AM"]).sum()
print(f"Most common sleep time : {sleep_counts.idxmax()}  ({sleep_counts.max()} students)")
print(f"Most common wake time  : {wake_counts.idxmax()}  ({wake_counts.max()} students)")
print(f"Night owls (after 1 AM): {pct(late_n, len(df))}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 4 — Dealbreaker Frequency Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

counts_all = get_dealbreaker_counts(df)
counts_1   = get_dealbreaker_counts(df, slots=("dealbreaker1",))
counts_2   = get_dealbreaker_counts(df, slots=("dealbreaker2",))
counts_3   = get_dealbreaker_counts(df, slots=("dealbreaker3",))

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Dealbreaker Analysis — All Priority Slots",
             fontsize=14, fontweight="bold")

plot_bars(axes[0, 0], counts_all, "All slots combined",             PALETTE[2])
plot_bars(axes[0, 1], counts_1,   "Dealbreaker #1 (most critical)", PALETTE[0])
plot_bars(axes[1, 0], counts_2,   "Dealbreaker #2",                 PALETTE[1])
plot_bars(axes[1, 1], counts_3,   "Dealbreaker #3",                 PALETTE[3])

plt.tight_layout()
save_fig(fig, "4_dealbreakers")
plt.show()

# Each student appears in up to 3 slots, so combined % uses len(df)*3
total_slots = len(df) * 3
print("All slots combined (% of total slot fills):")
for label, cnt in counts_all.items():
    print(f"  {label:<35} {cnt:>3}  ({cnt / total_slots * 100:.1f}%)")

print("\nDealbreaker #1 only (% of respondents):")
for label, cnt in counts_1.items():
    print(f"  {label:<35} {cnt:>3}  ({cnt / len(df) * 100:.1f}%)")

print(f"\nMost critical (#1)  : {counts_1.idxmax()}  ({counts_1.max()} students)")
print(f"Most mentioned total: {counts_all.idxmax()}  ({counts_all.max()} mentions)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 4b — Gender Split Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 55)
print("CELL 4b — Gender Split Analysis")
print("=" * 55)
print(f"Female : {len(df_female)}  |  Male : {len(df_male)}", end="")
if missing_gender:
    print(f"  |  Missing (excluded): {missing_gender}")
else:
    print()

# ── Section A: All columns ────────────────────────────────────────────
n_rows = (len(COMPARE_COLS) + 1) // 2
fig, axes = plt.subplots(n_rows, 2, figsize=(16, n_rows * 5))
fig.suptitle("Male vs Female — All Parameters", fontsize=14, fontweight="bold")
axes_flat = axes.flatten()

for idx, (col, order) in enumerate(COMPARE_COLS.items()):
    f_c = df_female[col].value_counts().reindex(order, fill_value=0)
    m_c = df_male[col].value_counts().reindex(order, fill_value=0)
    grouped_bar(axes_flat[idx], f_c, m_c, order, col.replace("_", " ").title())

for idx in range(len(COMPARE_COLS), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.tight_layout()
save_fig(fig, "4b_A_gender_all_cols")
plt.show()

# ── Section B: Numeric means ──────────────────────────────────────────
print("\nNumeric columns — mean by gender:")
print(f"{'Column':<20} {'Female':>10} {'Male':>10} {'Diff':>8}")
print("-" * 50)
for col in ["convo_level", "music_bother"]:
    fm, mm = df_female[col].mean(), df_male[col].mean()
    diff = round(fm - mm, 2)
    sign = "+" if diff > 0 else ""
    print(f"  {col:<18} {fm:>10.2f} {mm:>10.2f}  {sign}{diff}")

# ── Section C: Dealbreaker deep dive ─────────────────────────────────
print("\n--- SECTION C: Dealbreaker Deep Dive by Gender ---")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Dealbreaker Deep Dive by Gender", fontsize=14, fontweight="bold")

for ax, slot, title in [
    (axes[0, 0], ("dealbreaker1", "dealbreaker2", "dealbreaker3"), "All slots combined"),
    (axes[0, 1], ("dealbreaker1",),                                "Dealbreaker #1 only"),
    (axes[1, 0], ("dealbreaker2",),                                "Dealbreaker #2 only"),
    (axes[1, 1], ("dealbreaker3",),                                "Dealbreaker #3 only"),
]:
    grouped_bar(ax,
                get_dealbreaker_counts(df_female, slots=slot),
                get_dealbreaker_counts(df_male,   slots=slot),
                VALID_DEALBREAKERS, title)

plt.tight_layout()
save_fig(fig, "4b_C_dealbreaker_gender")
plt.show()

# ── Section D: Cross analysis within each gender ─────────────────────
print("\n--- SECTION D: Cross Analysis Within Each Gender ---")

cross_pairs = [
    ("sleep_time",   "food_pref",      SLEEP_ORDER,        FOOD_ORDER,         "Sleep vs Food preference"),
    ("cleanliness",  "conflict_style", CLEANLINESS_ORDER,  CONFLICT_ORDER,     "Cleanliness vs Conflict style"),
    ("study_env",    "music_bother",   STUDY_ORDER,        MUSIC_BOTHER_ORDER, "Study env vs Music bother"),
    ("evening_pref", "guest_leaves",   EVENING_ORDER,      GUEST_ORDER,        "Evening pref vs Guest timing"),
]

for col1, col2, order1, order2, title in cross_pairs:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"Cross Analysis: {title}", fontsize=13, fontweight="bold")
    for ax, subset, label in zip(
        axes,
        [df_female, df_male],
        [f"Female (n={len(df_female)})", f"Male   (n={len(df_male)})"],
    ):
        plot_crosstab(ax, subset, col1, col2, order1, order2, label)
    plt.tight_layout()
    save_fig(fig, f"4b_D_{title}")
    plt.show()

# ── Section E: Top answer comparison ─────────────────────────────────
print("\n" + "=" * 55)
print("SUMMARY — KEY GENDER DIFFERENCES")
print("=" * 55)
for col, order in COMPARE_COLS.items():
    if df_female[col].dropna().empty or df_male[col].dropna().empty:
        continue
    f_top = df_female[col].value_counts().idxmax()
    m_top = df_male[col].value_counts().idxmax()
    flag  = "✅" if f_top == m_top else "❌"
    print(f"\n{col.replace('_', ' ').title():<25} {flag}")
    print(f"  Female : {f_top[:60]}")
    print(f"  Male   : {m_top[:60]}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 5 — Food Preference Breakdown
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

food_counts = df["food_pref"].value_counts().reindex(FOOD_ORDER, fill_value=0)

print("=" * 55)
print("SECTION A — Overall Food Preference")
print("=" * 55)
for cat, cnt in food_counts.items():
    print(f"  {cat:<30} {pct(cnt, len(df))}")

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(food_counts.index, food_counts.values,
              color=PALETTE[3], edgecolor="white", width=0.6)
ax.set_title("Overall food preference distribution")
ax.set_ylabel("Number of students")
ax.set_xticklabels(food_counts.index, rotation=25, ha="right")
add_bar_labels(ax, bars)
plt.tight_layout()
save_fig(fig, "5_A_food_overall")
plt.show()

print("\n" + "=" * 55)
print("SECTION B — Non-Veg Sensitivity Among Veg Students")
print("=" * 55)

veg_df = df[df["food_pref"].isin([
    "Strictly vegetarian", "Jain (no root vegetables)", "Vegan"
])].copy()
sens_counts     = veg_df["nonveg_sensitivity"].value_counts().reindex(
    SENSITIVITY_ORDER, fill_value=0)
dealbreaker_cnt = sens_counts["Very — it's a dealbreaker"]

print(f"Veg + Jain + Vegan students : {len(veg_df)}")
for cat, cnt in sens_counts.items():
    print(f"  {cat:<35} {pct(cnt, len(veg_df))}")
print(f"\nWould reject non-veg roommate : {pct(dealbreaker_cnt, len(veg_df))}")

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(sens_counts.index, sens_counts.values,
              color=PALETTE[2], edgecolor="white", width=0.6)
ax.set_title(f"Non-veg sensitivity — veg/Jain/Vegan students (n={len(veg_df)})")
ax.set_ylabel("Count")
ax.set_xticklabels(sens_counts.index, rotation=25, ha="right")
add_bar_labels(ax, bars)
plt.tight_layout()
save_fig(fig, "5_B_nonveg_sensitivity")
plt.show()

print("\n" + "=" * 55)
print("SECTION C — Food Preference by Gender")
print("=" * 55)

f_food = df_female["food_pref"].value_counts().reindex(FOOD_ORDER, fill_value=0)
m_food = df_male["food_pref"].value_counts().reindex(FOOD_ORDER, fill_value=0)

print(f"\n{'Category':<30} {'Female':>8} {'Male':>8}")
print("-" * 48)
for cat in FOOD_ORDER:
    print(f"  {cat:<28} {f_food[cat]:>8} {m_food[cat]:>8}")

fig, ax = plt.subplots(figsize=(12, 5))
x, width = np.arange(len(FOOD_ORDER)), 0.35
bars_f = ax.bar(x - width / 2, f_food.values, width,
                label="Female", color=PALETTE[4], edgecolor="white")
bars_m = ax.bar(x + width / 2, m_food.values, width,
                label="Male",   color=PALETTE[0], edgecolor="white")
add_bar_labels(ax, list(bars_f) + list(bars_m))
ax.set_title("Food Preference by Gender")
ax.set_ylabel("Count")
ax.set_xticks(x)
ax.set_xticklabels(FOOD_ORDER, rotation=25, ha="right")
ax.legend()
plt.tight_layout()
save_fig(fig, "5_C_food_by_gender")
plt.show()

print(f"\n  Female top food pref : {f_food.idxmax()}")
print(f"  Male top food pref   : {m_food.idxmax()}")
print(f"  ⚠ {dealbreaker_cnt} veg students CANNOT share with "
      f"{food_counts['Non-vegetarian']} non-veg students")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 6 — Cross-Tab Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 55)
print("SECTION A — Cross-Tabs on Full Dataset")
print("=" * 55)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Cross-Tab Analysis — Full Dataset", fontsize=14, fontweight="bold")

plot_crosstab(axes[0, 0], df, "sleep_time",   "evening_pref",
              SLEEP_ORDER, EVENING_ORDER, "Sleep time vs Evening preference")
plot_crosstab(axes[0, 1], df, "cleanliness",  "conflict_style",
              CLEANLINESS_ORDER, CONFLICT_ORDER, "Cleanliness vs Conflict style")
plot_crosstab(axes[1, 0], df, "study_env",    "music_bother",
              STUDY_ORDER, MUSIC_BOTHER_ORDER, "Study env vs Music bother")
plot_crosstab(axes[1, 1], df, "evening_pref", "guest_leaves",
              EVENING_ORDER, GUEST_ORDER, "Evening pref vs Guest timing")

plt.tight_layout()
save_fig(fig, "6_A_full_crosstabs")
plt.show()

print("\n" + "=" * 55)
print("SECTION B — Cross-Tabs Split by Gender")
print("=" * 55)

crosstab_pairs = [
    ("sleep_time",   "evening_pref",      SLEEP_ORDER,        EVENING_ORDER,       "Sleep vs Evening pref"),
    ("cleanliness",  "conflict_style",     CLEANLINESS_ORDER,  CONFLICT_ORDER,      "Cleanliness vs Conflict"),
    ("study_env",    "music_bother",       STUDY_ORDER,        MUSIC_BOTHER_ORDER,  "Study env vs Music bother"),
    ("evening_pref", "guest_leaves",       EVENING_ORDER,      GUEST_ORDER,         "Evening pref vs Guests"),
    ("food_pref",    "nonveg_sensitivity", FOOD_ORDER,         SENSITIVITY_ORDER,   "Food vs NonVeg sensitivity"),
]

for col1, col2, order1, order2, title in crosstab_pairs:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"{title} — by Gender", fontsize=13, fontweight="bold")
    plot_crosstab(axes[0], df_female, col1, col2, order1, order2,
                  f"Female (n={len(df_female)})")
    plot_crosstab(axes[1], df_male,   col1, col2, order1, order2,
                  f"Male   (n={len(df_male)})")
    plt.tight_layout()
    save_fig(fig, f"6_B_{title}")
    plt.show()

print("\n" + "=" * 55)
print("SECTION C — Key Findings")
print("=" * 55)

late_sleepers_df = df[df["sleep_time"].isin(["1 AM – 3 AM", "After 3 AM"])]
late_social      = late_sleepers_df[
    late_sleepers_df["evening_pref"] == "Friends in the room, casual hangout"]
daily_cleaners   = df[df["cleanliness"] == "Daily"]
clean_direct     = daily_cleaners[
    daily_cleaners["conflict_style"] == "Set a clear boundary and expect it to be respected"]
silence_df       = df[df["study_env"] == "Complete silence — any noise breaks focus"]
silence_music    = silence_df[silence_df["music_bother"] >= 4]
social_evening   = df[df["evening_pref"] == "Friends in the room, casual hangout"]
social_late      = social_evening[
    social_evening["guest_leaves"].isin(["Midnight", "2 AM or later"])]

print(f"Late sleepers who want friends over : {pct(len(late_social),   len(late_sleepers_df))}")
print(f"Daily cleaners who set clear bounds : {pct(len(clean_direct),  len(daily_cleaners))}")
print(f"Need silence + hate music (≥4/5)    : {pct(len(silence_music), len(silence_df))}")
print(f"Social evening + late guests        : {pct(len(social_late),   len(social_evening))}")

print(f"\n✅ Cell 6 complete")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 7 — Correlation Heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Named df_corr (not df_encoded) to avoid collision with preprocessing.py
# when both files are discussed together. food_pref excluded — nominal.

print("=" * 55)
print("CELL 7 — Correlation Heatmap")
print("=" * 55)

ORDINAL_ENCODE = {k: v for k, v in COMPARE_COLS.items() if k != "food_pref"}

df_corr = pd.DataFrame({
    col: df[col].map({val: idx for idx, val in enumerate(order)})
    for col, order in ORDINAL_ENCODE.items()
})
df_corr["convo_level"]  = df["convo_level"]
df_corr["music_bother"] = df["music_bother"].astype(float)

corr = df_corr.corr(method="pearson")

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    corr,
    mask=np.triu(np.ones_like(corr, dtype=bool)),
    annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, vmin=-1, vmax=1, ax=ax,
    linewidths=0.3, linecolor="#e0e0e0",
    annot_kws={"size": 8},
    cbar_kws={"shrink": 0.6, "label": "Pearson r"},
)
ax.set_title(
    "Lifestyle Variable Correlations\n"
    "(ordinal encoded — treat as directional signals)",
    fontsize=12,
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
plt.tight_layout()
save_fig(fig, "7_correlation_heatmap")
plt.show()

corr_pairs = (
    corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))
        .stack().reset_index()
)
corr_pairs.columns = ["var1", "var2", "r"]
strong = corr_pairs[corr_pairs["r"].abs() > 0.25].sort_values(
    "r", key=abs, ascending=False)

print("\nStrongest correlations (|r| > 0.25):")
if strong.empty:
    print("  None — variables are largely independent.")
else:
    for _, row in strong.iterrows():
        print(f"  {row['var1']:<22} × {row['var2']:<22} "
              f"{'↑↑' if row['r'] > 0 else '↑↓'}  r={row['r']:+.2f}")

print("\n✅ Cell 7 complete")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 8 — Outlier Detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 55)
print("CELL 8 — Outlier Detection")
print("=" * 55)

EXTREME_THRESHOLD = 1.5
FLAG_MIN_COLS     = 4

nan_threshold = int(df_corr.shape[1] * 0.3)
df_enc_valid  = df_corr[df_corr.isna().sum(axis=1) <= nan_threshold]
df_enc_filled = df_enc_valid.fillna(df_enc_valid.median(numeric_only=True))

print(f"Students scored : {len(df_enc_filled)}  |  "
      f"Excluded (>30% missing): {len(df) - len(df_enc_filled)}")

scaler        = StandardScaler()
z_matrix      = scaler.fit_transform(df_enc_filled)
z_df          = pd.DataFrame(z_matrix, columns=df_enc_filled.columns,
                              index=df_enc_filled.index)
extreme_counts = (z_df.abs() > EXTREME_THRESHOLD).sum(axis=1)
flagged_z      = extreme_counts[extreme_counts >= FLAG_MIN_COLS].index
print(f"Z-score flagged (≥{FLAG_MIN_COLS} extreme cols) : {len(flagged_z)}")

maha_ok, maha_flagged, maha_series = False, pd.Index([]), pd.Series(dtype=float)

try:
    cov_inv     = np.linalg.pinv(np.cov(z_matrix, rowvar=False))
    diff        = z_matrix - z_matrix.mean(axis=0)
    maha_dist   = np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))
    maha_series = pd.Series(maha_dist, index=df_enc_filled.index)
    maha_thresh = chi2.ppf(0.99, df=df_enc_filled.shape[1])
    maha_flagged = maha_series[maha_series > maha_thresh].index
    maha_ok      = True
    print(f"Mahalanobis flagged (top 1%)         : {len(maha_flagged)}")
except np.linalg.LinAlgError:
    print("Mahalanobis: skipped (singular matrix — likely too few rows)")

combined_flagged = flagged_z.union(maha_flagged)
print(f"Combined flagged (either method)     : {len(combined_flagged)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Outlier Detection — Lifestyle Extremity",
             fontsize=14, fontweight="bold")

axes[0].hist(extreme_counts.values,
             bins=range(0, int(extreme_counts.max()) + 2),
             color=PALETTE[0], edgecolor="white", align="left")
axes[0].axvline(FLAG_MIN_COLS, color=PALETTE[1], linestyle="--", linewidth=1.5,
                label=f"Flag threshold ({FLAG_MIN_COLS}+ cols)")
axes[0].set_title("Extreme lifestyle axes per student")
axes[0].set_xlabel("Columns with |z| > 1.5")
axes[0].set_ylabel("Number of students")
axes[0].legend()

if maha_ok:
    axes[1].hist(maha_dist, bins=25, color=PALETTE[2], edgecolor="white")
    axes[1].axvline(maha_thresh, color=PALETTE[1], linestyle="--", linewidth=1.5,
                    label="χ² threshold (p=0.01)")
    axes[1].set_title("Mahalanobis distance distribution")
    axes[1].set_xlabel("Mahalanobis distance")
    axes[1].set_ylabel("Number of students")
    axes[1].legend()
else:
    axes[1].set_visible(False)

plt.tight_layout()
save_fig(fig, "8_outlier_detection")
plt.show()

KEY_COLS = [
    "sleep_time", "cleanliness", "evening_pref",
    "guest_leaves", "conflict_style", "study_env", "music_bother",
]
print("\n" + "=" * 55)
print("FLAGGED STUDENT PROFILES")
print("=" * 55)

if len(combined_flagged) == 0:
    print("No students flagged — cohort is relatively homogeneous.")
else:
    flagged_df = df.loc[combined_flagged, KEY_COLS].copy()
    flagged_df["extreme_cols"] = extreme_counts.reindex(combined_flagged).values
    if maha_ok:
        flagged_df["mahalanobis"] = maha_series.reindex(combined_flagged).round(2).values

    for i, (idx, row) in enumerate(flagged_df.iterrows(), 1):
        maha_str = f", maha={row['mahalanobis']:.1f}" if maha_ok else ""
        print(f"\n  Student #{i} (row {idx})  "
              f"[extreme cols: {int(row['extreme_cols'])}{maha_str}]")
        for col in KEY_COLS:
            print(f"    {col:<20} : {str(df.loc[idx, col])[:55]}")

print(f"\n⚠  {len(combined_flagged)} students flagged — match last or "
      f"consider single-room allocation.")
print("\n✅ Cell 8 complete")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 9 — Final Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print()
print("╔" + "═" * 53 + "╗")
print("║   HOSTEL ROOMMATE COMPATIBILITY — FINAL SUMMARY   ║")
print("╚" + "═" * 53 + "╝")

different_cols, same_cols = [], []
for col in COMPARE_COLS:
    if df_female[col].dropna().empty or df_male[col].dropna().empty:
        continue
    f_top = df_female[col].value_counts().idxmax()
    m_top = df_male[col].value_counts().idxmax()
    (same_cols if f_top == m_top else different_cols).append(col.replace("_", " "))

print(f"""
COHORT
  Total responses         : {len(df)}
  Female (confirmed)      : {len(df_female)}
  Male   (confirmed)      : {len(df_male)}
  Missing gender          : {missing_gender}

SLEEP
  Most common bedtime     : {sleep_counts.idxmax()}  ({sleep_counts.max()} students)
  Night owls (after 1 AM) : {pct(late_n, len(df))}

DEALBREAKERS
  Most critical (#1)      : {counts_1.idxmax()}  ({counts_1.max()} students)
  Most mentioned overall  : {counts_all.idxmax()}  ({counts_all.max()} mentions)
  Top 3 combined:""")

for label, cnt in counts_all.nlargest(3).items():
    print(f"    {label:<35} {pct(cnt, total_slots)}")

print(f"""
FOOD & COMPATIBILITY RISK
  Strictly vegetarian     : {pct(food_counts['Strictly vegetarian'], len(df))}
  Non-vegetarian          : {pct(food_counts['Non-vegetarian'], len(df))}
  Veg smell dealbreaker   : {pct(dealbreaker_cnt, len(veg_df))} of veg students
  ⚠ Hard incompatibility : {dealbreaker_cnt} veg students cannot share with
                            {food_counts['Non-vegetarian']} non-veg students

BEHAVIOURAL PATTERNS
  Late sleepers wanting friends over : {pct(len(late_social),   len(late_sleepers_df))}
  Silence-needing + hate music (≥4)  : {pct(len(silence_music), len(silence_df))}

HARD-TO-MATCH STUDENTS
  Flagged as outliers     : {len(combined_flagged)} / {len(df_enc_filled)}
  Recommendation          : match last or consider single-room allocation

GENDER DIFFERENCES
  Same top answer         : {", ".join(same_cols) or "None"}
  Different top answer    : {", ".join(different_cols) or "None"}

MATCHING IMPLICATIONS
  1. Match on food first — {dealbreaker_cnt} veg students are hard-incompatible
     with all {food_counts['Non-vegetarian']} non-veg students. No algorithm overrides this.

  2. Sleep schedule is the #1 dealbreaker. Use it as the second filter
     after food before scoring any other axis.

  3. Run matching on the {len(df) - len(combined_flagged)} non-outlier students first.
     Handle {len(combined_flagged)} outliers manually afterward.

""")

print("✅ EDA complete")
print(f"✅ All figures saved to: {OUTPUT_DIR}/")