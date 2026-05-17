# =============================================================
# config.py  —  single source of truth, no dealbreakers
# =============================================================
import os

DATA_PATH  = os.environ.get("DATA_PATH",  "data/responses.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "eda_figures")
MODEL_DIR  = os.environ.get("MODEL_DIR",  "models")

# ── canonical ordinal orders ──────────────────────────────────
SLEEP_ORDER = [
    "10 PM – 11:30 PM", "11:30 PM – 1 AM",
    "1 AM – 3 AM", "After 3 AM", "Varies wildly day to day",
]
WAKE_ORDER = ["Before 6 AM", "6 – 8 AM", "8 – 10 AM", "After 10 AM"]
FOOD_ORDER = [
    "Strictly vegetarian", "Jain (no root vegetables)",
    "Vegan", "Vegetarian + eggs", "No preference", "Non-vegetarian",
]
SENSITIVITY_ORDER = [
    "Very — it's a dealbreaker", "I'd prefer it not happen",
    "Doesn't bother me", "I eat non-veg myself",
]
EVENING_ORDER = [
    "Alone in the room — lights dim, silence",
    "Low-key — one person, quiet chat or a show",
    "Friends in the room, casual hangout",
    "Out with people, barely in the room",
]
CLEANLINESS_ORDER = [
    "Daily", "Every 2–3 days", "Weekly",
    "When it gets messy enough to bother me", "Rarely / Never",
]
CONFLICT_ORDER = [
    "I stay quiet to keep the peace",
    "Dropped a hint but haven't addressed it directly",
    "Had one direct conversation about it",
    "Set a clear boundary and expect it to be respected",
    "Involved a third person or authority",
]
STUDY_ORDER = [
    "Complete silence — any noise breaks focus",
    "Quiet — soft background is okay",
    "I use headphones, so room noise doesn't matter",
    "I can study through most noise",
]
GUEST_ORDER    = ["I don't bring friends", "10 PM", "Midnight", "2 AM or later"]
SMOKE_ORDER    = [
    "I don't smoke/drink at all",
    "I smoke/drink only outside / designated areas",
    "I smoke/drink in presence of peers",
    "I sometimes smoke/drink in the room",
    "I smoke/drink regularly in shared spaces",
]
LATE_RETURN_ORDER = [
    "I almost never return after 11 PM",
    "Use phone torch, try to stay quiet",
    "Stay out until I know roommate is awake",
    "Turn on the main light and move normally",
]
SHARING_ORDER = [
    "Never — always ask first", "Only small things like a pen",
    "Most daily-use items are fine", "Basically anything is fine",
]
LIGHT_ORDER = [
    "Complete darkness — lights off by 10 PM",
    "Dim lamp is fine, main light off",
    "Main light on is okay if someone's working",
    "I wear a sleep mask — doesn't matter",
]
TEMP_ORDER = [
    "Cool (Cooler/AC at 18–20°C)",
    "Comfortable (Cooler/Fan at 22–24°C)",
    "Warm (fan, no AC or AC at 26°C+)",
    "No strong preference",
]

# list of (column_name, ordered_categories)  — used for encoding
ORDINAL_COLS = [
    ("sleep_time",         SLEEP_ORDER),
    ("wake_time",          WAKE_ORDER),
    ("cleanliness",        CLEANLINESS_ORDER),
    ("late_return",        LATE_RETURN_ORDER),
    ("conflict_style",     CONFLICT_ORDER),
    ("sharing",            SHARING_ORDER),
    ("nonveg_sensitivity", SENSITIVITY_ORDER),
    ("guest_leaves",       GUEST_ORDER),
    ("evening_pref",       EVENING_ORDER),
    ("light_pref",         LIGHT_ORDER),
    ("temp_pref",          TEMP_ORDER),
    ("smoke_drink",        SMOKE_ORDER),
]

OHE_COLS = ["food_pref", "study_env", "gender"]

JAIN_VARIANTS = [
    "Jain(no root vegetable)", "Jain (no root vegetable)",
    "Jain(no root vegetables)",
]

# ── Compatibility weights (must sum to 1.0 per gender) ────────
# food weight is injected at FOOD_W and the rest are scaled down
FOOD_W = 0.08

_RAW_F = {
    "sleep_time":0.20,"wake_time":0.07,"cleanliness":0.13,
    "late_return":0.05,"conflict_style":0.11,"sharing":0.06,
    "nonveg_sensitivity":0.04,"guest_leaves":0.06,
    "evening_pref":0.07,"light_pref":0.05,"temp_pref":0.04,
    "convo_level":0.07,"music_bother":0.05,
}
_RAW_M = {
    "sleep_time":0.22,"wake_time":0.08,"cleanliness":0.09,
    "late_return":0.06,"conflict_style":0.08,"sharing":0.05,
    "nonveg_sensitivity":0.04,"guest_leaves":0.09,
    "evening_pref":0.09,"light_pref":0.04,"temp_pref":0.04,
    "convo_level":0.06,"music_bother":0.06,
}

def _norm(raw, fw=FOOD_W):
    t = sum(raw.values())
    s = (1.0 - fw) / t
    d = {k: v * s for k, v in raw.items()}
    d["food"] = fw
    return d

WEIGHTS_F = _norm(_RAW_F)
WEIGHTS_M = _norm(_RAW_M)

for _n, _w in [("F", WEIGHTS_F), ("M", WEIGHTS_M)]:
    assert abs(sum(_w.values()) - 1.0) < 1e-4, f"{_n} weights sum error"

# ── Compatibility matrices ─────────────────────────────────────
DEFAULT_SCORE = 0.5

FOOD_MATRIX = {
    ("Strictly vegetarian",      "Strictly vegetarian"):       1.00,
    ("Strictly vegetarian",      "Jain (no root vegetables)"): 0.90,
    ("Strictly vegetarian",      "Vegan"):                     0.85,
    ("Strictly vegetarian",      "Vegetarian + eggs"):         0.70,
    ("Strictly vegetarian",      "No preference"):             0.50,
    ("Strictly vegetarian",      "Non-vegetarian"):            0.20,
    ("Jain (no root vegetables)","Jain (no root vegetables)"): 1.00,
    ("Jain (no root vegetables)","Vegan"):                     0.80,
    ("Jain (no root vegetables)","Vegetarian + eggs"):         0.60,
    ("Jain (no root vegetables)","No preference"):             0.40,
    ("Jain (no root vegetables)","Non-vegetarian"):            0.10,
    ("Vegan",                    "Vegan"):                     1.00,
    ("Vegan",                    "Vegetarian + eggs"):         0.60,
    ("Vegan",                    "No preference"):             0.40,
    ("Vegan",                    "Non-vegetarian"):            0.10,
    ("Vegetarian + eggs",        "Vegetarian + eggs"):         1.00,
    ("Vegetarian + eggs",        "No preference"):             0.70,
    ("Vegetarian + eggs",        "Non-vegetarian"):            0.40,
    ("No preference",            "No preference"):             1.00,
    ("No preference",            "Non-vegetarian"):            0.90,
    ("Non-vegetarian",           "Non-vegetarian"):            1.00,
}
SMOKE_MATRIX = {
    ("I don't smoke/drink at all",                    "I don't smoke/drink at all"):                    1.00,
    ("I don't smoke/drink at all",                    "I smoke/drink only outside / designated areas"): 0.80,
    ("I don't smoke/drink at all",                    "I smoke/drink in presence of peers"):             0.40,
    ("I don't smoke/drink at all",                    "I sometimes smoke/drink in the room"):            0.10,
    ("I don't smoke/drink at all",                    "I smoke/drink regularly in shared spaces"):       0.00,
    ("I smoke/drink only outside / designated areas", "I smoke/drink only outside / designated areas"):  1.00,
    ("I smoke/drink only outside / designated areas", "I smoke/drink in presence of peers"):             0.70,
    ("I smoke/drink only outside / designated areas", "I sometimes smoke/drink in the room"):            0.30,
    ("I smoke/drink only outside / designated areas", "I smoke/drink regularly in shared spaces"):       0.10,
    ("I smoke/drink in presence of peers",            "I smoke/drink in presence of peers"):             1.00,
    ("I smoke/drink in presence of peers",            "I sometimes smoke/drink in the room"):            0.60,
    ("I smoke/drink in presence of peers",            "I smoke/drink regularly in shared spaces"):       0.40,
    ("I sometimes smoke/drink in the room",           "I sometimes smoke/drink in the room"):            1.00,
    ("I sometimes smoke/drink in the room",           "I smoke/drink regularly in shared spaces"):       0.80,
    ("I smoke/drink regularly in shared spaces",      "I smoke/drink regularly in shared spaces"):       1.00,
}

PALETTE = ["#378ADD","#D85A30","#1D9E75","#7F77DD","#EF9F27","#D4537E"]

FEATURE_LABELS = {
    "sleep_time":         "Sleep Time",
    "wake_time":          "Wake Time",
    "cleanliness":        "Cleanliness",
    "late_return":        "Late Return",
    "conflict_style":     "Conflict Style",
    "sharing":            "Sharing Habits",
    "nonveg_sensitivity": "Non-Veg Sensitivity",
    "guest_leaves":       "Guest Timing",
    "evening_pref":       "Evening Preference",
    "light_pref":         "Light Preference",
    "temp_pref":          "Temperature",
    "convo_level":        "Conversation Level",
    "music_bother":       "Music Tolerance",
    "food":               "Food Preference",
    "smoke_drink":        "Smoking/Drinking",
}
