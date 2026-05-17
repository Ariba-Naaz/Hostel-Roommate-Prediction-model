# =============================================================
# scoring.py  —  compatibility score + full feature explanation
# =============================================================
import pandas as pd
import numpy as np
from config import (
    WEIGHTS_F, WEIGHTS_M,
    FOOD_MATRIX, SMOKE_MATRIX, DEFAULT_SCORE,
    ORDINAL_COLS, FEATURE_LABELS,
)

AGREE   = 0.75
PARTIAL = 0.35


def _mat(matrix, a, b):
    s = matrix.get((a, b)) or matrix.get((b, a))
    return s if s is not None else DEFAULT_SCORE

def food_score(a, b):  return _mat(FOOD_MATRIX,  a, b)
def smoke_score(a, b): return _mat(SMOKE_MATRIX, a, b)


def total_compatibility(row_a: pd.Series, row_b: pd.Series) -> float:
    """
    Score [0,1].  Both rows are MinMaxScaled → sim = 1 - |a-b|.
    Hard filters:  smoke == 0.0  or  food < 0.2  → return 0.
    """
    w = WEIGHTS_F if str(row_a.get("gender", "")) == "Female" else WEIGHTS_M

    ss = smoke_score(str(row_a["smoke_drink_raw"]), str(row_b["smoke_drink_raw"]))
    if ss == 0.0:
        return 0.0

    fs = food_score(str(row_a["food_pref_raw"]), str(row_b["food_pref_raw"]))
    if fs < 0.2:
        return 0.0

    def osim(col):
        va, vb = row_a.get(col), row_b.get(col)
        if va is None or vb is None or pd.isna(va) or pd.isna(vb):
            return 0.5
        return max(0.0, 1.0 - abs(float(va) - float(vb)))

    score = (
        osim("sleep_time")         * w["sleep_time"]
      + osim("wake_time")          * w["wake_time"]
      + osim("cleanliness")        * w["cleanliness"]
      + osim("late_return")        * w["late_return"]
      + osim("conflict_style")     * w["conflict_style"]
      + osim("sharing")            * w["sharing"]
      + osim("nonveg_sensitivity") * w["nonveg_sensitivity"]
      + osim("guest_leaves")       * w["guest_leaves"]
      + osim("evening_pref")       * w["evening_pref"]
      + osim("light_pref")         * w["light_pref"]
      + osim("temp_pref")          * w["temp_pref"]
      + osim("convo_level")        * w["convo_level"]
      + osim("music_bother")       * w["music_bother"]
      + fs                         * w["food"]
    )
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def score_label(s: float) -> str:
    if s >= 0.80: return "Excellent"
    if s >= 0.65: return "Good"
    if s >= 0.50: return "Moderate"
    return "Low"


def decode_ordinal(col, scaled_val):
    """Convert a scaled [0,1] ordinal value back to its category string."""
    orders = {c: o for c, o in ORDINAL_COLS}
    order  = orders.get(col)
    if order is None:
        return str(round(float(scaled_val), 2))
    try:
        idx = round(float(scaled_val) * (len(order) - 1))
        return order[max(0, min(idx, len(order) - 1))]
    except Exception:
        return str(scaled_val)


def explain_pair(row_a: pd.Series, row_b: pd.Series) -> dict:
    """
    Full feature-level breakdown.
    Returns:
      matched  : list of feature dicts where sim >= 0.75
      partial  : list where 0.35 <= sim < 0.75
      conflict : list where sim < 0.35
      + food_score, smoke_score, summary counts
    """
    w = WEIGHTS_F if str(row_a.get("gender", "")) == "Female" else WEIGHTS_M

    fs = food_score(str(row_a["food_pref_raw"]),   str(row_b["food_pref_raw"]))
    ss = smoke_score(str(row_a["smoke_drink_raw"]), str(row_b["smoke_drink_raw"]))

    matched, partial, conflict = [], [], []

    def classify(key, label, val_a, val_b, sim, weight):
        entry = {
            "feature": key, "label": label,
            "val_a":   str(val_a)[:60],
            "val_b":   str(val_b)[:60],
            "sim":     round(float(sim), 3),
            "weight":  round(float(weight), 4),
        }
        if sim >= AGREE:
            matched.append(entry)
        elif sim >= PARTIAL:
            partial.append(entry)
        else:
            conflict.append(entry)

    # ordinal features
    for col, _ in ORDINAL_COLS:
        va  = row_a.get(col, np.nan)
        vb  = row_b.get(col, np.nan)
        sim = max(0.0, 1.0 - abs(float(va) - float(vb))) \
              if not (pd.isna(va) or pd.isna(vb)) else 0.5
        classify(col, FEATURE_LABELS.get(col, col),
                 decode_ordinal(col, va), decode_ordinal(col, vb),
                 sim, w.get(col, 0))

    # numeric
    for col in ["convo_level", "music_bother"]:
        va  = float(row_a.get(col, 0.5))
        vb  = float(row_b.get(col, 0.5))
        sim = max(0.0, 1.0 - abs(va - vb))
        classify(col, FEATURE_LABELS.get(col, col),
                 f"{va:.2f}/5", f"{vb:.2f}/5",
                 sim, w.get(col, 0))

    # matrix features
    classify("food",        "Food Preference",   row_a["food_pref_raw"],   row_b["food_pref_raw"],   fs, w["food"])
    classify("smoke_drink", "Smoking / Drinking",row_a["smoke_drink_raw"], row_b["smoke_drink_raw"], ss, 0.0)

    matched.sort(key=lambda x: -x["sim"])
    partial.sort(key=lambda x: -x["sim"])
    conflict.sort(key=lambda x:  x["sim"])

    return {
        "matched":     matched,
        "partial":     partial,
        "conflict":    conflict,
        "food_score":  round(fs, 3),
        "smoke_score": round(ss, 3),
        "summary": {
            "n_matched":  len(matched),
            "n_partial":  len(partial),
            "n_conflict": len(conflict),
            "total":      len(matched) + len(partial) + len(conflict),
        },
    }
