# =============================================================
# lookup.py  —  query functions used by the API
# =============================================================
import pickle, json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from config import MODEL_DIR
from preprocessing import encode_single
from scoring import total_compatibility, explain_pair, score_label


def load_artefacts(d=MODEL_DIR):
    with open(f"{d}/artefacts.pkl","rb") as f:
        art = pickle.load(f)
    df_final = pd.read_parquet(f"{d}/df_final.parquet")
    df_fc    = pd.read_parquet(f"{d}/df_fc.parquet")
    df_mc    = pd.read_parquet(f"{d}/df_mc.parquet")
    return art, df_final, df_fc, df_mc


def _key_map(df_final):
    return {str(r["student_key"]): df_final.loc[i]
            for i, r in df_final.iterrows()}


def _build_match(rank, row_b, score, cosine, expl):
    return {
        "rank":         rank,
        "student_key":  str(row_b["student_key"]),
        "display_name": str(row_b["display_name"]),
        "score":        score,
        "score_pct":    round(score*100, 1),
        "score_label":  score_label(score),
        "cosine":       round(cosine, 4),
        "food":         str(row_b["food_pref_raw"]),
        
        "explanation":  expl,
    }


# ── Find matches for a brand-new student (not in pool) ────────
def find_matches_new(raw, art, df_final, df_fc, df_mc, top_n=3):
    row = encode_single(raw, art["ord_enc"], art["ohe"], art["scaler"],
                        art["ohe_names"], art["scale_cols"], art["noscale_cols"])
    gender   = str(row["gender"])
    df_clust = df_fc if gender == "Female" else df_mc
    pca_obj  = art["pca_f"] if gender == "Female" else art["pca_m"]
    km_obj   = art["km_f"]  if gender == "Female" else art["km_m"]

    fv       = row[art["feat_cols"]].values.astype(float).reshape(1,-1)
    cid      = int(km_obj.predict(pca_obj.transform(fv))[0])
    cluster_rows = df_clust[df_clust["cluster"] == cid]

    if cluster_rows.empty:
        return {"cluster": cid, "matches": []}

    mat      = cluster_rows[art["feat_cols"]].values.astype(float)
    cos_vals = cosine_similarity(row[art["feat_cols"]].values.astype(float).reshape(1,-1), mat)[0]
    top_idx  = np.argsort(cos_vals)[::-1][:10]

    km       = _key_map(df_final)
    scored   = []
    for i in top_idx:
        kb   = str(cluster_rows["student_key"].iloc[i])
        if kb not in km: continue
        row_b = km[kb]
        sc   = total_compatibility(row, row_b)
        if sc > 0.0:
            scored.append((kb, float(cos_vals[i]), sc, row_b))
    scored.sort(key=lambda x: (x[2], x[1]), reverse=True)

    matches = []
    for rank, (kb, cos, sc, row_b) in enumerate(scored[:top_n], 1):
        expl = explain_pair(row, row_b)
        matches.append(_build_match(rank, row_b, sc, cos, expl))
    return {"cluster": cid, "matches": matches}


# ── Check two students from the pool by student_key ───────────
def check_two(key_a, key_b, df_final):
    km = _key_map(df_final)
    if key_a not in km: return {"error": f"Student '{key_a}' not found"}
    if key_b not in km: return {"error": f"Student '{key_b}' not found"}
    row_a  = km[key_a]
    row_b  = km[key_b]
    score  = total_compatibility(row_a, row_b)
    expl   = explain_pair(row_a, row_b)
    return {
        "student_a":      key_a,
        "display_a":      str(row_a["display_name"]),
        "student_b":      key_b,
        "display_b":      str(row_b["display_name"]),
        "score":          score,
        "score_pct":      round(score*100, 1),
        "score_label":    score_label(score),
        "food_a":         str(row_a["food_pref_raw"]),
        "food_b":         str(row_b["food_pref_raw"]),
        "explanation":    expl,
    }


# ── Batch: return pre-computed top-3 for ALL students ─────────
def get_all_matches(art, df_final, df_fc, df_mc):
    """
    Returns a list of student records, each with their top-3 matches.
    Uses the pre-computed top3 JSON files saved by pipeline.
    """
    import json, os
    results = []
    km = _key_map(df_final)

    for fname, gender in [(f"{MODEL_DIR}/top3_f.json","Female"),
                           (f"{MODEL_DIR}/top3_m.json","Male")]:
        if not os.path.exists(fname):
            continue
        with open(fname) as f:
            top3 = json.load(f)
        for key, matches in top3.items():
            if key not in km:
                continue
            row = km[key]
            results.append({
                "student_key":  key,
                "display_name": str(row["display_name"]),
                "gender":       str(row["gender"]),
                "food":         str(row["food_pref_raw"]),
                "matches":      matches,
            })
    return results


# ── Check two students given raw preference dicts (no pool) ───
def check_two_raw(raw_a, raw_b, art):
    """For the 'check compatibility' mode where user fills two forms."""
    row_a = encode_single(raw_a, art["ord_enc"], art["ohe"], art["scaler"],
                          art["ohe_names"], art["scale_cols"], art["noscale_cols"])
    row_b = encode_single(raw_b, art["ord_enc"], art["ohe"], art["scaler"],
                          art["ohe_names"], art["scale_cols"], art["noscale_cols"])
    score = total_compatibility(row_a, row_b)
    expl  = explain_pair(row_a, row_b)
    return {
        "display_a":   row_a["display_name"],
        "display_b":   row_b["display_name"],
        "score":       score,
        "score_pct":   round(score*100, 1),
        "score_label": score_label(score),
        "food_a":      str(row_a["food_pref_raw"]),
        "food_b":      str(row_b["food_pref_raw"]),
        "explanation": expl,
    }
