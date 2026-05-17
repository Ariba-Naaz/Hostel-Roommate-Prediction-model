# =============================================================
# matching.py  —  cosine shortlist + top-3 scoring
# =============================================================
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scoring import total_compatibility, explain_pair, score_label


def _key_map(df_final):
    """Build {student_key -> row Series} lookup."""
    return {str(r["student_key"]): df_final.loc[i]
            for i, r in df_final.iterrows()}


def _make_result(rank, row_match, score, cosine, expl):
    return {
        "rank":         rank,
        "student_key":  str(row_match["student_key"]),
        "display_name": str(row_match["display_name"]),
        "score":        score,
        "score_pct":    round(score * 100, 1),
        "score_label":  score_label(score),
        "cosine":       round(cosine, 4),
        "food":         str(row_match["food_pref_raw"]),
        "smoke":        str(row_match["smoke_drink_raw"])[:55],
        "explanation":  expl,
    }


def build_candidates(df_clustered, feat_cols, top_n=10):
    """
    Within each cluster, cosine-similarity shortlist.
    Returns {student_key: [(cand_key, cosine), ...]}
    """
    cands = {}
    for cid in sorted(df_clustered["cluster"].unique()):
        rows = df_clustered[df_clustered["cluster"] == cid]
        if len(rows) < 2:
            for k in rows["student_key"]:
                cands[str(k)] = []
            continue
        mat  = rows[feat_cols].values.astype(float)
        keys = rows["student_key"].tolist()
        cos  = cosine_similarity(mat)
        for i, ka in enumerate(keys):
            scored = [(str(keys[j]), float(cos[i, j]))
                      for j in range(len(keys)) if j != i]
            scored.sort(key=lambda x: x[1], reverse=True)
            cands[str(ka)] = scored[:top_n]
    return cands


def score_top3(cands, df_final, label=""):
    """
    Score each shortlist via total_compatibility().
    Returns {student_key: [match_dict, ...]}  (up to 3 matches each).
    """
    km = _key_map(df_final)
    results = {}

    for ka, cand_list in cands.items():
        if ka not in km:
            results[ka] = []
            continue
        row_a  = km[ka]
        scored = []
        for kb, cos in cand_list:
            if kb not in km: continue
            row_b = km[kb]
            sc    = total_compatibility(row_a, row_b)
            if sc > 0.0:
                scored.append((kb, cos, sc, row_b))
        scored.sort(key=lambda x: (x[2], x[1]), reverse=True)
        top3 = []
        for rank, (kb, cos, sc, row_b) in enumerate(scored[:3], 1):
            expl = explain_pair(row_a, row_b)
            top3.append(_make_result(rank, row_b, sc, cos, expl))
        results[ka] = top3

    n_m = sum(1 for v in results.values() if v)
    n_u = len(results) - n_m
    print(f"{label}: matched={n_m}  unmatched={n_u}")
    return results
print(f"Done scoring all top-3 matches for ")