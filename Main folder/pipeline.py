# =============================================================
# pipeline.py  —  master orchestrator
# Run: python pipeline.py
# =============================================================
import os, pickle, json
import numpy as np
import pandas as pd
from config import MODEL_DIR
from data_loader import load_and_prepare
from preprocessing import build_encoders, encode_and_scale
from clustering import cluster_subset
from matching import build_candidates, score_top3
from sklearn.metrics import silhouette_score

os.makedirs(MODEL_DIR, exist_ok=True)
META = ["food_pref_raw","smoke_drink_raw","gender",
        "gender_imputed","student_key","display_name"]


def feat_cols(df_final):
    return [c for c in df_final.columns if c not in META]


def run_pipeline(data_path=None, plot=True):
    from config import DATA_PATH as DP
    path = data_path or DP

    print("\n── STAGE 1: Load ──")
    df = load_and_prepare(path)

    print("\n── STAGE 2: Encode ──")
    ord_enc, ohe, scaler = build_encoders(df)
    df_final, ohe_names, scale_cols, noscale_cols = \
        encode_and_scale(df, ord_enc, ohe, scaler)
    fc = feat_cols(df_final)

    df_f = df_final[(df_final["gender"]=="Female")&(~df_final["gender_imputed"])].copy()
    df_m = df_final[(df_final["gender"]=="Male")  &(~df_final["gender_imputed"])].copy()
    print(f"  Female={len(df_f)}  Male={len(df_m)}")

    print("\n── STAGE 3: Cluster ──")
    df_fc, pca_f, Xpca_f, win_f, km_f = cluster_subset(df_f, fc, "Female", plot)
    df_mc, pca_m, Xpca_m, win_m, km_m = cluster_subset(df_m, fc, "Male",   plot)

    print("\n── STAGE 4: Cosine ──")
    cands_f = build_candidates(df_fc, fc)
    cands_m = build_candidates(df_mc, fc)

    print("\n── STAGE 5: Score ──")
    top3_f = score_top3(cands_f, df_final, "Female")
    top3_m = score_top3(cands_m, df_final, "Male")

    print("\n── STAGE 6: Accuracy ──")
    def sil(Xpca, labels):
        try:    return float(silhouette_score(Xpca, labels))
        except: return 0.0

    lbl_f = df_fc["cluster"].values
    lbl_m = df_mc["cluster"].values
    sf = sil(Xpca_f, lbl_f)
    sm = sil(Xpca_m, lbl_m)
    top1_f = [v[0]["score"] for v in top3_f.values() if v]
    top1_m = [v[0]["score"] for v in top3_m.values() if v]

    def pct_exc(lst): return round(sum(s>=0.8 for s in lst)/max(len(lst),1)*100,1)
    def mean_s(lst):  return round(np.mean(lst),3) if lst else 0.0

    print(f"  Female sil={sf:.4f}  mean_top1={mean_s(top1_f)}  exc={pct_exc(top1_f)}%")
    print(f"  Male   sil={sm:.4f}  mean_top1={mean_s(top1_m)}  exc={pct_exc(top1_m)}%")

    print("\n── STAGE 7: Save ──")
    art = {
        "ord_enc":     ord_enc,
        "ohe":         ohe,
        "scaler":      scaler,
        "pca_f":       pca_f,
        "pca_m":       pca_m,
        "km_f":        km_f,
        "km_m":        km_m,
        "ohe_names":   ohe_names,
        "scale_cols":  scale_cols,
        "noscale_cols":noscale_cols,
        "feat_cols":   fc,
    }
    with open(f"{MODEL_DIR}/artefacts.pkl","wb") as f:
        pickle.dump(art, f)
    df_final.to_parquet(f"{MODEL_DIR}/df_final.parquet")
    df_fc.to_parquet(f"{MODEL_DIR}/df_fc.parquet")
    df_mc.to_parquet(f"{MODEL_DIR}/df_mc.parquet")

    # Save top3 keyed by student_key
    def _js(d):
        return {k: v for k,v in d.items()}
    with open(f"{MODEL_DIR}/top3_f.json","w") as f:
        json.dump(_js(top3_f), f, indent=2)
    with open(f"{MODEL_DIR}/top3_m.json","w") as f:
        json.dump(_js(top3_m), f, indent=2)

    summary = {
        "n_total":       int(len(df)),
        "n_female":      int(len(df_f)),
        "n_male":        int(len(df_m)),
        "winner_female": win_f,
        "winner_male":   win_m,
        "k_female":      int(len(set(lbl_f))),
        "k_male":        int(len(set(lbl_m))),
        "sil_female":    sf,
        "sil_male":      sm,
        "mean_top1_female": mean_s(top1_f),
        "mean_top1_male":   mean_s(top1_m),
        "pct_exc_female":   pct_exc(top1_f),
        "pct_exc_male":     pct_exc(top1_m),
    }
    with open(f"{MODEL_DIR}/summary.json","w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ All artefacts saved to {MODEL_DIR}/")
    print(f"   Female winner: {win_f}  Male winner: {win_m}")
    return summary


if __name__ == "__main__":
    run_pipeline()
