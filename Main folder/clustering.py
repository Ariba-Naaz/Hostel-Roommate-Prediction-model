# =============================================================
# clustering.py  —  4-model selection + PCA
# =============================================================
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from config import PALETTE
warnings.filterwarnings("ignore")


def run_pca(X: np.ndarray, label: str, n_components=0.90, plot=True):
    pca  = PCA(n_components=n_components, random_state=42)
    Xpca = pca.fit_transform(X)
    ev   = pca.explained_variance_ratio_
    print(f"  {label} PCA: {pca.n_components_} components "
          f"({np.cumsum(ev)[-1]*100:.1f}% variance)")
    if plot:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(range(1, len(ev)+1), ev*100, color=PALETTE[0], edgecolor="white")
        ax.set_title(f"PCA variance — {label}")
        ax.set_xlabel("PC"); ax.set_ylabel("Variance %")
        plt.tight_layout(); plt.show()
    return pca, Xpca


def _km(X_orig, Xpca, max_k):
    best = {"score": -1}
    for k in range(2, max_k+1):
        km  = KMeans(n_clusters=k, random_state=42, n_init=15)
        lbl = km.fit_predict(Xpca)
        if len(set(lbl)) < 2: continue
        s = silhouette_score(X_orig, lbl)
        if s > best["score"]:
            best = {"score": s, "model": km, "labels": lbl, "k": k}
    return best


def _db(X_orig):
    best = {"score": -1}
    for eps in np.arange(0.2, 2.1, 0.1):
        for ms in [3, 5, 7]:
            db  = DBSCAN(eps=round(eps,2), min_samples=ms)
            lbl = db.fit_predict(X_orig)
            nc  = len(set(lbl)) - (1 if -1 in lbl else 0)
            if nc < 2 or (lbl == -1).mean() > 0.30: continue
            mask = lbl != -1
            if mask.sum() < 3: continue
            s = silhouette_score(X_orig[mask], lbl[mask])
            if s > best["score"]:
                best = {"score": s, "model": db, "labels": lbl,
                        "eps": round(eps,2), "min_samples": ms, "n_clusters": nc}
    return best


def _gmm(X_orig, Xpca, max_k):
    best = {"score": -1}
    for n in range(2, max_k+1):
        g   = GaussianMixture(n_components=n, covariance_type="full",
                              random_state=42, n_init=5)
        g.fit(Xpca)
        lbl = g.predict(Xpca)
        if len(set(lbl)) < 2: continue
        s = silhouette_score(X_orig, lbl)
        if s > best["score"]:
            best = {"score": s, "model": g, "labels": lbl, "n": n}
    return best


def _agg(X_orig, max_k):
    best = {"score": -1}
    for lnk in ["ward", "average", "complete"]:
        for k in range(2, max_k+1):
            a   = AgglomerativeClustering(n_clusters=k, linkage=lnk)
            lbl = a.fit_predict(X_orig)
            if len(set(lbl)) < 2: continue
            s = silhouette_score(X_orig, lbl)
            if s > best["score"]:
                best = {"score": s, "model": a, "labels": lbl,
                        "k": k, "linkage": lnk}
    return best


def _fix_noise(labels, X):
    labels = labels.copy()
    nm = labels == -1
    if not nm.any(): return labels
    cids = [c for c in set(labels) if c != -1]
    cens = {c: X[labels==c].mean(axis=0) for c in cids}
    for i in np.where(nm)[0]:
        labels[i] = min(cids, key=lambda c: np.linalg.norm(X[i]-cens[c]))
    return labels


def cluster_subset(df_sub, feat_cols, label, plot=True):
    """
    Run 4 models, pick winner by silhouette on original space.
    Returns (df_sub_with_cluster, pca, Xpca, winner_name, km_for_assign).
    """
    X    = df_sub[feat_cols].values.astype(float)
    max_k = min(10, len(X)//3)

    if max_k < 2:
        df_out = df_sub.copy(); df_out["cluster"] = 0
        pca, Xpca = run_pca(X, label, plot=False)
        return df_out, pca, Xpca, "trivial", \
               KMeans(n_clusters=1, random_state=42, n_init=1).fit(Xpca)

    pca, Xpca = run_pca(X, label, plot=plot)

    results = {}
    results["KMeans"]       = _km(X, Xpca, max_k)
    db = _db(X)
    if db["score"] > -1: results["DBSCAN"] = db
    results["GMM"]          = _gmm(X, Xpca, max_k)
    results["Agglomerative"]= _agg(X, max_k)

    winner_name = max(results, key=lambda n: results[n]["score"])
    winner      = results[winner_name]

    print(f"\n  {label} model comparison:")
    for n, r in results.items():
        flag = " ← WINNER" if n == winner_name else ""
        print(f"    {n:<18} sil={r['score']:.4f}{flag}")

    labels = winner["labels"].copy()
    if winner_name == "DBSCAN" and -1 in labels:
        labels = _fix_noise(labels, X)

    k_final = len(set(labels))
    print(f"  {label}: {k_final} clusters")

    df_out = df_sub.copy()
    df_out["cluster"] = labels

    # Always keep a KMeans for fast new-user cluster assignment
    if winner_name == "KMeans":
        km_assign = winner["model"]
    else:
        km_assign = KMeans(n_clusters=k_final, random_state=42, n_init=15)
        km_assign.fit(Xpca)

    return df_out, pca, Xpca, winner_name, km_assign
