# =============================================================
# api.py  —  FastAPI backend
# Install: pip install fastapi uvicorn pandas pyarrow scikit-learn scipy matplotlib seaborn
# Run:     uvicorn api:app --reload --port 8000
# =============================================================
import json, os
from typing import Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import MODEL_DIR
from lookup import (load_artefacts, find_matches_new,
                    check_two, get_all_matches, check_two_raw)

app = FastAPI(title="RoomSync API", version="4.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)

ST = {}   # global state


@app.on_event("startup")
async def startup():
    try:
        art, df, df_f, df_m = load_artefacts()
        ST.update({"art":art,"df":df,"df_f":df_f,"df_m":df_m,"ready":True})
        print("✅ Artefacts loaded")
    except Exception as e:
        ST["ready"] = False
        print(f"⚠  {e}\n   Run: python pipeline.py")


def _chk():
    if not ST.get("ready"):
        raise HTTPException(503, "Model not ready. Run python pipeline.py first.")


# ── Pydantic models ──────────────────────────────────────────
class Prefs(BaseModel):
    name:               str   = Field(..., example="Priya Sharma")
    roll_no:            str   = Field(..., example="22BCS041")
    gender:             str   = Field(..., example="Female")
    food_pref:          str   = Field(..., example="Strictly vegetarian")
    nonveg_sensitivity: str   = Field(..., example="I'd prefer it not happen")
    smoke_drink:        str   = Field(..., example="I don't smoke/drink at all")
    sleep_time:         str   = Field(..., example="11:30 PM – 1 AM")
    wake_time:          str   = Field(..., example="8 – 10 AM")
    late_return:        str   = Field(..., example="Use phone torch, try to stay quiet")
    guest_leaves:       str   = Field(..., example="10 PM")
    cleanliness:        str   = Field(..., example="Every 2–3 days")
    evening_pref:       str   = Field(..., example="Low-key — one person, quiet chat or a show")
    temp_pref:          str   = Field(..., example="Comfortable (Cooler/Fan at 22–24°C)")
    light_pref:         str   = Field(..., example="Dim lamp is fine, main light off")
    study_env:          str   = Field(..., example="Quiet — soft background is okay")
    music_bother:       float = Field(..., ge=1, le=5)
    conflict_style:     str   = Field(..., example="Had one direct conversation about it")
    sharing:            str   = Field(..., example="Most daily-use items are fine")
    convo_level:        float = Field(..., ge=1, le=10)
    top_n:              int   = Field(3, ge=1, le=10)


class TwoPrefs(BaseModel):
    student_a: Prefs
    student_b: Prefs


class TwoKeys(BaseModel):
    key_a: str
    key_b: str


# ── Endpoints ────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model_ready": ST.get("ready", False)}


@app.get("/summary")
async def summary():
    _chk()
    try:
        with open(f"{MODEL_DIR}/summary.json") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(404, "summary.json not found — run pipeline first")


@app.post("/match/new")
async def match_new(body: Prefs):
    """New student (not in pool) finds their top-N matches from existing pool."""
    _chk()
    raw   = body.dict()
    top_n = raw.pop("top_n", 3)
    try:
        return find_matches_new(raw, ST["art"], ST["df"],
                                ST["df_f"], ST["df_m"], top_n)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/match/two-raw")
async def match_two_raw(body: TwoPrefs):
    """Check compatibility between two students using their raw preferences."""
    _chk()
    ra = body.student_a.dict(); ra.pop("top_n", None)
    rb = body.student_b.dict(); rb.pop("top_n", None)
    try:
        return check_two_raw(ra, rb, ST["art"])
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/match/two-keys")
async def match_two_keys(body: TwoKeys):
    """Check compatibility between two students already in the pool by roll number."""
    _chk()
    r = check_two(body.key_a.upper(), body.key_b.upper(), ST["df"])
    if "error" in r:
        raise HTTPException(404, r["error"])
    return r


@app.get("/match/all")
async def match_all():
    """
    Return the pre-computed top-3 matches for EVERY student in the pool.
    This is the 'batch / group' view — all students at once.
    """
    _chk()
    try:
        return {"students": get_all_matches(ST["art"], ST["df"],
                                             ST["df_f"], ST["df_m"])}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/students")
async def list_students():
    _chk()
    df = ST["df"]
    out = [
        {"student_key":  str(r["student_key"]),
         "display_name": str(r["display_name"]),
         "gender":       str(r["gender"]),
         "food":         str(r["food_pref_raw"])}
        for _, r in df.iterrows()
    ]
    return {"students": out, "total": len(out)}


@app.get("/model/clusters")
async def cluster_data():
    """PCA cluster coordinates for scatter visualisation."""
    _chk()
    try:
        art  = ST["art"]
        df_f = ST["df_f"]
        df_m = ST["df_m"]

        def proj(df_c, pca_obj, fc):
            X   = df_c[fc].values.astype(float)
            Xp  = pca_obj.transform(X)
            return [{"student_key": str(df_c["student_key"].iloc[i]),
                     "cluster":     int(df_c["cluster"].iloc[i]),
                     "pc1":         round(float(Xp[i,0]),4),
                     "pc2":         round(float(Xp[i,1]),4)}
                    for i in range(len(df_c))]

        fc = art["feat_cols"]
        return {
            "female": proj(df_f, art["pca_f"], fc),
            "male":   proj(df_m, art["pca_m"], fc),
            "ev_female": [round(float(v),4) for v in art["pca_f"].explained_variance_ratio_],
            "ev_male":   [round(float(v),4) for v in art["pca_m"].explained_variance_ratio_],
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/run-pipeline")
async def run_pipeline_ep(bg: BackgroundTasks, data_path: Optional[str] = None):
    def _run():
        from pipeline import run_pipeline
        run_pipeline(data_path=data_path, plot=False)
        art, df, df_f, df_m = load_artefacts()
        ST.update({"art":art,"df":df,"df_f":df_f,"df_m":df_m,"ready":True})
        print("✅ Re-run complete")
    bg.add_task(_run)
    return {"message": "Pipeline re-training queued. Check server logs."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
