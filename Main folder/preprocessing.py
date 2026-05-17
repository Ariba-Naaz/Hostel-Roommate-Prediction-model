# =============================================================
# preprocessing.py  —  encode + scale, no dealbreakers
# =============================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from config import ORDINAL_COLS, OHE_COLS


def build_encoders(df: pd.DataFrame):
    """Fit and return (ord_enc, ohe, scaler)."""
    ord_names  = [c for c, _ in ORDINAL_COLS]
    ord_orders = [o for _, o in ORDINAL_COLS]

    ord_enc = OrdinalEncoder(
        categories=ord_orders,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    ord_enc.fit(df[ord_names])

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(df[OHE_COLS])

    # Fit scaler on ordinal + numeric cols
    tmp        = _assemble_raw(df, ord_enc, ohe, ord_names)
    scale_cols = ord_names + ["convo_level", "music_bother"]
    scaler     = MinMaxScaler()
    scaler.fit(tmp[scale_cols])
    return ord_enc, ohe, scaler


def _assemble_raw(df, ord_enc, ohe, ord_names):
    ohe_names = ohe.get_feature_names_out(OHE_COLS).tolist()
    df_ord = pd.DataFrame(
        ord_enc.transform(df[ord_names]),
        columns=ord_names, index=df.index,
    )
    df_ohe = pd.DataFrame(
        ohe.transform(df[OHE_COLS]),
        columns=ohe_names, index=df.index,
    )
    df_num = df[["convo_level", "music_bother"]].copy().astype(float)
    return pd.concat([df_ord, df_ohe, df_num], axis=1)


def encode_and_scale(df, ord_enc, ohe, scaler):
    """
    Returns (df_final, ohe_names, scale_cols, noscale_cols).
    df_final has all scaled features + metadata columns.
    """
    ord_names    = [c for c, _ in ORDINAL_COLS]
    ohe_names    = ohe.get_feature_names_out(OHE_COLS).tolist()
    scale_cols   = ord_names + ["convo_level", "music_bother"]
    noscale_cols = ohe_names

    assembled = _assemble_raw(df, ord_enc, ohe, ord_names)
    scaled_df = pd.DataFrame(
        scaler.transform(assembled[scale_cols]),
        columns=scale_cols, index=df.index,
    )
    df_final = pd.concat([scaled_df, assembled[noscale_cols]], axis=1)

    # Attach metadata needed by scorer and API
    df_final["food_pref_raw"]    = df["food_pref"].values
    df_final["smoke_drink_raw"]  = df["smoke_drink"].values
    df_final["gender"]           = df["gender"].values
    df_final["gender_imputed"]   = df["gender_imputed"].values
    df_final["student_key"]      = df["student_key"].values
    df_final["display_name"]     = df["display_name"].values

    print(f"✅ Encoded shape: {df_final.shape}")
    return df_final, ohe_names, scale_cols, noscale_cols


def encode_single(raw: dict, ord_enc, ohe, scaler,
                  ohe_names, scale_cols, noscale_cols) -> pd.Series:
    """
    Encode a single student dict (e.g. from API POST body).
    Returns a pd.Series with the same columns as df_final.
    """
    ord_names  = [c for c, _ in ORDINAL_COLS]
    orders_map = {c: o for c, o in ORDINAL_COLS}

    # ordinal
    ord_input = pd.DataFrame(
        [[raw.get(c, orders_map[c][0]) for c in ord_names]],
        columns=ord_names,
    )
    df_ord = pd.DataFrame(ord_enc.transform(ord_input), columns=ord_names)

    # ohe
    ohe_input = pd.DataFrame(
        [[raw.get("food_pref", "No preference"),
          raw.get("study_env", "I use headphones, so room noise doesn't matter"),
          raw.get("gender", "Male")]],
        columns=OHE_COLS,
    )
    df_ohe = pd.DataFrame(ohe.transform(ohe_input), columns=ohe_names)

    # numeric
    df_num = pd.DataFrame(
        [[float(raw.get("convo_level", 3)), float(raw.get("music_bother", 3))]],
        columns=["convo_level", "music_bother"],
    )

    combined = pd.concat([df_ord, df_ohe, df_num], axis=1)
    scaled   = pd.DataFrame(
        scaler.transform(combined[scale_cols]), columns=scale_cols
    )
    encoded  = pd.concat([scaled, combined[noscale_cols]], axis=1)

    row = encoded.iloc[0].copy()
    row["food_pref_raw"]   = raw.get("food_pref",  "No preference")
    row["smoke_drink_raw"] = raw.get("smoke_drink", "I don't smoke/drink at all")
    row["gender"]          = raw.get("gender",      "Male")
    row["gender_imputed"]  = False
    row["student_key"]     = raw.get("roll_no", "TEMP").upper().strip()
    row["display_name"]    = raw.get("name", "New Student") + \
                             " (" + row["student_key"] + ")"
    return row
