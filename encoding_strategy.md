# ⚙️ Encoding Strategy — Per Feature

Documents the actual encoding method implemented in `02_preprocessing.py` for every feature.

---

## Encoding Methods Used

| Method | When Used |
|---|---|
| **Ordinal Encoding** | Columns with a natural rank order — encoded as integers 0, 1, 2, ... |
| **One-Hot Encoding** | Nominal columns with no meaningful order |
| **MinMax Scaling** | Ordinal + numeric columns — scaled to [0.0, 1.0] after encoding |
| **Binary Flag (0/1)** | Dealbreaker selections — one column per valid dealbreaker option |
| **Custom Matrix Score** | Food and smoking — qualitative compatibility score, not numeric distance |

> OHE and binary flag columns are **not scaled** — they are already in [0, 1].

---

## Ordinal Encoded Columns (12)

Implemented via `sklearn.OrdinalEncoder` with explicit category orders. `handle_unknown="use_encoded_value"` with `unknown_value=-1`.

| Column | Order (0 → N) | Notes |
|---|---|---|
| `sleep_time` | 10 PM–11:30 PM → 11:30 PM–1 AM → 1 AM–3 AM → After 3 AM → Varies wildly | Later = higher value |
| `wake_time` | Before 6 AM → 6–8 AM → 8–10 AM → After 10 AM | Later = higher value |
| `cleanliness` | Daily → Every 2–3 days → Weekly → When messy → Rarely/Never | Cleaner = lower value |
| `late_return` | Almost never → Phone torch quiet → Stay out → Main light on | More considerate = lower value |
| `conflict_style` | Stay quiet → Dropped hint → Direct conversation → Set boundary → Third party | Avoidance → escalation |
| `sharing` | Never ask first → Small things only → Most items fine → Anything fine | More private = lower value |
| `nonveg_sensitivity` | Very (dealbreaker) → Prefer not → Doesn't bother → I eat non-veg | More sensitive = lower value |
| `guest_leaves` | Don't bring friends → 10 PM → Midnight → 2 AM or later | Earlier = lower value |
| `evening_pref` | Alone dim silence → Low-key one person → Friends hangout → Out with people | More introverted = lower value |
| `light_pref` | Complete darkness → Dim lamp → Main light okay → Sleep mask | Darker preference = lower value |
| `temp_pref` | Cool (18–20°C) → Comfortable (22–24°C) → Warm (26°C+) → No preference | Cooler = lower value |
| `smoke_drink` | Don't at all → Outside only → In presence of peers → Sometimes in room → Regularly in shared spaces | Non-smoker = lower value |

---

## One-Hot Encoded Columns (3)

Implemented via `sklearn.OneHotEncoder(sparse_output=False, handle_unknown="ignore")`.

| Column | Reason for OHE | Output columns |
|---|---|---|
| `food_pref` | No meaningful ordinal rank between categories | `food_pref_Strictly vegetarian`, `food_pref_Jain...`, `food_pref_Vegan`, `food_pref_Vegetarian + eggs`, `food_pref_No preference`, `food_pref_Non-vegetarian` |
| `study_env` | "I use headphones" breaks any ordinal rank | `study_env_Complete silence...`, `study_env_Quiet...`, `study_env_I use headphones...`, `study_env_I can study through...` |
| `gender` | Nominal — Female/Male/Other | `gender_Female`, `gender_Male`, `gender_Other` etc. |

> `food_pref` and `smoke_drink` are **also** handled by custom matrices in the compatibility scoring function (Stage 5/6). The OHE/ordinal encoding is used in PCA and clustering; the matrix scores are used in `total_compatibility()`.

---

## Dealbreaker Binary Flags (7)

One column per valid dealbreaker option. A student gets a `1` in a column if they selected that dealbreaker in **any** of their 3 slots.

| Flag Column | Dealbreaker |
|---|---|
| `db_smoking_habits` | Smoking Habits |
| `db_sleep_schedule` | Sleep Schedule |
| `db_cleanliness` | Cleanliness |
| `db_loud_noise___music` | Loud Noise / Music |
| `db_frequent_guests_at_night` | Frequent guests at night |
| `db_lack_of_privacy` | Lack of privacy |
| `db_roommate_avoids_all_conflict` | Roommate avoids all conflict |

> Note: "Temperature" was listed as a dealbreaker option in the survey design docs but is **not** in `VALID_DEALBREAKERS` in the current code. Confirm whether to add it.

---

## Numeric Columns (2)

Kept as raw numeric values, then MinMax scaled alongside ordinal columns.

| Column | Scale | Notes |
|---|---|---|
| `convo_level` | 1–10 | Self-rated desired conversation level with roommate |
| `music_bother` | 1–5 | Parsed from string format `"4 - Very bothered"` if needed |

---

## Custom Compatibility Matrices (Stage 5)

Used **only** in `total_compatibility()` — not for PCA or clustering.

### `FOOD_MATRIX`
Symmetric lookup returning a score in [0.0, 1.0] for each food preference pair. Key entries:

| Pair | Score |
|---|---|
| Same category (any) | 1.0 |
| Strictly veg + Jain | 0.9 |
| Strictly veg + Non-veg | 0.2 |
| Jain + Non-veg | 0.1 |
| Vegan + Non-veg | 0.1 |
| No preference + Non-veg | 0.9 |

Food scores < 0.2 trigger a hard zero in `total_compatibility()` — the pair is incompatible regardless of all other scores.

### `SMOKE_MATRIX`
Symmetric lookup returning a score in [0.0, 1.0] for each smoking pair. Key entries:

| Pair | Score |
|---|---|
| Non-smoker + Non-smoker | 1.0 |
| Non-smoker + Outside only | 0.8 |
| Non-smoker + In room sometimes | 0.1 |
| Non-smoker + Regularly in shared spaces | 0.0 |
| Outside only + In room sometimes | 0.3 |

A score of 0.0 triggers an immediate return of 0.0 from `total_compatibility()`.

---






