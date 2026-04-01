# 🧩 Challenges, Mistakes & Solutions

A detailed log of significant problems encountered, design mistakes made, and how they were resolved.

---

## Challenge 1 — Self-Reporting Bias in Form Design

**Phase:** Survey Design
**Status:** ✅ Resolved

### Problem
The original form contained questions phrased as preference or self-assessment questions. Respondents answered based on who they want to be perceived as, not how they actually behave.

> Example: *"Would you be quiet when coming in late?"* → Almost universally gets "Yes." Socially desirable, not behavioural reality.

### Solution
The entire form was redesigned to use scenario-based and behavioural anchor questions. Questions were reworded from *"Would you..."* to *"When X happens, you..."* with realistic options that don't have an obvious correct answer. The form instruction now reads: *"Answer based on what you ACTUALLY do, not what you think is ideal."*

### Lesson Learned
For behavioural data collection, hypothetical preference questions are unreliable. Scenario-based questions with realistic options are significantly more effective.

---

## Challenge 2 — Smoking Questions Were Insufficient

**Phase:** Survey Design
**Status:** ✅ Resolved

### Problem
A single smoking question missed two important cases: students who smoke socially but self-identify as non-smokers, and students who smoke outside but carry the smell indoors.

### Solution
Added a separate option — *"I smoke/drink in presence of peers"* — to capture situational/peer-pressure behaviour. This is encoded as a distinct value in `SMOKE_ORDER` and handled with a dedicated entry in `SMOKE_MATRIX` in preprocessing.

### Lesson Learned
A single question is often not sufficient for complex, nuanced behaviours.

---

## Challenge 3 — Jain Food Spelling Variants in Raw Data

**Phase:** Data Collection → Preprocessing
**Status:** ✅ Resolved

### Problem
The Google Form collected Jain food preference responses with minor spelling variations (`"Jain(no root vegetable)"`, `"Jain (no root vegetable)"`, `"Jain(no root vegetables)"`). These were treated as distinct categories by pandas, breaking the food distribution counts and the `FOOD_MATRIX` lookup.

### Solution
A `JAIN_VARIANTS` list is defined in both `01_eda.py` and `02_preprocessing.py` and replaced with the single canonical form `"Jain (no root vegetables)"` immediately after loading the CSV. This list is maintained in both files and must stay in sync.

### Lesson Learned
Normalise free-text variants immediately on load, before any analysis or encoding. Document the canonical form explicitly.

---

## Challenge 4 — `music_bother` Stored as String in CSV

**Phase:** Data Collection → EDA
**Status:** ✅ Resolved

### Problem
The Google Form stored `music_bother` responses as strings like `"4 - Very bothered"` rather than plain integers. This caused the column to load as `object` dtype, making numeric operations fail silently.

### Solution
Both scripts detect if the column is `object` dtype and extract the leading digit using `str.extract(r"(\d)")` before casting to `Int64`. This is applied before any analysis or encoding.

---

## Challenge 5 - PCA needed to reduce number of parameters

One of the major challenges we faced was the high dimensionality of our dataset. The number of features (parameters) was large, which created multiple issues:

Overfitting risk: With too many features, the model started capturing noise instead of actual patterns.
Increased computational cost: Training time and memory usage increased significantly.
Multicollinearity: Many features were correlated, leading to redundancy and unstable model performance.
Difficulty in visualization and interpretation: It became hard to understand relationships in high-dimensional space.


## Challenge 6 — Ordinal Distance Doesn't Capture Food or Smoking Incompatibility

**Phase:** Preprocessing
**Status:** ✅ Resolved

### Problem
Encoding food preference ordinally (Vegan=1 ... Non-veg=6) and computing `|encoded_A - encoded_B|` as a distance measure produces scores that don't reflect actual compatibility. A Vegan and a Non-veg student are not just "5 steps apart" — they are fundamentally incompatible on shared space food habits.

### Solution
`FOOD_MATRIX` and `SMOKE_MATRIX` replace ordinal distance entirely for these two features. Scores are qualitative assessments (e.g. Jain + Non-veg = 0.1, Non-smoker + regular in-room smoker = 0.0). Hard zeros in `SMOKE_MATRIX` short-circuit the scoring function before any weighted sum is computed.

### Lesson Learned
Not all features can be meaningfully encoded as numbers on a continuous scale.


---

## Challenge 7 — Gender-Imputed Rows Must Be Excluded From Clustering

**Phase:** Preprocessing
**Status:** ✅ Resolved

### Problem
Some respondents didn't provide their gender. Mode-filling these with the most common gender value would fabricate gender data, leading to incorrectly grouped students in gender-split PCA and KMeans.

### Solution
A `gender_imputed` boolean flag column is added during Stage 2. All gender-missing rows are mode-filled so they remain in `df_final` for compatibility scoring, but are explicitly excluded from `df_female_clean` and `df_male_clean` used for PCA and clustering.

---


---

## Ongoing / Open Issues

| Issue | Status |
|---|---|
| `DATA_PATH` is hardcoded to a local Windows path | ⚠️ Each team member must update locally — not committed |
| Tokenisation not yet implemented | 🔲 Planned for Phase 6 ||
