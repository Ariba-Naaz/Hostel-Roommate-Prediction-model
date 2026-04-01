# 🗺️ Project Roadmap — Hostel Roommate Compatibility Model

---

## ✅ Phase 1 — Ideation & Survey Design *(Complete)*

- [x] Define problem statement and project goal
- [x] Identify 8 behavioural dimensions to capture
- [x] Design Google Form with anti-bias techniques
- [x] Implement scenario-based questions (not preference-based)
- [x] Add dual smoking capture — self-identified vs peer-pressure/situational
- [x] Add conditional Jain food follow-up question
- [x] Finalise 3-slot ranked dealbreaker flags for model weighting
- [x] Document encoding strategy per feature

---

## ✅ Phase 2 — Data Collection *(Complete)*

- [x] Distribute Google Form to hostel students (MNIT Jaipur)
- [x] Collect responses via Google Forms CSV export
- [x] Validate response distribution across all dimensions
- [x] Normalise inconsistent category spellings (e.g. Jain variants)

---

## ✅ Phase 3 — EDA (`01_eda.py`) *(Complete)*

- [x] Load and clean raw CSV — column renaming, type parsing, missing value handling
- [x] Gender split — `df_female` / `df_male` (missing gender rows excluded from split charts)
- [x] Sleep & wake time distribution analysis
- [x] Dealbreaker frequency analysis — all 3 priority slots, full dataset and gender split
- [x] Food preference breakdown — overall, by gender, non-veg sensitivity among veg students
- [x] Cross-tab analysis — 4 pairs on full dataset, all 5 pairs repeated by gender
- [x] Correlation heatmap — Pearson r across all ordinal-encoded features
- [x] Outlier detection — Z-score (≥4 extreme columns) + Mahalanobis distance (χ² p=0.01)
- [x] Final summary report — matching implications, gender differences, flagged student count
- [x] All figures saved to `eda_figures/`

---

## ✅ Phase 4 — Preprocessing (`02_preprocessing.py`) *(Complete)*

- [x] Load & clean — consistent with EDA, `gender_imputed` flag added
- [x] Missing value handling — mode fill (categorical), median fill (numeric), `"None"` for dealbreaker slots
- [x] Ordinal encoding — 12 columns via `OrdinalEncoder` with explicit category orders
- [x] One-hot encoding — `food_pref`, `study_env`, `gender`
- [x] Dealbreaker binary flags — 7 columns, one per valid dealbreaker option
- [x] MinMax normalisation — ordinal + numeric columns only; OHE and binary excluded
- [x] Custom food compatibility matrix (`FOOD_MATRIX`) — qualitative scoring, not ordinal distance
- [x] Custom smoking compatibility matrix (`SMOKE_MATRIX`) — hard zero for full incompatibility
- [x] `total_compatibility(idx_a, idx_b)` scoring function — hard filters + gender-split weights
- [x] Weight sets validated — `WEIGHTS_FEMALE` and `WEIGHTS_MALE` both sum to 1.0
- [x] PCA — separate per gender, 90% variance retained, scree plots generated
- [x] KMeans clustering — separate per gender, elbow + silhouette analysis, k=4 both genders
- [x] Cluster profile summary — mode answer per key column per cluster
- [x] Cluster scatter plots — PC1 vs PC2, named archetypes per cluster

---

## 🔄 Phase 5 — Model Experiments (`03_model_experiments.py`) *(In Progress)*

- [ ] Run `total_compatibility()` for all pairs within each cluster
- [ ] Generate ranked compatibility list per student
- [ ] Surface top 3 matches per student token
- [ ] Add confidence score per match
- [ ] Handle edge cases — clusters with very few students
- [ ] Review match quality manually — do top 3 make intuitive sense?
- [ ] Tune dealbreaker weight multipliers if output quality is poor

---

## 🔲 Phase 6 — Evaluation & Refinement

- [ ] Implement tokenisation — assign random tokens (e.g. `RT-4829-XK`) before any output
- [ ] Evaluate match quality across different clusters
- [ ] Tune k and weight multipliers with justification
- [ ] Document final chosen parameters

---

## 🔭 Future (Post-Project)

- **Admin Interface** — simple UI for hostel staff to upload responses and receive ranked match outputs
- **Feedback Loop** — validate output against real-world roommate satisfaction after assignment
- **Version 2** — optional preference fields (hobbies, academic stream) as secondary tie-breakers
- **Ensemble Research** — Random Forest for semi-supervised matching once labelled pairs are available




