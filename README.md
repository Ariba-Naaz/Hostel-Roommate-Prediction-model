
# 🏠 Hostel Roommate Compatibility Model

> A machine learning project that takes behavioural and lifestyle survey responses from hostel students and produces a **ranked list of the 3 most compatible roommate matches** for each individual — driven by data, not guesswork.

**Institution:** Malaviya National Institute of Technology, Jaipur

---

## 👥 Team

| Name | Role |
|---|---|
| Ariba Naaz | Co-developer |
| Kirti Nagori | Co-developer |
| Sanvitha Reddy | Co-developer |

---

## 📌 Project Overview

Most hostel institutions assign roommates randomly or allow self-selection — both approaches fail to account for behavioural compatibility in any structured way. Incompatibility in sleep schedules, cleanliness habits, noise sensitivity, and lifestyle choices creates real conflicts that affect academic performance and mental wellbeing.

This project builds a data-driven solution: a machine learning pipeline trained on **behavioural survey data** (not self-reported ideals) that outputs the **top 3 most compatible roommate matches** for each student, ranked by a weighted compatibility score.

| Property | Details |
|---|---|
| **Status** | Model experiments in progress |
| **Output** | Top 3 ranked compatibility matches per student |
| **Data Source** | Google Form — behavioural survey |
| **Tools** | Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn |
| **Techniques Used** | Ordinal + One-Hot Encoding, MinMax Scaling, Custom Compatibility Matrices, PCA, KMeans Clustering |

---

## 🗂️ Repository Structure

```
roommate-compatibility/
│
├── 📁 data/
│   └── raw/                        # Raw Google Form responses (anonymised)
│
├── 📁 Code/
│   ├── 01_eda.py                   # EDA — distributions, gender split, dealbreaker analysis,
│   │                               # cross-tabs, correlation heatmap, outlier detection
│   ├── 02_preprocessing.py         # Encoding, scaling, custom compatibility matrices,
│   │                               # compatibility scoring function, PCA, KMeans clustering
│   └── 03_model_experiments.py     # (In progress) Matching loop, top-3 output per student
│
├── 📁 docs/
│   ├── challenges.md               # Problems encountered & solutions
│   ├── encoding_strategy.md        # Full encoding decisions per feature
│   └── survey_design.md            # Form design rationale & anti-bias techniques
│
├── README.md
├── ROADMAP.md
├── CONTRIBUTING.md
└── requirements.txt
```

---

## 📋 Survey Design

Data is collected via a **Google Form** estimated at 2–5 minutes. The form is structured around **8 behavioural dimensions** — all questions are framed around real scenarios and actual habits, not self-reported ideals.

> 🔑 Core design principle: *"What you actually do"* vs *"what you think is ideal"* — this distinction drove every form design decision.

### Survey Dimensions

| Dimension | What It Captures |
|---|---|
| **A. Sleep Schedule** | Sleep/wake times, late return behaviour |
| **B. Cleanliness & Hygiene** | Tidying frequency, clutter tolerance |
| **C. Social Energy & Guests** | Guest frequency, visitor departure times, conversation preference |
| **D. Study Habits & Noise** | Study environment needs, music/audio sensitivity |
| **E. Food, Smoking & Alcohol** | Dietary preferences, smoking habits |
| **F. Conflict Handling** | Response to friction — scenario-based |
| **G. Sharing & Boundaries** | Item sharing comfort, personal space |
| **H. Environment & Dealbreakers** | Temperature, lighting, top 3 ranked dealbreaker flags |

📎 [View the live Google Form](https://docs.google.com/forms/d/e/1FAIpQLSc0KeZjZ_6HPUYbRpaoyT4PZrDWOAbVfOtBdgNBHtMk8FZZCQ/viewform)

---
### Anti-Bias Design Techniques

A critical design principle was eliminating **self-reporting bias**:

- **Scenario-based questions** — *"When you return after 11 PM, you..."* instead of *"Would you be quiet?"*
- **Behavioural anchors on scales** — verbal anchors at both ends prevent defaulting to middle values
- **Dual smoking capture** — one question for self-identified smokers, a separate one for situational/peer-pressure smokers
- **Residual sensitivity capture** — a question about smoke smell on clothes, not just in-room smoking
- **Conditional Jain food question** — only shown to Jain respondents, preventing bias for others
---

## 🤖 Model Selection

### Approaches 

| Model | Approach | Strength | Limitation |
|---|---|---|---|
| **Cosine Similarity** | Vector similarity between feature profiles | No training required, interpretable | Doesn't capture non-linear interactions |
| **KNN** | Find K nearest neighbours by feature distance | Produces ranked output naturally | Sensitive to feature scaling |
| **K-Means / Hierarchical Clustering** | Group similar profiles into clusters | Good for finding natural groupings | Doesn't directly produce ranked pairs |
| **Random Forest** | Ensemble classification/regression | Can learn complex feature interactions | Requires labelled training data — not available yet |


## ⚙️ Pipeline Overview

### Stage 1 — EDA (`01_eda.py`)

Nine analysis cells run independently on the raw CSV:

- **Sleep & wake distributions** — identifies night owl concentration
- **Dealbreaker frequency analysis** — across all 3 priority slots, and split by gender
- **Food preference breakdown** — including non-veg sensitivity among veg students
- **Cross-tab analysis** — sleep vs evening preference, cleanliness vs conflict style, study env vs music bother, evening pref vs guest timing (full dataset + gender split)
- **Correlation heatmap** — Pearson r across all ordinal-encoded features
- **Outlier detection** — Z-score flagging (≥4 extreme columns) + Mahalanobis distance (χ² p=0.01)
- **Final summary** — matching implications derived from EDA findings

### Stage 2 — Preprocessing (`02_preprocessing.py`)

Eight pipeline stages:

1. **Load & clean** — column renaming, Jain spelling normalisation, `music_bother` string parsing
2. **Missing value handling** — mode fill (categorical), median fill (numeric), `gender_imputed` flag
3. **Encoding** — ordinal encoding (12 columns), one-hot encoding (`food_pref`, `study_env`, `gender`), dealbreaker binary flags (7 columns)
4. **Normalisation** — MinMaxScaler on ordinal + numeric columns only; OHE and binary columns excluded
5. **Custom compatibility matrices** — `FOOD_MATRIX` and `SMOKE_MATRIX` replace ordinal distance for food and smoking, capturing qualitative incompatibility that numeric distance misses
6. **Compatibility scoring** — `total_compatibility(idx_a, idx_b)` returns a score in [0.0, 1.0] with hard filters (smoking incompatibility → 0.0, food score < 0.2 → 0.0) and gender-split weight sets derived from EDA
7. **PCA** — separate per gender, retaining 90% of variance; used for clustering only, not scoring
8. **KMeans clustering** — separate per gender, elbow + silhouette analysis, currently k=4 for both

### Stage 3 — Model Experiments (`03_model_experiments.py`)
*(In progress)*

- Run `total_compatibility()` for all pairs within each cluster
- Surface top 3 matches per student token
- Add confidence score per match

---

## 🤖 Key Design Decisions

### Custom Compatibility Matrices
Food and smoking use lookup matrices instead of ordinal distance. A Jain student paired with a non-vegetarian receives a score of 0.1, not a scaled numeric distance — because the incompatibility is qualitative, not proportional.

### Gender-Split Weights
Two weight sets (`WEIGHTS_FEMALE`, `WEIGHTS_MALE`) are derived from EDA findings. Female weights are higher on cleanliness and conflict style; male weights are higher on guest timing and evening preference. Both sets are validated to sum to 1.0.

### Hard Filters Before Scoring
Smoking incompatibility (score = 0.0) and severe food mismatch (score < 0.2) short-circuit the scoring function entirely. No weighted average can rescue a fundamentally incompatible pair on these axes.

### Outlier Handling
Students flagged by Z-score or Mahalanobis distance are matched last or recommended for single-room allocation rather than being forced into a cluster.
---

## 🧩 Challenges & Key Decisions

### Self-Reporting Bias in the Form
**Problem:** Original questions like *"Would you be quiet when coming in late?"* almost universally got *"Yes"* — socially desirable, not behavioural reality.  
**Solution:** Redesigned entire form to use scenario-based questions with realistic options that don't have an obvious "correct" answer.

### Smoking — Single Question Wasn't Enough
**Problem:** Students who smoke occasionally in social situations self-identify as non-smokers. A roommate who smokes outside but carries the smell indoors causes friction even if they technically "don't smoke in the room."  
**Solution:** Added two independent questions — one for peer-pressure/situational smoking, one for residual smell sensitivity.

> 📄 Full challenge log: [`docs/challenges.md`](challenges.md)

---

## 🔒 Ethics & Privacy

- No personally identifiable information is collected in the form
- Tokenisation planned before any data processing or external sharing
- Model output uses only anonymous indices — no names in results
- Data is used solely for roommate matching purposes

---

## 🚀 Roadmap

See [`ROADMAP.md`](ROADMAP.md) for full development plan.

**Immediate next steps:**
- [ ] Complete data collection via Google Form
- [ ] Data cleaning — missing values, response validation
- [ ] Apply encoding strategy
- [ ] Implement dealbreaker weighting(PCA and custom matrix)
- [ ] Apply Min-Max scaling
- [ ] Implement K Means for clustering.
- [ ] Build baseline Cosine Similarity model
- [ ] Compare with KNN — evaluate ranked output quality


**Longer term:**
- [ ] Simple admin interface for hostel staff
- [ ] Validate output against real-world satisfaction (feedback loop)
- [ ] Add confidence score per match
## 🛠️ Setup & Usage

```bash
# Clone the repo
git clone https://github.com/Ariba-Naaz/Hostel-Roommate-Prediction-model.git
cd Hostel-Roommate-Prediction-model

# Install dependencies
pip install -r requirements.txt

# Run in order
python Code/01_eda.py
python Code/02_preprocessing.py
# 03_model_experiments.py — in progress
```

> ⚠️ Update `DATA_PATH` in both scripts to point to your local CSV before running.  
> ⚠️ Raw data is not included in this repo to protect respondent privacy.

---
