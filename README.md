# 🏠 Hostel Roommate Compatibility Model

> A machine learning project that takes behavioural and lifestyle survey responses from hostel students and produces a ranked list of the 3 most compatible roommate matches for each individual — driven by data, not guesswork.

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

This project builds a data-driven solution: a machine learning pipeline trained on behavioural survey data (not self-reported ideals) that outputs the top 3 most compatible roommate matches for each student, ranked by a weighted compatibility score.

| Property | Details |
|---|---|
| **Status** | Model experiments in progress |
| **Output** | Top 3 ranked compatibility matches per student |
| **Data Source** | Google Form — behavioural survey |
| **Tools** | Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn |
| **Techniques Used** | Ordinal + One-Hot Encoding, MinMax Scaling, Custom Compatibility Matrices, PCA, KMeans Clustering |

---

# 🗂️ Repository Structure

```text
Hostel-Roommate-Prediction-model/
│
├── data/
│   └── responses.csv                 # Behavioural survey dataset
│
├── models/
│   ├── artefacts.pkl                 # Saved preprocessing/model artefacts
│   ├── df_fc.parquet
│   ├── df_mc.parquet
│   ├── df_final.parquet
│   ├── summary.json
│   ├── top3_f.json
│   └── top3_m.json
│
├── api.py                            # API/backend interface
├── clustering.py                     # PCA + KMeans clustering logic
├── config.py                         # Global configs and constants
├── data_loader.py                    # Data loading and cleaning
├── lookup.py                         # Lookup utilities and mappings
├── matching.py                       # Compatibility matching logic
├── pipeline.py                       # Main execution pipeline
├── preprocessing.py                  # Encoding, scaling, feature engineering
├── scoring.py                        # Compatibility scoring system
├── index.html                        # Frontend/demo interface
├── requirements.txt                  # Project dependencies
└── README.md
```

---

## 📋 Survey Design

Data is collected via a Google Form estimated at 2–5 minutes. The form is structured around 8 behavioural dimensions — all questions are framed around real scenarios and actual habits, not self-reported ideals.

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

## ⚖️ Anti-Bias Design Techniques

A critical design principle was eliminating self-reporting bias:

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

---

## ⚙️ Pipeline Overview

The current implementation is modularised into multiple components:

| File | Purpose |
|---|---|
| `data_loader.py` | Loads survey data and performs cleaning |
| `preprocessing.py` | Encoding, scaling, feature transformation |
| `clustering.py` | PCA dimensionality reduction and KMeans clustering |
| `scoring.py` | Weighted compatibility scoring system |
| `matching.py` | Computes and ranks roommate matches |
| `lookup.py` | Helper lookup utilities |
| `pipeline.py` | Main orchestration pipeline |
| `api.py` | Backend/API integration layer |

---

## 🤖 Key Design Decisions

### Custom Compatibility Matrices

Food and smoking use lookup matrices instead of ordinal distance. A Jain student paired with a non-vegetarian receives a score of 0.1, not a scaled numeric distance — because the incompatibility is qualitative, not proportional.

### Gender-Split Weights

Two weight sets (`WEIGHTS_FEMALE`, `WEIGHTS_MALE`) are derived from EDA findings. Female weights are higher on cleanliness and conflict style; male weights are higher on guest timing and evening preference. Both sets are validated to sum to 1.0.

### Hard Filters Before Scoring

Smoking incompatibility (`score = 0.0`) and severe food mismatch (`score < 0.2`) short-circuit the scoring function entirely. No weighted average can rescue a fundamentally incompatible pair on these axes.

### Outlier Handling

Students flagged by Z-score or Mahalanobis distance are matched last or recommended for single-room allocation rather than being forced into a cluster.

---

## 🧩 Challenges & Key Decisions

### Self-Reporting Bias in the Form

**Problem:** Original questions like *"Would you be quiet when coming in late?"* almost universally got *"Yes"* — socially desirable, not behavioural reality.

**Solution:** Redesigned the form using scenario-based questions with realistic options that do not have an obvious "correct" answer.

---

### Smoking — Single Question Wasn't Enough

**Problem:** Students who smoke occasionally in social situations self-identify as non-smokers. A roommate who smokes outside but carries the smell indoors still creates friction even if they technically "don't smoke in the room."

**Solution:** Added two independent questions — one for peer-pressure/situational smoking and another for residual smell sensitivity.

---

## 🔒 Ethics & Privacy

- No personally identifiable information is collected in the form
- Raw responses are anonymised
- Model output uses anonymous indices/tokens
- Data is used solely for roommate matching purposes

---

## 🚀 Roadmap

### Immediate Next Steps

- [ ] Complete data collection via Google Form
- [ ] Improve compatibility weighting
- [ ] Optimise clustering performance
- [ ] Add confidence scoring
- [ ] Improve ranked matching quality
- [ ] Compare KNN vs Cosine Similarity performance

### Long-Term Goals

- [ ] Admin dashboard for hostel allocation
- [ ] Real-world validation feedback loop
- [ ] Web deployment
- [ ] Automated recommendation interface

---

# 🛠️ Setup & Usage

```bash
# Clone the repository
git clone https://github.com/Ariba-Naaz/Hostel-Roommate-Prediction-model.git

# Move into project directory
cd Hostel-Roommate-Prediction-model/"Main folder"

# Create virtual environment
py -m venv venv

# Activate virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python pipeline.py
```

> ⚠️ Raw data is included in this repository with name and id tokenised to protect respondent privacy.

---

## 📦 Dependencies

Major libraries used:

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

