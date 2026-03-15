# Hostel-Roommate-Prediction-model
This is a project based on predicting compatibility of hostel roommates using machine learning algorithms.# 🏠 Hostel Roommate Compatibility Model

> A machine learning project that takes behavioural and lifestyle survey responses from hostel students and produces a **ranked list of the 3 most compatible roommate matches** for each individual — driven by data, not guesswork.

---

## 👥 Team

| Name | Role |
|---|---|
| Ariba Naaz | Co-developer |
| Kirti Nagori | Co-developer |
|Sanvitha Reddy | Co-developer |

---

## 📌 Project Overview

Most hostel institutions assign roommates randomly or allow self-selection — both approaches fail to account for behavioural compatibility in any structured way. Incompatibility in sleep schedules, cleanliness habits, noise sensitivity, and lifestyle choices creates real conflicts that affect academic performance and mental wellbeing.

This project builds a data-driven solution: a machine learning model trained on **behavioural survey data** (not self-reported ideals) that outputs the **top 3 most compatible roommate matches** for each student, ranked by a weighted compatibility score.

| Property | Details |
|---|---|
| **Status** | 🔄 Data Collection & Preprocessing Phase |
| **Output** | Top 3 ranked compatibility matches per student |
| **Data Source** | Google Form — behavioural survey |
| **Tools** | Python, Jupyter Notebook, Scikit-learn, Pandas, NumPy |
| **Models Planned** | Cosine Similarity, KNN, K-Means Clustering |

---

## ⚠️ Scope & Limitations

This is a **student learning project** in active development. Important caveats:

- Dataset is still being collected — model has not yet been trained
- Model selection is pending final data volume and distribution analysis
- Labelled training data (known compatible/incompatible pairs) is not available yet, limiting supervised approaches
- Compatibility is multi-dimensional — no model perfectly predicts real-world roommate success
- The output is a ranked suggestion list, not a guarantee — human judgment should still play a role

---

## 🗂️ Repository Structure

```
roommate-compatibility/
│
├── 📁 data/
│   ├── raw/                        # Raw Google Form responses (anonymised)
│   └── processed/                  # Encoded and scaled dataset
│
├── 📁 notebooks/
│   ├── 01_data_cleaning.ipynb      # Cleaning, tokenisation, missing values
│   ├── 02_feature_engineering.ipynb # Encoding and scaling
│   └── 03_model_experiments.ipynb  # Model training and evaluation
│
├── 📁 src/
│   ├── tokenise.py                 # Anonymisation — assign random tokens
│   ├── encode.py                   # Encoding pipeline
│   ├── model.py                    # Compatibility scoring logic
│   └── match.py                    # Ranked output — top 3 matches
│
├── 📁 docs/
│   ├── research.md                 # References and research notes
│   ├── challenges.md               # Problems encountered & solutions
│   ├── encoding_strategy.md        # Full encoding decisions per feature
│   └── survey_design.md            # Form design rationale & anti-bias techniques
│
├── README.md
├── ROADMAP.md
├── requirements.txt

```

---

## 📋 Survey Design

Data is collected via a **Google Form** estimated at 3-5 minutes. The form is structured around **7 behavioural dimensions** — all questions are framed around real scenarios and actual habits, not self-reported ideals.

> 🔑 Core design principle: *"What you actually do" vs "what you think is ideal"* — this distinction drove every form design decision.

### Survey Dimensions

| Dimension | What It Captures |
|---|---|
| **A. Sleep Schedule** | Sleep/wake times, late return behaviour, alarm sensitivity |
| **B. Cleanliness & Hygiene** | Tidying frequency, clutter tolerance, shared space habits |
| **C. Social Energy & Guests** | Guest frequency, visitor departure times, conversation preference |
| **D. Study Habits & Noise** | Study environment needs, noise and audio sensitivity |
| **E. Food, Smoking & Alcohol** | Dietary preferences, smoking habits, alcohol tolerance |
| **F. Conflict Handling** | Response to friction — scenario-based, not hypothetical |
| **G. Sharing & Boundaries** | Item sharing comfort, personal space requirements, dealbreakers |
| **H. Environment** | Temperature, lighting, top dealbreaker flags (used for weighting) |

📎 [View the live Google Form](https://docs.google.com/forms/d/e/1FAIpQLSc0KeZjZ_6HPUYbRpaoyT4PZrDWOAbVfOtBdgNBHtMk8FZZCQ/viewform)

### Anti-Bias Design Techniques

A critical design principle was eliminating **self-reporting bias**:

- **Scenario-based questions** — *"When you return after 11 PM, you..."* instead of *"Would you be quiet?"*
- **Behavioural anchors on scales** — verbal anchors at both ends prevent defaulting to middle values
- **Dual smoking capture** — one question for self-identified smokers, a separate one for situational/peer-pressure smokers
- **Residual sensitivity capture** — a question about smoke smell on clothes, not just in-room smoking
- **Conditional Jain food question** — only shown to Jain respondents, preventing bias for others

---

## ⚙️ Feature Engineering

### Encoding Strategy

| Encoding Method | When Used | Examples |
|---|---|---|
| **Label Encoding** | Ordinal data with natural order | Sleep time, tidying frequency, smell sensitivity |
| **One-Hot Encoding** | Nominal data with no natural order | Late return behaviour, study location, ideal evening |
| **Min-Max Scaling** | Numeric scale responses (1–5) | Conversation level, noise sensitivity, personal space |
| **Binary (0/1)** | Yes/No questions | Peer-pressure smoking |
| **Multi-Hot / Binary Flags** | Multi-select questions | Off-limits items, dealbreaker flags |

### Anonymisation — Tokenisation
Each submission is assigned a unique random token (e.g. `RT-4829-XK`) before processing. No personally identifiable information is collected or stored in the model pipeline. Token-to-response mapping is stored separately and never exposed in outputs.

> ⚠️ Status: Tokenisation is planned — not yet implemented.

---

## 🤖 Model Selection

### Approaches Under Consideration

| Model | Approach | Strength | Limitation |
|---|---|---|---|
| **Cosine Similarity** | Vector similarity between feature profiles | No training required, interpretable | Doesn't capture non-linear interactions |
| **KNN** | Find K nearest neighbours by feature distance | Produces ranked output naturally | Sensitive to feature scaling |
| **K-Means / Hierarchical Clustering** | Group similar profiles into clusters | Good for finding natural groupings | Doesn't directly produce ranked pairs |
| **Random Forest** | Ensemble classification/regression | Can learn complex feature interactions | Requires labelled training data — not available yet |

> 🧠 **Current plan:** Start with Cosine Similarity as a baseline (no training required), then layer KNN for ranked matching. Clustering may be used to segment users before matching within clusters.

### Output
For each student token, the model returns the **top 3 most compatible matches**, ranked by weighted compatibility score. Dealbreaker dimensions flagged by each respondent receive higher weight in the scoring.

---

## 🧩 Challenges & Key Decisions

### 1. Self-Reporting Bias in the Form
**Problem:** Original questions like *"Would you be quiet when coming in late?"* almost universally got *"Yes"* — socially desirable, not behavioural reality.  
**Solution:** Redesigned entire form to use scenario-based questions with realistic options that don't have an obvious "correct" answer.

### 2. Smoking — Single Question Wasn't Enough
**Problem:** Students who smoke occasionally in social situations self-identify as non-smokers. A roommate who smokes outside but carries the smell indoors causes friction even if they technically "don't smoke in the room."  
**Solution:** Added two independent questions — one for peer-pressure/situational smoking, one for residual smell sensitivity.

### 3. Jain Food Complexity
**Problem:** Initial food preference question didn't capture the spectrum of Jain dietary practices and was biasing non-Jain respondents.  
**Solution:** Added a conditional follow-up question comsidering respondents about shared space food strictness.

### 4. Form Version Control Gap
**Problem:** Several form fields were removed during refinement but the changes were not documented at the time.  
**Lesson:** Future form changes will be version-controlled with a change log documenting what was removed and why.

> 📄 Full challenge log: [`docs/challenges.md`](docs/challenges.md)

---

## 🚀 Roadmap

See [`ROADMAP.md`](ROADMAP.md) for full development plan.

**Immediate next steps:**
- [ ] Complete data collection via Google Form
- [ ] Implement tokenisation before any analysis
- [ ] Data cleaning — missing values, response validation
- [ ] Apply encoding strategy (Section 3.2)
- [ ] Apply Min-Max scaling
- [ ] Build baseline Cosine Similarity model
- [ ] Compare with KNN — evaluate ranked output quality
- [ ] Implement dealbreaker weighting

**Longer term:**
- [ ] Simple admin interface for hostel staff
- [ ] Validate output against real-world satisfaction (feedback loop)
- [ ] Add confidence score per match
- [ ] Version 2: optional preference fields as tie-breakers

---

## 🛠️ Setup & Usage

```bash
# Clone the repo
git clone https://github.com/YOUR-USERNAME/roommate-compatibility.git
cd roommate-compatibility

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/01_data_cleaning.ipynb
```

> ⚠️ Raw data is not included in this repo to protect respondent privacy.

---

## 🔒 Ethics & Privacy

- No personally identifiable information is collected in the form
- Tokenisation is applied before any data processing or sharing
- Model output uses only anonymous tokens — no names in results
- Data is used solely for roommate matching purposes

---

## 📚 References

See [`docs/research.md`](docs/research.md) for full references.


This project is open for educational and non-commercial use. See [LICENSE](LICENSE) for details.

---

*Built on behaviour, not assumptions.*
