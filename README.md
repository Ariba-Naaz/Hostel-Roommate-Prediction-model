# 🏠 Hostel Roommate Compatibility Model

> A machine learning project that takes behavioural and lifestyle survey responses from hostel students and produces a ranked list of the 3 most compatible roommate matches for each individual — driven by data, not guesswork.

**Institution:** Malaviya National Institute of Technology, Jaipur

---

## 👥 Team

| Name           | Role         |
| -------------- | ------------ |
| Ariba Naaz     | Co-developer |
| Kirti Nagori   | Co-developer |
| Sanvitha Reddy | Co-developer |

---

## 📌 Project Overview

Most hostel institutions assign roommates randomly or allow self-selection — both approaches fail to account for behavioural compatibility in any structured way. Incompatibility in sleep schedules, cleanliness habits, noise sensitivity, and lifestyle choices creates real conflicts that affect academic performance and mental wellbeing.

This project builds a data-driven solution: a machine learning pipeline trained on behavioural survey data (not self-reported ideals) that outputs the top 3 most compatible roommate matches for each student, ranked by a weighted compatibility score.

| Property            | Details                                                                                            |
| ------------------- | -------------------------------------------------------------------------------------------------- |
| **Status**          | Active development                                                                                 |
| **Output**          | Top 3 ranked compatibility matches per student                                                     |
| **Data Source**     | Google Form — behavioural survey                                                                   |
| **Tools**           | Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn                                           |
| **Techniques Used** | Ordinal + One-Hot Encoding, MinMax Scaling, PCA, KMeans Clustering, Weighted Compatibility Scoring |

---

# 🗂️ Repository Structure

```text
Hostel-Roommate-Prediction-model/
│
├── Main folder/
│   │
│   ├── data/                           # Behavioural survey datasets and cleaned inputs
│   │
│   ├── models/                         # Saved model artefacts, processed outputs, match results
│   │
│   ├── api.py                          # API/backend interface for serving roommate recommendations
│   │
│   ├── clustering.py                   # PCA dimensionality reduction and KMeans clustering logic
│   │
│   ├── config.py                       # Global constants, feature weights, compatibility settings
│   │
│   ├── data_loader.py                  # Dataset loading, validation, and cleaning utilities
│   │
│   ├── lookup.py                       # Lookup tables and categorical mapping helpers
│   │
│   ├── matching.py                     # Core roommate matching and ranking engine
│   │
│   ├── pipeline.py                     # Main orchestration pipeline connecting all modules
│   │
│   ├── preprocessing.py                # Encoding, scaling, preprocessing, and feature engineering
│   │
│   ├── scoring.py                      # Weighted compatibility scoring calculations
│   │
│   ├── index.html                      # Frontend/demo interface for displaying results
│   │
│   ├── requirements.txt                # Python dependencies required for the project
│   │
│   └── help/                           # Helper scripts, utilities, and experimental files
│
├── eda_figures/                        # Generated EDA visualizations and statistical plots
│   │
│   ├── 3_sleep_wake.png                # Sleep/wake behavioural distribution analysis
│   ├── 4_dealbreakers.png              # Dealbreaker preference analysis
│   ├── 5_A_food_overall.png            # Overall food preference visualization
│   ├── 6_A_full_crosstabs.png          # Cross-tab behavioural relationship plots
│   ├── 7_correlation_heatmap.png       # Feature correlation heatmap
│   └── 8_outlier_detection.png         # Outlier detection visualization
│
├── eda_plots/                          # Additional EDA outputs and grouped visualization folders
│   │
│   ├── cross_tab_figures/              # Detailed behavioural cross-tab comparison plots
│   │
│   ├── 01_sleep_wake.png               # Summary sleep behaviour plot
│   ├── 02_dealbreakers.png             # Summary dealbreaker analysis
│   ├── 03_food.png                     # Food preference summary visualization
│   ├── 04_crosstabs.png                # Behavioural interaction summaries
│   ├── 05_correlation.png              # Correlation analysis summary
│   └── 06_outliers.png                 # Outlier summary visualization
│
├── Docs/                               # Project documentation and design rationale
│   │
│   ├── challenges.md                   # Challenges faced and engineering decisions
│   ├── encoding_strategy.md            # Encoding methodology and preprocessing rationale
│   └── survey_design.md                # Survey design principles and behavioural framing
│
├── eda.py                              # Exploratory Data Analysis generation pipeline
│
├── CONTRIBUTING.md                     # Contribution guidelines for collaborators
│
├── ROADMAP.md                          # Planned features and future development goals
│
├── README.md                           # Main project documentation
│
└── .gitignore                          # Ignored files and folders for Git version control
```


## 📋 Survey Design

Data is collected via a Google Form estimated at 2–5 minutes. The form is structured around behavioural dimensions focused on real habits and lifestyle patterns rather than idealised responses.

> 🔑 Core design principle: *"What you actually do"* vs *"what you think is ideal"* — this distinction drove the survey design.

### Survey Dimensions

| Dimension                   | What It Captures                                |
| --------------------------- | ----------------------------------------------- |
| **Sleep Schedule**          | Sleep/wake times, late return behaviour         |
| **Cleanliness & Hygiene**   | Tidying frequency, clutter tolerance            |
| **Social Energy & Guests**  | Guest frequency, visitor departure times        |
| **Study Habits & Noise**    | Study environment preference, music sensitivity |
| **Food & Smoking**          | Dietary preferences and smoking compatibility   |
| **Conflict Handling**       | Behaviour during disagreements                  |
| **Sharing & Boundaries**    | Sharing comfort and personal space              |
| **Environment Preferences** | Temperature, lighting, dealbreakers             |

📎 [View the live Google Form](https://docs.google.com/forms/d/e/1FAIpQLSc0KeZjZ_6HPUYbRpaoyT4PZrDWOAbVfOtBdgNBHtMk8FZZCQ/viewform)

---

## 📊 Exploratory Data Analysis (EDA)

The project includes extensive exploratory analysis and visualization pipelines to identify behavioural trends and compatibility patterns.

### Included Analysis

* Sleep/wake behaviour distributions
* Food preference analysis
* Smoking sensitivity analysis
* Gender-based behavioural comparisons
* Cross-tab behavioural interactions
* Correlation heatmaps
* Outlier detection
* Conflict-style analysis

### Visualization Outputs

Generated visualizations are stored inside:

* `eda_figures/`
* `eda_plots/`
* `cross_tab_figures/`

---

## ⚖️ Anti-Bias Design Techniques

Several measures were used to reduce self-reporting bias:

* Scenario-based questions instead of idealised yes/no responses
* Behavioural anchors on scales
* Separate situational smoking capture
* Residual smoke sensitivity questions
* Conditional dietary questions

---

## ⚙️ Machine Learning Pipeline

### Preprocessing

* Missing value handling
* Ordinal Encoding
* One-Hot Encoding
* MinMax Scaling

### Dimensionality Reduction

* Principal Component Analysis (PCA)

### Clustering

* KMeans Clustering

### Compatibility Logic

* Weighted scoring system
* Custom compatibility matrices
* Hard incompatibility filters

---

## 🤖 Model Components

| File               | Purpose                            |
| ------------------ | ---------------------------------- |
| `data_loader.py`   | Loads and cleans survey data       |
| `preprocessing.py` | Feature encoding and scaling       |
| `clustering.py`    | PCA and KMeans implementation      |
| `scoring.py`       | Weighted compatibility scoring     |
| `matching.py`      | Match generation and ranking       |
| `lookup.py`        | Lookup utilities and mappings      |
| `pipeline.py`      | Main orchestration pipeline        |
| `api.py`           | API/backend interface              |
| `eda.py`           | Exploratory data analysis pipeline |

---

## 🧠 Key Design Decisions

### Custom Compatibility Matrices

Food and smoking compatibility are computed using domain-specific lookup matrices rather than direct ordinal distance.

### Weighted Compatibility Scoring

Different behavioural features contribute with different importance levels toward the final compatibility score.

### Hard Compatibility Filters

Critical incompatibilities such as severe smoking or food preference mismatches are filtered before weighted scoring.

### PCA Before Clustering

Dimensionality reduction helps reduce noise and improve clustering quality.

---

## 🔒 Ethics & Privacy

* Personally identifiable information is excluded
* Survey responses are anonymised
* Outputs use anonymous identifiers
* Data is used solely for roommate compatibility research

---

## 🚀 Roadmap

### Immediate Goals

* [ ] Improve compatibility weighting
* [ ] Compare KNN vs cosine similarity
* [ ] Improve clustering performance
* [ ] Add confidence scoring
* [ ] Enhance recommendation quality

### Future Goals

* [ ] Real-time hostel allocation system
* [ ] Web deployment
* [ ] Admin dashboard
* [ ] Feedback-driven recommendation refinement
* [ ] Deep learning-based matching experiments

---

# 🛠️ Setup & Usage

```bash
# Clone repository
git clone https://github.com/Ariba-Naaz/Hostel-Roommate-Prediction-model.git

# Move into project directory
cd Hostel-Roommate-Prediction-model/"Main folder"

# Create virtual environment
py -m venv venv

# Activate virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run main pipeline
python pipeline.py

# Run API
python api.py
```

---

## 📦 Dependencies

Major libraries used:

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 📄 Additional Documentation

Additional project documentation is available inside the `Docs/` folder:

* `survey_design.md`
* `encoding_strategy.md`
* `challenges.md`

---

## 🤝 Contributing

See `CONTRIBUTING.md` for contribution guidelines.

---

## 📌 Project Status

The project is currently under active development and experimentation. Model tuning, compatibility refinement, and evaluation pipelines are continuously being improved.

---
