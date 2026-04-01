# 🤝 Contributing — Team Guide

This guide is for the three team members working on this project.

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Ariba-Naaz/Hostel-Roommate-Prediction-model.git
cd Hostel-Roommate-Prediction-model
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your data path
Both `01_eda.py` and `02_preprocessing.py` have a `DATA_PATH` variable at the top. Update it to point to your local copy of the CSV before running either script.

---

## How to Work Without Overwriting Each Other

### Always pull before you start working
```bash
git pull origin main
```

### Create a branch for your work
```bash
git checkout -b your-name/what-youre-doing

# Examples:
git checkout -b ariba/model-experiments
git checkout -b kirti/evaluation
git checkout -b sanvitha/confidence-score
```

### Commit your changes
```bash
git add .
git commit -m "Short description of what you did"
```

### Push your branch
```bash
git push origin your-name/what-youre-doing
```

### Open a Pull Request on GitHub
- Go to the repo on GitHub
- Click **"Compare & pull request"**
- Write a short description of what you changed
- Tag a teammate to review
- Merge only when approved

---

## Current Work Division

| Area | Status | Owner |
|---|---|---|
| EDA (`01_eda.py`) | ✅ Complete | Ariba |
| Preprocessing (`02_preprocessing.py`) | ✅ Complete | All three |
| Model experiments (`03_model_experiments.py`) | 🔄 In progress | *(assign)* |
| Tokenisation | 🔲 Planned | *(assign)* |
| Evaluation & tuning | 🔲 Planned | *(assign)* |
| Documentation | 🔄 Ongoing | Ariba |

---

## Important Rules

- **Never push directly to `main`** — always use a branch and pull request
- **Always pull before starting** — avoids merge conflicts
- **Do not commit the raw data CSV** — keep response data out of the repo for privacy
- **Keep `DATA_PATH` local** — do not commit your personal file path; each person sets their own
- **Commit often** — small commits are easier to review and undo
- **Write clear commit messages** — `"fix food matrix score for Jain + vegan pair"` not `"fix stuff"`

---

## Shared Constants Rule

`01_eda.py` and `02_preprocessing.py` share category orderings (e.g. `SLEEP_ORDER`, `FOOD_ORDER`, `VALID_DEALBREAKERS`). These are defined in both files and **must stay identical**. If you update a category list in one file, update it in the other immediately and note it in your commit message.

---

## Running the Pipeline

Scripts must be run in order. Each is fully independent (no imports between them), but preprocessing depends on the cleaned data that EDA validates first.

```bash
python Code/01_eda.py          # EDA — run first, check for unexpected category warnings
python Code/02_preprocessing.py  # Preprocessing — run after EDA passes cleanly
# 03_model_experiments.py — in progress
```
