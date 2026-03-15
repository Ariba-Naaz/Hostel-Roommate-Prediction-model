# 📚 Research & References

---

## 🤖 Machine Learning Concepts

### Cosine Similarity
Measures similarity between two vectors based on the angle between them — not magnitude. Ideal for comparing student behavioural profiles as feature vectors. Requires no training data.

### K-Nearest Neighbours (KNN)
Finds the K most similar respondents based on feature distance. Naturally produces a ranked output. Sensitive to feature scaling — Min-Max normalisation is essential before applying KNN.

### K-Means / Hierarchical Clustering
Groups similar profiles into clusters without supervision. May be used as a pre-processing step before pairwise matching — match students within the same cluster to reduce computational load and improve match quality.

### Random Forest
Ensemble method that can learn complex feature interactions. Requires labelled training data (known compatible/incompatible pairs). Not applicable at current stage — noted as a future direction once feedback data is available.

---

## 📐 Feature Engineering

### Encoding Methods Used
| Method | Use Case |
|---|---|
| Label Encoding | Ordinal features with natural order |
| One-Hot Encoding | Nominal features with no order |
| Min-Max Scaling | Numeric 1–5 scale responses |
| Binary (0/1) | Yes/No questions |
| Multi-Hot Flags | Multi-select questions |

### Dealbreaker Weighting
Respondents select up to 2 dealbreaker dimensions. These dimensions receive increased weight in the compatibility score computation. This ensures the most important factors for each individual have higher influence on their match results.

---

## 🏠 Existing Roommate Matching Systems

### College Housing Platforms (US)
Several US university housing systems use compatibility-based matching:
- Survey-based systems asking about sleep schedules, cleanliness, social habits
- Some use simple preference matching; more advanced ones use weighted scoring

### Open Source / Research Projects
- Academic research on peer compatibility in shared living environments
- Studies on the effect of roommate compatibility on academic performance and mental health

---

## 📖 Academic References

- Behavioural survey design and anti-bias techniques in self-report instruments
- Anthropometric and behavioural compatibility in shared living research
- Feature engineering for categorical and ordinal survey data in ML pipelines
- Unsupervised matching and recommendation systems literature

---

## 🔗 Tools & Libraries

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical operations |
| Scikit-learn | Encoding, scaling, KNN, clustering |
| Jupyter Notebook | Exploratory analysis and model experiments |
| Google Forms | Survey data collection |

---

## 🔭 Research Gaps (To Be Addressed)

- Deeper understanding of Random Forest for unsupervised/semi-supervised matching
- Effect of feature weighting on KNN distance metrics
- Whether clustering before pairwise matching improves computational efficiency at scale
- Benchmarking against existing roommate matching systems
