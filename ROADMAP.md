# 🗺️ Project Roadmap — Hostel Roommate Compatibility Model

---

## ✅ Phase 1 — Ideation & Survey Design *(Complete)*

- [x] Define problem statement and project goal
- [x] Identify 7 behavioural dimensions to capture
- [x] Design Google Form with anti-bias techniques
- [x] Implement scenario-based questions (not preference-based)
- [x] Add dual smoking capture and residual smell sensitivity question
- [x] Add conditional Jain food follow-up question
- [x] Finalise dealbreaker flags for model weighting
- [x] Document encoding strategy per feature

---

## 🔄 Phase 2 — Data Collection *(In Progress)*

- [ ] Distribute Google Form to hostel students
- [ ] Collect minimum viable dataset (target: 50+ responses)
- [ ] Monitor response distribution across dimensions
- [ ] Address any ambiguous or inconsistent responses

---

## 🔲 Phase 3 — Preprocessing & Feature Engineering

- [ ] Implement tokenisation — assign random tokens (e.g. RT-4829-XK) to each submission
- [ ] Store token-to-response mapping separately and securely
- [ ] Data cleaning — handle missing values, validate response distributions
- [ ] Apply Label Encoding to ordinal features
- [ ] Apply One-Hot Encoding to nominal features
- [ ] Apply Min-Max Scaling to 1–5 scale responses
- [ ] Apply Binary encoding to Yes/No questions
- [ ] Apply Multi-Hot encoding to multi-select fields
- [ ] Validate final feature matrix

---

## 🔲 Phase 4 — Baseline Model (Cosine Similarity)

- [ ] Build pairwise cosine similarity computation across all student vectors
- [ ] Implement dealbreaker weighting — increase weight for flagged dimensions
- [ ] Generate ranked compatibility list (top 3 matches) per token
- [ ] Evaluate output quality manually — do matches make intuitive sense?

---

## 🔲 Phase 5 — KNN Comparison

- [ ] Implement KNN-based matching
- [ ] Compare KNN ranked output with cosine similarity output
- [ ] Tune K and distance metric
- [ ] Document differences and choose best approach

---

## 🔲 Phase 6 — Clustering Exploration

- [ ] Apply K-Means or Hierarchical Clustering on feature matrix
- [ ] Evaluate if natural groupings exist in the data
- [ ] Test: does matching within clusters improve output quality?
- [ ] Decision: use clustering as pre-processing step or skip

---

## 🔲 Phase 7 — Final Model & Output

- [ ] Finalise model pipeline
- [ ] Output: top 3 ranked compatible matches per student token
- [ ] Add confidence score per match (how strongly compatible)
- [ ] Test with full dataset

---

## 🔭 Future (Post-Project)

- **Admin Interface** — simple UI for hostel staff to upload form responses and receive ranked match outputs
- **Feedback Loop** — validate model output against real-world roommate satisfaction after assignment
- **Version 2** — optional preference fields (hobbies, academic stream) as secondary tie-breakers
- **Ensemble Research** — deeper study of Random Forest for semi-supervised matching once labelled pairs are available

---

*Last updated: March 2026*
