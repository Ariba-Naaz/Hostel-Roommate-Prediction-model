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

## 🔲 Phase 4 — K-Means Clustering (Pre-Processing Step)
 
- [ ] Apply K-Means clustering on the full encoded feature matrix
- [ ] Experiment with different values of K — evaluate cluster quality (elbow method / silhouette score)
- [ ] Assign each student token to a cluster
- [ ] Inspect clusters manually — do groupings make intuitive sense? (e.g. night owls together, clean + quiet together)
- [ ] Finalise K and cluster assignments before moving to matching
 
> **Why cluster first?** Matching every student against every other student is computationally expensive and produces poor matches across very different profiles. K-Means groups similar students first — matching then happens **only within each cluster**, producing better and faster results.

---

## 🔲 Phase 5 — Cosine Similarity Within Clusters
 
- [ ] For each cluster, compute pairwise cosine similarity between all student vectors inside that cluster
- [ ] Apply **dealbreaker weighting** — for each student pair, increase the weight of dimensions flagged as dealbreakers by either student
  - Preference 1 dealbreaker → highest weight multiplier (e.g. 3×)
  - Preference 2 → medium weight (e.g. 2×)
  - Preference 3 → lower weight (e.g. 1.5×)
  - If either student in a pair has flagged a dimension, apply the higher of the two weights
- [ ] Also assugn weughts to each question according to their importance. eg. smoking habits should have more weight over music preference 
- [ ] Compute final weighted compatibility score for each pair
- [ ] Generate ranked list — **top 3 most compatible matches** per student token
- [ ] Evaluate output quality manually — do top 3 matches make intuitive sense?
---

## 🔲 Phase 6 — Evaluation & Refinement
 
- [ ] Review match quality across different clusters
- [ ] Check edge cases — what happens when a cluster has very few students?
- [ ] Tune dealbreaker weight multipliers if output quality is poor
- [ ] Document final chosen K value and weight multipliers with justification
- [ ] Implement several methods to test the compatibility of prediction models used

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

