# 🧩 Challenges, Mistakes & Solutions

A detailed log of significant problems encountered, design mistakes made, and how they were resolved.

---

## Challenge 1 — Self-Reporting Bias in Form Design

**Phase:** Survey Design  
**Status:** ✅ Resolved

### Problem
The original form contained questions phrased as preference or self-assessment questions. This created a significant bias problem: respondents were answering based on who they want to be perceived as, not how they actually behave.

> Example: *"Would you be quiet when coming in late?"* → Almost universally gets "Yes." This is socially desirable, not behavioural reality.

This would have produced a dataset full of idealised responses — useless for predicting real compatibility.

### Solution
The entire form was redesigned to use **scenario-based and behavioural anchor questions**. Questions were reworded from *"Would you..."* and *"Do you prefer..."* to *"When X happens, you..."* with realistic options that don't have an obvious "correct" answer.

The form instruction now reads: *"Answer based on what you actually do, not what you think is ideal."* — placed at the top of the form.

### Lesson Learned
For behavioural data collection, hypothetical preference questions are unreliable. Scenario-based questions with realistic options are significantly more effective at capturing actual behaviour.

---

## Challenge 2 — Smoking Questions Were Insufficient

**Phase:** Survey Design  
**Status:** ✅ Resolved

### Problem
The original form had a single smoking question. This was insufficient for two reasons:
- Students who smoke occasionally in social situations self-identify as non-smokers — a single question misses them entirely
- A roommate who smokes outside but carries the smell indoors on their clothes causes friction even if they technically "don't smoke in the room" — a single question doesn't capture this

### Solution
Added two independent questions:
1. A question capturing peer-pressure/situational smoking — catches students who self-label as non-smokers but smoke in certain situations
2. A separate question about reaction to smoke smell on a roommate's clothes/breath — captures residual smell sensitivity as an independent feature

These are now two separate features in the dataset, not one.

### Lesson Learned
A single question is often not sufficient for complex, nuanced behaviours. Think about edge cases and indirect manifestations of the behaviour you're trying to capture.

---

## Challenge 3 — Jain Food Question Was Biasing Other Respondents

**Phase:** Survey Design  
**Status:** ✅ Resolved

### Problem
The initial food preference question did not adequately capture the spectrum of Jain dietary practices. 

### Solution
Added a **conditional follow-up question**, asking specifically about strictness of food practices in shared spaces. This prevents the question from being shown to or biasing non-Jain respondents, while still capturing the nuanced data needed for accurate matching.


---

## Challenge 4 — Form Version Control Gap

**Phase:** Survey Design → Refinement  
**Status:** ⚠️ Acknowledged — Fix in Progress

### Problem
Several fields that were initially included in the form were removed during refinement for reasons of redundancy, ambiguity, or potential to re-identify respondents. However, these changes were **not documented at the time** — there is no record of what was removed and why.

This is a gap in project record-keeping that makes it difficult to understand the full evolution of the form design.

### Solution (Going Forward)
All future form changes will be version-controlled with a change log documenting:
- What was added/removed
- Why the change was made
- What version of the form it applies to

### Lesson Learned
Treat form design like code — version control it. Changes made without documentation create confusion later, especially in a team project.

---

## Challenge 5 — Model Selection is Non-Trivial Without Labelled Data

**Phase:** Model Planning  
**Status:** 🔄 In Progress — Pending Data Collection

### Problem
Selecting the right model is difficult without knowing the final dataset size and distribution. Additionally:
- Random Forest and other supervised models require labelled training data (known compatible/incompatible pairs) — we don't have this
- The team lacks sufficient research depth on ensemble methods for this type of matching problem at this stage

### Current Approach
- Explore **clustering** once initial responses are in to see if natural groupings exis.
- Use **Cosine Similarity** as a baseline — requires no training data, interpretable output
- Layer **KNN** for ranked matching and compare
- Explore **clustering** once initial responses are in to see if natural groupings exist

### Lesson Learned
Don't over-engineer model selection before data is available. A simple, interpretable baseline is more valuable than a complex model that can't be trained yet.

---

## Ongoing / Open Issues

| Issue | Status |
|---|---|
| Tokenisation not yet implemented | 🔲 Planned for preprocessing phase |
| Multi-select columns may produce sparse binary features | 🔲 To be evaluated after encoding |
| Small dataset in early collection — may not represent full student profile range | 🔄 Monitoring |
| Form version history not fully reconstructable | ⚠️ Acknowledged gap |
