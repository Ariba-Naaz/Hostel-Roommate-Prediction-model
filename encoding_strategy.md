# ⚙️ Encoding Strategy — Per Feature

Based on the actual final form. Documents the planned encoding method for every question.

---

## Encoding Methods Reference

| Method | When Used |
|---|---|
| **Label Encoding** | Ordinal categories with a natural order |
| **One-Hot Encoding** | Nominal categories with no natural order |
| **Min-Max Scaling** | Numeric 1–5 scale responses |
| **Binary (0/1)** | Yes/No or two-option questions |
| **Multi-Hot Flags** | Multi-select / ranked questions |
| **Special Handling** | Non-standard options needing custom logic |

---

## Feature-Level Encoding

| Q | Feature | Encoding | Notes |
|---|---|---|---|
| — | Gender | Drop or One-Hot | Not a matching feature — demographic only |
| Q1 | Sleep time | Label Encoding + Special | Before 10 PM=1 … After 3 AM=5. "Varies wildly" = separate binary flag or median imputation |
| Q2 | Wake-up time | Label Encoding | Before 6 AM=1, 6–8 AM=2, 8–10 AM=3, After 10 AM=4 |
| Q3 | Late return behaviour | One-Hot Encoding | Nominal — no natural order between options |
| Q4 | Tidying frequency | Label Encoding | Daily=5 … Rarely/Never=1 (higher = cleaner) |
| Q5 | Ideal evening | One-Hot Encoding | Nominal — introvert/extrovert spectrum |
| Q6 | Guest departure time | Label Encoding + Special | 10 PM=1, Midnight=2, 2 AM+=3. "I don't bring friends" = separate binary flag |
| Q7 | Conversation level | Min-Max Scaling | 1–5 scale → normalise to 0–1 |
| Q8 | Study environment needs | One-Hot Encoding | Nominal — headphones option makes order unclear |
| Q9 | Audio/noise sensitivity | Min-Max Scaling | 1–5 scale → normalise to 0–1 |
| Q10 | Food preference | One-Hot Encoding | Nominal — veg/Jain/eggs/non-veg/vegan/no preference |
| Q11 | Non-veg smell sensitivity | Label Encoding | Very (dealbreaker)=4 … I eat non-veg=1 |
| Q12 | Smoking/drinking habits | Label Encoding | None=1, outside only=2, sometimes in room=3, regularly=4, peer pressure=2.5 (special) |
| Q13 | Conflict handling | Label Encoding | Quiet=1 … Third party=5 (avoidance → escalation) |
| Q14 | Item sharing comfort | Label Encoding | Never=1 … Anything fine=4 |
| Q15 | Temperature preference | One-Hot Encoding | Nominal — no universal order (cool vs warm is personal) |
| Q16 | Night lighting preference | One-Hot Encoding | Nominal — sleep mask option breaks ordinal order |
| Q17 | Dealbreaker flags | Multi-Hot + Weight Map | 3 ranked selections → binary flags per dimension + weight multiplier in scoring |
| Q18 | Open-ended text | Not encoded | Qualitative only — excluded from model |

---

## Special Cases

### Q1 — "Varies wildly day to day"
This option breaks the ordinal sleep time scale. Two approaches:
- **Option A:** Assign a median value (e.g. 3 = 11:30 PM–1 AM) and add a separate binary `sleep_irregular` flag
- **Option B:** Treat as its own category in One-Hot encoding, separate from the ordinal scale
> Decision pending — evaluate distribution of this response in actual data.

### Q6 — "I don't bring friends"
Breaks the ordinal guest departure time scale.
- Encode departure time ordinally (10 PM=1, Midnight=2, 2 AM+=3)
- Add a separate binary flag: `no_guests = 1`
> This preserves both pieces of information independently.

### Q12 — "In presence of peers"
Peer-pressure/situational behaviour — doesn't fit cleanly into the ordinal scale.
- Treat as a value between "outside only" and "sometimes in room" (e.g. 2.5) **or**
- Add a separate binary flag: `peer_pressure_smoker = 1`
> Separate flag approach is preferred — preserves the nuance.

---

## Dealbreaker Weighting (Q17)

Each respondent selects up to 3 dealbreaker dimensions ranked as Preference 1, 2, 3.

**Implementation:**
1. Create binary flags for each of the 8 dealbreaker dimensions per respondent
2. Assign weights: Preference 1 = highest weight, Preference 3 = lower weight (e.g. 3x, 2x, 1.5x)
3. When computing pairwise compatibility score, multiply the similarity contribution of each flagged dimension by the respondent's weight for that dimension
4. If **either** respondent in a pair has flagged a dimension, apply the higher of the two weights

**Dealbreaker dimensions and their corresponding features:**

| Dealbreaker | Maps to Feature(s) |
|---|---|
| Sleep Schedule | Q1, Q2, Q3 |
| Smoking Habits | Q12 |
| Cleanliness | Q4 |
| Loud Noise / Music | Q9 |
| Lack of privacy | Q14 |
| Roommate avoids all conflict | Q13 |
| Temperature | Q15 |
| Frequent guests at night | Q6 |

---

## Notes

- Apply Min-Max scaling **after** all label and one-hot encoding is complete
- One-Hot encoding will increase feature dimensionality significantly — review total column count after encoding
- Multi-Hot columns for dealbreakers are sparse by design (max 3 of 8 selected) — this is expected and fine
- Do not scale binary or one-hot columns — only scale ordinal label-encoded and 1–5 scale features
