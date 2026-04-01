# 📋 Survey Design — Final Form

Full documentation of the actual Google Form used for data collection, including all questions, answer options, and design rationale.
 
**Form Link:** [Live Form](https://docs.google.com/forms/d/e/1FAIpQLSc0KeZjZ_6HPUYbRpaoyT4PZrDWOAbVfOtBdgNBHtMk8FZZCQ/viewform)  


---

## Design Philosophy

> *"Answer based on what you ACTUALLY do, not what you think is ideal."*

Every question is designed to capture **real behaviour**, not socially desirable answers. The form avoids hypothetical preference questions and instead uses scenario-based and behavioural anchor questions throughout.

---

## Questions — Full List

---

### 🧍 Demographics

**Gender** *(optional)*
- Female / Male / Other / Do not wish to specify


---

### 😴 Sleep Schedule

**Q1. In the last month, on most days, you fell asleep around…**
- Before 10 PM
- 10 PM – 11:30 PM
- 11:30 PM – 1 AM
- 1 AM – 3 AM
- After 3 AM
- Varies wildly day to day


**Q2. You typically wake up (without being forced to)…**
- Before 6 AM
- 6 – 8 AM
- 8 – 10 AM
- After 10 AM

> "Without being forced to" captures natural rhythm, not alarm-driven behaviour.

**Q3. When you return to the room late (after 11 PM), you…**
- Turn on the main light and move normally
- Use phone torch, try to stay quiet
- Stay out until I know roommate is awake
- I almost never return after 11 PM

> Scenario-based. No option is framed as the "correct" answer.

---

### 🧹 Cleanliness

**Q4. How often do you sweep/tidy your side of the room?**
- Daily
- Every 2–3 days
- Weekly
- When it gets messy enough to bother me
- Rarely / Never

---

### 👥 Social Energy & Guests

**Q5. After a full day of classes, your ideal evening is…**
- Alone in the room — lights dim, silence
- Low-key — one person, quiet chat or a show
- Friends in the room, casual hangout
- Out with people, barely in the room

> Captures introvert/extrovert spectrum without labelling.

**Q6. When friends visit, they typically leave by…**
- 10 PM
- Midnight
- 2 AM or later
- I don't bring friends

**Q7. How much daily conversation do you want with a roommate?**
*(1 = barely acknowledge each other, 5 = talk regularly through the day)*
- Scale: 1 (Prefer near-silence) → 5 (Active conversation)

> Verbal anchors at both ends prevent defaulting to the middle.

---

### 📚 Study Habits & Noise

**Q8. When you study in the room, you need…**
- Complete silence — any noise breaks focus
- Quiet — soft background is okay
- I use headphones, so room noise doesn't matter
- I can study through most noise

**Q9. How much does music or audio from a roommate's device bother you (without headphones)?**
*(1 = no issue at all, 5 = immediately affects my mood or work)*
- Scale: 1 (Doesn't bother me) → 5 (Significant issue)

> Captured as an independent dimension from study habits — someone might use headphones but still be bothered by noise at other times.

---

### 🍱 Food, Smoking & Alcohol

**Q10. Your food preference is…**
- Strictly vegetarian
- Jain (no root vegetables)
- Vegetarian + eggs
- Non-vegetarian
- Vegan
- No preference

**Q11. How sensitive are you to non-veg food smells in a shared room?**
- Very — it's a dealbreaker
- I'd prefer it not happen
- Doesn't bother me
- I eat non-veg myself

> Smell sensitivity is a separate feature from food preference — a non-veg person may still be sensitive to strong smells from others.

**Q12. Do you smoke/drink, and where?**
- I don't smoke/drink at all
- I smoke/drink only outside / designated areas
- I sometimes smoke/drink in the room
- I smoke/drink regularly in shared spaces
- I smoke/drink in presence of peers

> "In presence of peers" captures social/situational behaviour — catches people who self-identify as non-smokers but participate in peer situations.

---

### ⚔️ Conflict Handling

**Q13. Scenario: A recurring habit of the roommate bothers you (e.g. not switching off lights, taking your stuff).**
- I stay quiet to keep the peace
- Dropped a hint but haven't addressed it directly
- Had one direct conversation about it
- Set a clear boundary and expect it to be respected
- Involved a third person or authority

> Options represent a clear avoidance → escalation spectrum. No "correct" answer.

---

### 🔒 Sharing & Boundaries

**Q14. You're okay with a roommate using your things (charger, stationery, snacks) without asking…**
- Never — always ask first
- Only small things like a pen
- Most daily-use items are fine
- Basically anything is fine

---

### 🌡️ Environment

**Q15. You prefer the room temperature to be…**
- Cool (Cooler/AC at 18–20°C)
- Comfortable (Cooler/Fan at 22–24°C)
- Warm (fan, no AC or AC at 26°C+)
- No strong preference

**Q16. Lights in the room when you're winding down at night — you prefer…**
- Complete darkness — lights off by 10 PM
- Dim lamp is fine, main light off
- Main light on is okay if someone's working
- I wear a sleep mask — doesn't matter

---

### 🚩 Dealbreakers

**Q17. Your most important dealbreakers in a roommate — pick up to 3 (ranked as Preference 1, 2, 3):**
- Sleep Schedule
- Smoking Habits
- Cleanliness
- Loud Noise / Music
- Lack of privacy
- Roommate avoids all conflict
- Frequent guests at night

> Top 3 ranked dealbreakers are used as **weighting flags** in the compatibility score. Mismatches on a respondent's flagged dimensions are penalised more heavily.

---

### 💬 Open-Ended (Optional)

**Q18. Enter any specific concerns or preferences not covered above.**

> Free text. Not used as a model feature — captured for qualitative review only.

---

## Summary

| Property | Value |
|---|---|
| Total questions | 17 required + 1 optional open-ended |
| Scale questions | 2 (Q7, Q9) |
| Scenario-based questions | 2 (Q3, Q13) |
| Multi-select/ranked | 1 (Q17 dealbreakers) |

---

## Anti-Bias Techniques

| Technique | Where Applied |
|---|---|
| *"Answer what you ACTUALLY do"* instruction | Top of form |
| Scenario-based questions | Q3 (late return), Q13 (conflict) |
| Verbal anchors on 1–5 scales | Q7 (conversation level), Q9 (noise sensitivity) |
| Peer-pressure behaviour capture | Q12 ("in presence of peers" option) |
| Smell sensitivity separate from food preference | Q10 + Q11 as independent features |
| No option framed as "correct" | All questions |
