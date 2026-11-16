# Activity 18 — Tool Reviews: AI at Work

**Diego Kamil Olea Ruiz** · Artificial Intelligence · Sessions 34–36

App: https://uam-aiclass-a18.streamlit.app/

---

## Part A — Content check (from the app)

**A1.** **Edge filter A** highlights the vertical edges. Its 9 numbers are:

```
-1  0  1
-2  0  2
-1  0  1
```

It subtracts the column on the left from the column on the right, so it reacts where brightness changes
from left to right, which is what a vertical edge is. Filter B is the same pattern turned on its side
(top row −1 −2 −1, bottom row 1 2 1), so it finds horizontal edges.

**A2.** For "Mérida is famous for its …", the word **"food"** has a probability of **55.0%** at
temperature **0.5** and **29.2%** at temperature **1.5**. Raising the temperature flattens the
probabilities: the top word loses share and unlikely words (like "traffic" or "snow") get picked more
often, so the output becomes more varied and less predictable.

**A3.** At step **t = 250** the image keeps **0.724** of the original (√ᾱₜ = 0.724). At step
**t = 500** it keeps only **0.280**. By the middle of the 1,000 steps the photo is mostly noise.

**A4.** The lecture says ChatGPT works "in much the same way as a Markov model". Both do predict the
next word, but a Markov model only looks at the last one or two words. A transformer uses attention to
look at the whole conversation at once, so it can keep track of something said many sentences earlier.
ChatGPT is also fine-tuned with human feedback (RLHF), which a Markov model has no equivalent for.

---

## Part B — Tool Review Cards

> **TO DO (Diego):** the four missions must be done with the real tools, using your school accounts,
> and the screenshots must come from your own screen. Fill in each card after doing the mission.
> Save screenshots in `screenshots/` (e.g. `M1_teachable.png`, `M2_gemini.png`, `M2_copilot.png`,
> `M3_notebook.png`, `M4_aistudio.png`).

### Mission 1: Image classifier with Google Teachable Machine

- **Tool(s) used and account type:** Google Teachable Machine (no account needed)
- **What I did** (2–4 sentences):
- **My exact prompts / classes:**
- **Evidence** (screenshots, counts, links):
- **Same vs. new conditions counts:**
- **Ratings** (1 = poor, 5 = excellent), with **one reason each**:
  - Usefulness for real work:
  - Accuracy / quality of the output:
  - Ease of use:
  - How much I would trust it without checking:
- **One error or limitation I found, and how I checked it:**
- **Free-plan limit I ran into (or "none"):**
- **Privacy:**
- **How I would use this tool in my field:**

### Mission 2: Two assistants, one work task (Gemini + Copilot Chat)

- **Tool(s) used and account type:** Gemini (school Google account), Copilot Chat (school Microsoft account)
- **What I did** (2–4 sentences):
- **My exact prompts** (same for both):
- **Evidence** (screenshots, counts, links):
- **Comparison of the two assistants:**
- **Fact I checked and its source:**
- **Ratings** (1 = poor, 5 = excellent), with **one reason each**:
  - Usefulness for real work:
  - Accuracy / quality of the output:
  - Ease of use:
  - How much I would trust it without checking:
- **One error or limitation I found, and how I checked it:**
- **Free-plan limit I ran into (or "none"):**
- **Privacy:**
- **How I would use this tool in my field:**

### Mission 3: Research with sources (Gemini Notebook)

- **Tool(s) used and account type:** Gemini Notebook (school Google account)
- **What I did** (2–4 sentences):
- **My exact prompts:**
- **Evidence** (screenshots, counts, links):
- **Citation I checked and what the original passage said:**
- **What happened with the question outside the document:**
- **Ratings** (1 = poor, 5 = excellent), with **one reason each**:
  - Usefulness for real work:
  - Accuracy / quality of the output:
  - Ease of use:
  - How much I would trust it without checking:
- **One error or limitation I found, and how I checked it:**
- **Free-plan limit I ran into (or "none"):**
- **Privacy:**
- **How I would use this tool in my field:**

### Mission 4: Build an app by describing it (Google AI Studio)

- **Tool(s) used and account type:** Google AI Studio (school Google account)
- **What I did** (2–4 sentences):
- **My exact prompts:**
- **Evidence** (screenshots, counts, links):
- **My 3 tests:**
- **The bug I found:**
- **My verdict on the lecture's "we're not there yet" quote:**
- **Ratings** (1 = poor, 5 = excellent), with **one reason each**:
  - Usefulness for real work:
  - Accuracy / quality of the output:
  - Ease of use:
  - How much I would trust it without checking:
- **One error or limitation I found, and how I checked it:**
- **Free-plan limit I ran into (or "none"):**
- **Privacy:**
- **How I would use this tool in my field:**

---

## Part C — Looking back

> **TO DO (Diego):** C1 and C2 depend on your ratings and evidence from Part B.

**C1. Comparison.**

**C2. Your career.**

**C3. AI-use disclosure.** I used Claude (Anthropic) to read the app's source code and compute the
Part A numbers, and to draft the wording of Part A and the empty card layout. The mission cards and
Part C are my own.
