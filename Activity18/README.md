# Activity 18: AI at Work — A Field Guide to Deep Learning and Generative AI
## Sessions 34–36
## Due date (mm/dd/yyyy): 11/15/2026
## Delivery Format: [] Video URL | [X] Markdown file | [] Jupyter Notebook file

---

# Activity Description

## The Story

Your first job will almost certainly involve AI tools: writing with an assistant, researching with
sources, prototyping an app, creating images, or training a model on your company's data. This
activity gives you real experience with several of them **before** you get there, and the knowledge
to use them well: what deep learning and generative AI actually are (updated for 2026), how people
build with them today, and how to use them responsibly.

Where the course text for Sessions 34–36 is outdated or wrong, the app says so and explains what is
accurate, with sources.

**No programming background is required**, and **every tool is free**. You never need to pay or enter
a credit card.

**App link:** https://uam-aiclass-a18.streamlit.app/

### The App

1. **🧠 Deep Learning (S34)**: layers, CNN filters (try them on a photo), and the shift from RNNs to transformers.
2. **✨ Generative AI (S35)**: LLMs, diffusion, GANs, and VAEs; try temperature and the diffusion noising process.
3. **🛠️ Building with GenAI (S36)**: how apps are really built with AI in 2026, and professional habits.
4. **🧪 Tool Lab**: four missions with real tools, plus a bonus.
5. **🛡️ Using AI Responsibly**: accounts, privacy, data settings, deepfakes, copyright, energy, and the law.

> 💡 **Hint:** Read the **🛡️ Using AI Responsibly** tab before starting the missions. Use your
> **school accounts** where a mission says so: they come with stronger privacy protection.

### Your Tasks

1. **Read the three content tabs** and try the three demos. Answer Part A of `A18_ToolReviews.md`.

   > 💡 **Hint:** Part A's answers come straight from the demos: change the filter, the temperature,
   > and the diffusion step, and read the numbers the app shows.

2. **Mission 1 (S34): train an image classifier** with Google Teachable Machine (no account needed).

   > 💡 **Hint:** Testing in *new* conditions is the point of this mission. A model that is perfect at
   > your desk and fails in the hallway is teaching you about training data.

3. **Mission 2 (S35): two assistants, one work task** with Gemini (school Google account) and Copilot
   Chat (school Microsoft account).

   > 💡 **Hint:** Give both assistants exactly the same memo and requests, so the comparison is fair.
   > Read their answers against the memo line by line: invented details are easy to miss.

4. **Mission 3 (S35): research with sources** in Gemini Notebook (school Google account).

   > 💡 **Hint:** Click a citation and read the original passage yourself. A correct-looking citation
   > is not the same as a correct answer.

5. **Mission 4 (S36): build an app by describing it** with Google AI Studio (or Bolt.new, or Lovable).

   > 💡 **Hint:** Keep the app small. Ask for one change at a time, and test after every change.

6. **Optional bonus: create a marketing image** with Adobe Firefly (or Canva) and check its Content
   Credentials.

7. **Fill out `A18_ToolReviews.md`** (one card per mission, plus Parts A and C) and submit it with your
   screenshots (a `.zip` with the `.md` file and the images is fine).

   > 💡 **Hint:** Before submitting, check each part against **How Your Grade Is Calculated** below.

If a tool's free plan has changed or a tool doesn't work for you, use another tool listed in the same
mission and say so in your card. That's a real-world skill too.

### How Your Grade Is Calculated

Your grade is out of **100 points**. The optional bonus can add up to **5 extra points** (the total is
capped at 100).

- **Part A (20 points): content check.** Questions A1–A3 have one correct answer, read from the app's
  demos (5 points each; partial credit if part is right). A4 is in your own words (5 points).
- **Part B (60 points): four Tool Review Cards, 15 points each.** Graded on four levels:
  - **Excellent (14–15):** every field filled in with *specific* details (your actual prompts, numbers,
    and screenshots); a real error or limitation found **and checked**; the mission's specific
    requirements done (see the note under the card template).
  - **Proficient (10–13):** complete, but some fields are generic or the error wasn't checked.
  - **Developing (5–9):** several fields missing or vague, or no evidence.
  - **Insufficient (0–4):** missing, or no sign the tool was actually used.
- **Part C (20 points):**

| Part | Points | Full points if you… | Partial credit |
|---|---:|---|---|
| C1. Comparison | 10 | compare at least three tools with reasons tied to your own ratings and evidence | Proficient 7–9 · Developing 3–6 · Insufficient 0–2 |
| C2. Your career | 6 | name two skills that become more valuable, each with a concrete example from your missions | Proficient 4–5 · Developing 2–3 · Insufficient 0–1 |
| C3. AI-use disclosure | 4 | give an honest, specific disclosure (which tool, for what) or a clear "none" | 2: vague · 0: missing |
| Bonus | +5 | complete the bonus card, including the Content Credentials result and the terms quote | 1–4: partly complete |

**Screenshots are evidence.** A card whose prompts, numbers, or screenshots don't match what the tool
would show gets no credit for those fields.

**Delivery format:** your answers must be in a Markdown file (`A18_ToolReviews.md`); a `.zip` with that
file and your screenshots is fine. Answers delivered in another format (PDF, Word, `.txt`) lose
**5 points** from the total.

### Running It Yourself (optional)

```bash
conda activate ai_uam
cd Activity18
pip install -r requirements.txt
streamlit run app.py
```

### About the Sources

Every fact in the app links to its source (original papers, official product pages, and law-firm
summaries for regulation). Tool information was checked on the tools' official pages on
October 9, 2026; free plans change often, so the app always shows that date. The photos are from the
scikit-image project's sample data (coffee: Rachel Michetti, CC0; chelsea: Stéfan van der Walt, CC0).

# References:
- [UNESCO (2023), Guidance for generative AI in education and research](https://unesdoc.unesco.org/ark:/48223/pf0000386693)
- [Content Credentials verify tool](https://contentcredentials.org/verify)
- [Markdown Guide](https://www.markdownguide.org/basic-syntax/)
