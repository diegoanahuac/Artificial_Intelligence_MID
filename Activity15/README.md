# Activity 15: Digit Lab — Teaching a Neural Network to Read Handwriting
## Sessions 25–27
## Due date (mm/dd/yyyy): 10/25/2026
## Delivery Format: [] Video URL | [X] Markdown file | [] Jupyter Notebook file

---

# Activity Description

## The Story

Before you deposit a check from your phone, a computer reads the handwritten amount. Before a
letter reaches your door, a machine reads the handwritten postal code. Both jobs are done by
**neural networks** — programs loosely inspired by the neurons in your brain, which *learn* to
recognize patterns from thousands of examples instead of being given rules.

In this activity you get your own neural network that reads handwritten digits (0–9). You will
look inside it (**Session 25**: how a network represents an image), watch it learn one step at a
time (**Session 26**: back-propagation), train your own versions by changing the settings, and
finally test it on your own handwriting (**Session 27**: neural networks applied to a real problem).

**No programming background is required.** Everything happens by clicking, moving sliders, and
drawing in the app.

**App link:** https://uam-aiclass-a15.streamlit.app/

If you'd rather run it on your own machine instead of using the shared link, see
**Running It Yourself** below.

### The App

Four tabs:

1. **🧠 Meet the Network** — how a 28 × 28 pixel image becomes a list of numbers, the **canonical
   network** (a fixed network that everyone gets exactly the same, used for the graded answers),
   its training curves, its confusion matrix, and pictures of what each hidden neuron learned.
2. **🔁 Back-propagation Step by Step** — a tiny 2-2-1 network where you click through one full
   training step: forward pass, error, deltas, gradients, and the weight update, with every number
   shown.
3. **🎛️ Training Lab** — choose your own hyperparameters (hidden layers, neurons, activation
   function, learning rate, epochs, batch size, initial weights), press **Train**, and compare your
   network with the canonical one.
4. **✏️ Draw & Classify** — classify any of the 1,000 test images, draw your own digit for the
   network to read, and browse the test images the canonical network gets wrong.

### Your Tasks

1. **In the Meet the Network tab**, study the canonical network: its layers, its number of
   trainable parameters, its train/test accuracy and loss, and its training curves. Take a
   screenshot of the diagram and the five numbers below it.

   > 💡 **Hint:** The diagram tells you how many neurons each layer has. Ask yourself *why* the input
   > layer has exactly that many neurons. The "What the network sees" grid above it is the clue.

2. **Scroll down to "What did the hidden neurons learn?"** Pick one neuron whose picture looks
   like it has a clear shape and take a screenshot of it.

   > 💡 **Hint:** Red areas are where ink makes the neuron fire; blue areas are where ink makes it
   > stay quiet. Squint: does a neuron look like part of a stroke, a loop, or an edge?

3. **In the Back-propagation tab, leave the learning rate at 0.50** and click **Next step ▶**
   through all 5 steps. Take a screenshot of step 2 (the error) and of step 5 (the weight table and
   the "Error after one update" number).

   > 💡 **Hint:** Read each step's sentence before clicking on. Step 3 starts the *backward* pass:
   > notice which direction the information is now flowing, compared with step 1.

4. **In the Training Lab tab, train at least 4 networks** and keep the run-history table:
   - one with **all settings at their defaults** (you should get exactly the canonical network),
   - one with a learning rate that is **too low**,
   - one with a learning rate that is **too high**,
   - one with **"all zeros"** initial weights.

   Then experiment freely to find the best test accuracy you can. Take a screenshot of your run
   history table.

   > 💡 **Hint:** Change **one** setting at a time, so you know what caused the difference. A
   > learning rate is "too low" when the network is still improving slowly at the last epoch, and
   > "too high" when the loss jumps around or the accuracy collapses. Training takes a few seconds,
   > and the same settings always give the same result.

5. **In the Draw & Classify tab**, look up **test images #24, #4, and #6** with the
   "Test image number" box. For each one, write down the true label, the canonical network's
   prediction, and its confidence. Take a screenshot of image #6.

   > 💡 **Hint:** Below the canvas, the "Where the canonical network gets it wrong" gallery shows
   > every misclassified test image. Its first sentence tells you how many there are.

6. **Draw at least 3 digits of your own** on the canvas and note what the network answers each
   time. Take a screenshot of at least one attempt, including the small "What the network sees"
   image.

   > 💡 **Hint:** Draw large and centered, then try a deliberately messy, tiny, or off-center digit.
   > Compare the 28 × 28 version of your drawing with the MNIST images in the first tab.

7. **Fill out `A15_ReflectionQuestions.md`**, using the exact numbers from your screenshots, and
   submit it along with your labeled screenshots (a `.zip` with the `.md` file and the images is
   fine).

   > 💡 **Hint:** Before submitting, check each answer against **How Your Grade Is Calculated**
   > below. It lists exactly what earns full points on every question.

### How Your Grade Is Calculated

Your grade is out of **100 points**, split across the 12 reflection questions. There are three
kinds of questions, and each kind is graded differently:

- **Exact answers** (Q1, Q2, Q3, Q5, Q10): there is one correct value, read from the app. With the
  default settings, everyone sees exactly the same numbers. You get full points for correct values and
  partial credit when some parts are right. Small rounding differences are accepted.
- **Your own experiments** (Q7, Q8, Q9): every student's runs are different, so there is no single
  right number. But the app always gives exactly the same result for the same settings, so any run
  you report can be re-run and checked. You are graded on whether your numbers match your settings
  and on your reasoning. Numbers that no setting can reproduce get no credit.
- **Explain in your own words** (Q4, Q6, Q11, Q12): graded on four levels.
  - **Excellent:** complete, correct, and specific to your own results.
  - **Proficient:** correct, but missing a required part or not tied to your own results.
  - **Developing:** vague, incomplete, or partly wrong.
  - **Insufficient:** missing or wrong.

| # | Question | Points | Full points if you… | Partial credit |
|---|---|---:|---|---|
| 1 | Layer sizes | 6 | give all three layer sizes **and** explain why the input and output layers have exactly that many neurons | 3–4: sizes right but explanation missing or wrong, or one size wrong |
| 2 | Trainable parameters | 6 | give the correct number **and** show how it's computed (weights + biases for both pairs of layers) | 3–4: right number with no breakdown, or a breakdown that misses the biases |
| 3 | Canonical accuracy and loss | 8 | report all 4 values correctly | 3–6: 2 or 3 of the 4 values correct |
| 4 | One hidden neuron | 6 | include its screenshot, describe its red/blue pattern, and explain where that pattern came from | Proficient 4–5 · Developing 2–3 · Insufficient 0–1 |
| 5 | Back-propagation numbers | 10 | report all 4 values correctly at learning rate 0.50 | 4–7: 2 or 3 of the 4 correct, or all 4 correct for a different learning rate that you state |
| 6 | The backward pass | 10 | explain how the error at the output (δo) is sent backward to give each hidden neuron its share (δh1, δh2), and how each weight's gradient decides how it changes, using the deltas you saw | Proficient 6–8 · Developing 3–5 · Insufficient 0–2 |
| 7 | Run-history table | 12 | include at least the 4 required runs (your defaults run matches the canonical network), with numbers that match their settings, and describe and explain the too-low and too-high curves | Proficient 7–10 · Developing 3–6 · Insufficient 0–2 |
| 8 | All-zeros initial weights | 8 | report a result that matches the app and explain why identical starting weights stop the network from learning | Proficient 5–6 · Developing 2–4 · Insufficient 0–1 |
| 9 | Best configuration | 10 | report a best run that can be reproduced, and answer "does bigger or longer always help?" using the train − test gap from your own runs | Proficient 6–8 · Developing 3–5 · Insufficient 0–2 |
| 10 | Test images and mistakes | 10 | give the true label, prediction, and confidence for all three images, plus the number of misclassified test images | 4–7: most values right, but one image or the count wrong |
| 11 | Your own drawings | 6 | report at least 3 drawings with the network's answers (with a screenshot) and give a real reason they can be harder than the test images | Proficient 4–5 · Developing 2–3 · Insufficient 0–1 |
| 12 | Real-world application | 8 | describe a Session 27 application with specific inputs, specific outputs, and a realistic consequence of a wrong prediction | Proficient 5–6 · Developing 2–4 · Insufficient 0–1 |
| | **Total** | **100** | | |

**Delivery format:** your answers must be in a Markdown file (`A15_ReflectionQuestions.md`); a `.zip`
with that file and your screenshots is fine. Answers delivered in another format (PDF, Word, `.txt`)
lose **5 points** from the total.

### Running It Yourself (optional)

```bash
conda activate ai_uam
cd Activity15
pip install -r requirements.txt
streamlit run app.py
```

### About the Data

The handwriting comes from **MNIST** (Yann LeCun, Corinna Cortes, and Christopher J.C. Burges), the
classic benchmark of 70,000 handwritten digits originally collected from U.S. Census Bureau employees
and high-school students. The app uses a fixed subset of 6,000 training images and 1,000 test
images (exactly 600 and 100 of each digit) so that training takes seconds.

# References:
- [Streamlit documentation](https://docs.streamlit.io/)
- [3Blue1Brown — But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
- [Markdown Guide](https://www.markdownguide.org/basic-syntax/)
