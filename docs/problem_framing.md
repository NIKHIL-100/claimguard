# Step 1: Problem framing

Write these decisions down before any code. Interviewers usually ask about them first.

## 1. What is the business problem?

Generative AI makes it cheap to fake claim evidence: a fully generated photo of a "damaged"
car, or a real photo with a dent or crack painted in (inpainting). Insurers such as PZU
(Poland), Mitsui Sumitomo and Aioi Nissay Dowa (Japan) and Admiral (UK) report this happening
and have started deploying detectors. Manual checking by adjusters does not scale.

## 2. What is the ML task?

**Binary image classification.** Input: one photo. Output: probability it is AI-made.

| Term | Our definition |
|---|---|
| Positive class (1) | **Fake** — the thing we are trying to catch |
| Negative class (0) | Real camera photo |
| Prediction | p(fake) between 0 and 1 |
| Decision rule | flag if p(fake) ≥ threshold t |

Why "fake = positive"? Precision, recall, false positives and false negatives are all defined
relative to the positive class, so it must be the class of interest.

## 3. The two errors and what they cost

| | Predicted fake | Predicted real |
|---|---|---|
| **Actually fake** | True positive — fraud caught | **False negative** — fraud paid out |
| **Actually real** | **False positive** — honest customer delayed | True negative — honest claim passes |

- A **false negative** is expensive per case (a fraudulent payout).
- A **false positive** is cheaper per case (an adjuster's review time and an annoyed customer),
  but they add up: if the tool flags too many honest claims, adjusters stop trusting it.

## 4. How the model will be used

As **triage**: anything above the threshold goes to a human adjuster. The model never denies
a claim by itself. This choice lowers the cost of a false positive and is the ethical
default, because detectors make mistakes on unseen generators.

## 5. Success criteria

| Criterion | Target | Why this one |
|---|---|---|
| ROC-AUC (headline) | as high as possible; compare models | Threshold-free ranking quality; the NTIRE 2026 challenge metric |
| Recall @ 5% FPR | ≥ 90% | "If we accept 1 false alarm per 20 honest claims, how much fraud do we catch?" |
| ROC-AUC on held-out generator (FLUX) | report honestly | Shows whether the model generalises to new generators |
| ROC-AUC drop under JPEG/resize/blur/crop | < 10 points | Claim photos arrive compressed via email or messaging apps |

## 6. Why not accuracy?

With 10% fakes, a model that says "real" for every photo has 90% accuracy and catches nothing.
Accuracy is reported, but never used to choose between models.

## 7. Scope

In scope: photos of vehicle and property damage; fully generated and inpainted fakes.
Out of scope (future work): documents and receipts, video, detecting photos copied from the
internet, provenance metadata such as C2PA.
