# Part B: AI System & Metrological Certification

## 1. AI System Plan

**Input:** grayscale image (hexgrid with artifacts). **Output:** per circle, (x, y, r) plus how sure the model is about each (σ_x, σ_y, σ_r).

**Model:** U-Net-style encoder-decoder (shrinks the image to find patterns, then rebuilds it to full size so it can point at exact pixel locations). Three output branches: where circle centers likely are, how big each circle is, and how confident it is.

**Loss:** focal loss for finding centers (far fewer circle pixels than background), smooth L1 for radius, and a negative-log-likelihood term for confidence, `(y-ŷ)²/(2σ²) + log(σ)`, which punishes the model for being confident and wrong.

**Pre-processing:** normalize pixel values, add random flips/rotations so the model sees more variety during training.

**Post-processing:** the model outputs a probability score for every single pixel, because the number of circles per image is not fixed. This step turns that into an actual list of circles: find the local peaks, remove duplicate peaks next to each other, then read the radius and confidence at each surviving peak.

## 2. Certification Test Plan

**Datasets:** single-artifact sets (clean, blur, noise, illumination, missing-circle) to isolate each effect, plus a combined set.

**Metrics:**

- Precision/recall: match predicted circles to true circles within a small distance. Precision is the fraction of predicted circles that were real, recall is the fraction of real circles that got found. This matters because circles can go missing (your artifact) or be invented by a wrong model.
- RMSE of (x, y, r) on the correctly matched circles, for accuracy.
- Calibration check: for each matched circle, compute (prediction minus truth) divided by σ_predicted. Too wide implies the model is overconfident (real errors exceed what it claims); too narrow implies it is overly conservative (claims more uncertainty than needed).

Since this stands in for a digital PCR use case, it may also be worth checking how a detection or radius error shifts the final concentration estimate a real dPCR device would report, as that could be a more meaningful pass/fail signal than pixel error alone.

**Acceptance criteria:** pass/fail numbers are needed too (e.g. minimum precision/recall, maximum position error), but real values must come from the target application, not be guessed here.

**Stress test:** raise artifact strength gradually until performance breaks down, to find the model's failure point.

**Limits and practical challenges:** synthetic artifacts only approximate real camera noise, optics, and lighting, so real images are still needed before final sign-off. The calibration check also needs many test circles to be statistically meaningful.

## 3. Role of Part A's Data in Both Sub-Problems

**Training (sub-problem 1):** the data is what the model learns from, free and perfectly labeled, no manual work needed.

**Certifying (sub-problem 2):** the task assumes an unknown party hands you an already-built model. So here the same kind of data is used differently, as an independent judge with known-correct answers, to check any model regardless of how it was built or trained.
