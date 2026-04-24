# Final Project Roadmap — Motion Estimation in the Fourier Domain

## Legend

- 🔴 = not implemented
- 🔵 = planned but not necessary
- 🟡 = in construction
- 🟢 = done and tested

---

## Major Steps Overview

| Status | Step | Why it matters | Minimum deliverable |
|---|---|---|---|
| 🟢 | 1. Data pipeline and visualization | Need a stable base before doing any motion analysis | Load video, inspect frames, save clips/plots |
| 🟢 | 2. Simple synthetic motion dataset | Lets us debug theory on easy controlled cases | One moving blob / patch with known velocity |
| 🔴 | 3. 3D Fourier transform baseline | Core representation of the project | Compute 3D FFT of video volume |
| 🔴 | 4. Dominant plane detection | Main proposed motion cue | Recover one dominant velocity from simple motion |
| 🔴 | 5. Velocity-to-plane conversion validation | Ensures the geometry is implemented correctly | Show detected plane matches known synthetic motion |
| 🔴 | 6. Phase-based motion baseline | Comparison method from proposal | Implement a simple phase-based estimator |
| 🔴 | 7. Compare plane vs phase cues | Important experiment in proposal | Quantitative comparison on controlled videos |
| 🔴 | 8. Plane selection strategies | Proposal explicitly mentions NMS / CFAR / peak selection | Compare at least 2 selection methods |
| 🔴 | 9. Localized motion estimation | Needed for multiple independently moving objects | Windowed velocity estimation map |
| 🔴 | 10. Multi-object motion compensation | Core visual result | Compensate each region using local motion |
| 🔴 | 11. Fusion / all-in-focus moving-object output | Final demo result | Single output where multiple objects are sharp |
| 🔵 | 12. Optical flow baseline (Lucas-Kanade / RAFT / PWC-Net) | Nice comparison, but not essential for a strong final | Compare against one classical or deep baseline |
| 🔵 | 13. Harder real videos / sparse-inspired setting | Extension if time allows | Show robustness beyond toy examples |
| 🔴 | 14. Evaluation and report figures | Needed for final writeup and presentation | Metrics + visual comparisons |

---

# 1. Project Goal in One Sentence

Estimate motion in video using the **3D spatiotemporal Fourier domain**, compare it against **phase-based motion cues**, and use localized motion estimates to produce a fused output where **multiple moving objects appear sharp simultaneously**.

---

# 2. What Counts as a Successful Final Project?

## Minimum viable final project
If time gets tight, this is already a good final project:

- load conventional image sequences
- compute the 3D Fourier volume
- detect a dominant motion plane for simple translational motion
- recover a velocity estimate
- compare against a simple phase-based baseline
- show at least one motion compensation result on a simple sequence

## Strong final project
This is the target version:

- everything in the minimum project
- localized motion estimation with spatial/temporal windows
- multiple independently moving objects
- compensation per region
- fusion into a single sharp output video
- experiments comparing plane detection, phase cues, and combinations
- comparison of plane selection methods such as NMS and CFAR

## Stretch project
Only if the core is done:

- optical flow comparison
- harder real-world videos
- sparse or photon-limited inspired experiments
- hybrid method combining magnitude and phase cues

---

# 3. Recommended Build Order

## Stage A — Build the skeleton first
Do these first no matter what:

1. video loading
2. frame visualization
3. synthetic data generation
4. 3D FFT computation
5. simple plotting of Fourier slices / projections

### Output of Stage A
You should be able to say:

> “I can load a video, create a synthetic moving object sequence, compute the 3D Fourier transform, and visualize the structure.”

If you get only this far plus one detection experiment, you still have a sensible project foundation.

---

## Stage B — Prove the core idea on easy data
Now focus on the most important scientific question:

> Does translational motion create detectable plane structure that can recover velocity?

Tasks:

1. create a synthetic sequence with one object moving at constant velocity
2. compute its 3D Fourier transform
3. inspect the spectrum
4. implement a plane score / accumulator over candidate velocities
5. detect the best velocity
6. compare detected velocity with ground truth

### Deliverable
A figure showing:

- input frames
- Fourier magnitude visualization
- detected dominant velocity
- error relative to ground truth

This is the first real milestone.

---

## Stage C — Add the comparison baseline
The proposal explicitly says to compare dominant-plane detection with phase-based motion estimation.

Tasks:

1. implement a simple phase-based motion estimator
2. run both methods on the same synthetic sequences
3. compare robustness under:
   - different speeds
   - noise
   - blur
   - low texture
   - multiple objects later

### Deliverable
A table or plot with:

- method
- sequence type
- velocity error
- runtime
- failure cases

If needed, keep the phase-based baseline simple. It does not need to be perfect.

---

## Stage D — Compare plane selection strategies
The proposal mentions:

- non-maximum suppression
- CFAR-style detection
- related peak-selection methods

Tasks:

1. define a plane score over candidate velocities
2. implement a simple top-peak detector
3. implement NMS
4. implement CFAR-style detection if possible
5. compare which method best isolates true motion hypotheses

### Deliverable
A figure with velocity-space heatmaps and selected peaks for each strategy.

This is a strong experiment even if the rest is incomplete.

---

## Stage E — Localize motion
This is where the project becomes much more interesting.

Tasks:

1. split the video into spatial windows
2. optionally use temporal windows too
3. estimate velocity per local region
4. produce a map of local motion hypotheses
5. test on scenes with 2 independently moving objects

### Deliverable
A motion map over the image, or windowed detections per patch.

This is the bridge from “single global motion” to “real multi-object scenes.”

---

## Stage F — Compensation and fusion
This is the final visual result.

Tasks:

1. group regions by estimated motion
2. compensate each region according to its velocity
3. fuse compensated information across frames
4. generate a single output sequence or frame where multiple moving objects are sharp

### Deliverable
A before/after comparison:

- naive average or uncompensated result
- global compensation result
- localized compensation result
- final fused result

This is likely the best demo for the final presentation.

---

# 4. Concrete Incremental Milestones

## Milestone 0 — Repo setup
**Goal:** avoid chaos later.

Tasks:

- create folder structure
- define config file format
- decide naming convention for experiments
- make one script that runs end-to-end on a tiny example

Suggested structure:

```text
project/
├── data/
├── notebooks/
├── src/
│   ├── io.py
│   ├── synthetic.py
│   ├── fourier.py
│   ├── plane_detection.py
│   ├── phase_baseline.py
│   ├── localization.py
│   ├── compensation.py
│   └── metrics.py
├── experiments/
├── results/
└── README.md
```
---

## Milestone 1 — Video I/O and visualization

**Goal:** make sure the data side is easy.

**Tasks:**
- read a video as a tensor `(T, H, W)` or `(T, H, W, C)`
- convert to grayscale if needed
- crop / resize
- save sample frames
- save GIF or MP4 visualizations

**Success check:**
- you can load a video and inspect it quickly
- all later experiments can reuse this pipeline

---

## Milestone 2 — Synthetic data generator

**Goal:** debug on known motion.

**Create:**
- one moving square
- one moving Gaussian blob
- one moving textured patch
- optional: two objects with different velocities

**Parameters:**
- velocity
- size
- contrast
- noise
- number of frames

**Success check:**
- you know the ground-truth velocity exactly
- you can generate many clean tests cheaply

---

## Milestone 3 — 3D FFT baseline

**Goal:** compute the core signal representation.

**Tasks:**
- stack frames into a 3D volume
- compute the 3D FFT
- center frequencies for visualization if useful
- inspect slices / projections

**Visuals to make:**
- x-t slice of the video
- y-t slice of the video
- magnitude spectrum views
- maybe max projection of Fourier magnitude

**Success check:**
- simple translational motion produces structured geometry
- plots are interpretable

---

## Milestone 4 — Dominant plane scoring

**Goal:** turn Fourier geometry into velocity estimation.

**Tasks:**
- define candidate velocities
- map each velocity to a plane in frequency space
- accumulate energy near that plane
- score each velocity candidate
- return the best one

**Simplest first version:**
- brute force search over a grid of velocities
- energy = sum of Fourier magnitude near corresponding plane

**Success check:**
- single moving object with constant velocity gives the correct or near-correct velocity

---

## Milestone 5 — Validate on controlled cases

**Goal:** prove the implementation is not nonsense.

**Experiments:**
- vary velocity magnitude
- vary direction
- vary number of frames
- vary noise
- vary object texture

**Metrics:**
- endpoint velocity error
- angular error
- top-1 success rate

**Success check:**
- results are stable on simple sequences

---

## Milestone 6 — Phase-based baseline

**Goal:** implement the comparison method from the proposal.

**Possible simple choices:**
- phase correlation between adjacent frames
- local phase shift method
- a simplified phase-based motion cue in the Fourier domain

Do not overcomplicate this step.  
It is a baseline, not the entire project.

**Success check:**
- it runs on the same synthetic examples
- gives a reasonable velocity estimate for simple motion

---

## Milestone 7 — Compare Fourier plane vs phase

**Goal:** answer one of the main proposal questions.

**Experiments:**
- clean synthetic motion
- noisy motion
- low texture
- fast motion
- motion blur
- short sequences

**Questions to answer:**
- which method is more stable?
- which fails first?
- does combining them help?

**Success check:**
- you can make a plot or table showing strengths and weaknesses

---

## Milestone 8 — Plane selection strategies

**Goal:** compare how to choose dominant planes.

**Implement at least:**
- simple top-peak selection
- NMS
- CFAR-style detection if possible

**Experiment idea:**
- create sequences with two motions
- compare whether each strategy recovers both peaks or merges them badly

**Success check:**
- you can justify why one selection strategy is better

---

## Milestone 9 — Localized motion estimation

**Goal:** separate multiple moving objects.

**Tasks:**
- divide the video into overlapping spatial windows
- run velocity estimation in each window
- optionally smooth or regularize outputs
- visualize local motion map

**Experiments:**
- two objects moving differently
- object + static background
- crossing motions if possible

**Success check:**
- different image regions produce different motion hypotheses

---

## Milestone 10 — Motion compensation

**Goal:** use estimated motion, not just measure it.

**Tasks:**
- warp or shift frames according to estimated velocity
- compensate globally first
- then compensate locally by region / window

**Compare:**
- original
- naive averaging
- global compensation
- local compensation

**Success check:**
- compensated object appears visibly sharper

---

## Milestone 11 — Fusion output

**Goal:** produce the final signature result.

**Tasks:**
- compensate each region according to its motion
- fuse contributions into one output
- choose blending strategy
- avoid obvious seams if possible

**Success check:**
- multiple moving objects look simultaneously sharper in a final output

This is the “demo” result that people will remember.

---

# 5. Suggested Experiments in Priority Order

## Priority 1 — Must do

These are the most important.

### Experiment 1: One object, one velocity, synthetic
- objective: verify basic plane detection
- output: velocity error and Fourier visualization

### Experiment 2: One object, compare plane vs phase
- objective: test the central comparison in the proposal
- output: table of errors and examples

### Experiment 3: Two objects, global failure case
- objective: show why one global motion is insufficient
- output: failure of a single compensation

### Experiment 4: Localized motion estimation
- objective: separate multiple motions
- output: local motion map

### Experiment 5: Final compensation/fusion result
- objective: show project payoff
- output: sharp multi-object output

---

## Priority 2 — Very good if time allows

### Experiment 6: NMS vs CFAR vs naive peak selection

### Experiment 7: Magnitude-only vs phase-only vs combined cue

### Experiment 8: Robustness to noise / blur / short clips

### Experiment 9: Real video instead of only synthetic

---

## Priority 3 — Nice but optional

### Experiment 10: Optical flow comparison

**Possible baselines:**
- Lucas-Kanade
- Farneback
- RAFT or PWC-Net if easy to run

### Experiment 11: Sparse-inspired degradation

**For example:**
- randomly remove pixels
- binarize or threshold strongly
- reduce temporal support

This connects the project back to harder sensing settings without needing a full SPAD pipeline.

---

# 6. Recommended Figures for the Final Report

You will probably want these:

1. pipeline overview diagram
2. synthetic data examples
3. Fourier magnitude visualization for simple motion
4. velocity-space score map
5. comparison of plane vs phase methods
6. comparison of peak selection methods
7. localized motion map for multi-object scene
8. before/after compensation and fusion results



