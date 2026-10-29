# Stage Log

A concise log of the 21 stages of this project, one per experiment or study step. All numbers come from my own runs and are copied from the per-stage notes. "Public score" and "private score" are real Kaggle leaderboard submissions. "Holdout" means a local 12-study validation split, which turned out to be very noisy (see Stages 12, 13, 19 and 20), so holdout numbers are good for comparing ideas, not for estimating real performance.

### Stage 1 - Exploratory data analysis
- **Goal:** Understand the dataset structure and where the labels come from.
- **What I did:** Inspected the CSVs, the DICOM folder layout, and class balance. Found that the dataset mounts under an extra `competitions/` path segment, which a silent `os.walk` had hidden.
- **Result:** Only 58 of 4407 studies (1.3%) have direct numeric labels. The other 4349 have only report text. 24,371 series in total (about 5.5 per study).
- **Takeaway:** The "gold" set is tiny, which shaped every later decision about validation and label sources.

### Stage 2 - Report label extraction
- **Goal:** Turn free-text radiology reports into labels for the 12 findings.
- **What I did:** Built regex extractors (v1-v5) with directional negation and per-class vocabulary, then switched to a small local LLM (3B parameters) covering all 12 classes (v6). Validated against the 58 gold studies.
- **Result:** Precision/recall on gold (not AUC): Fracture regex 0.88/0.39, LLM ACL 0.94/0.67, Effusion 0.73/0.86, Fracture 0.60/0.17. Six of 12 classes had LLM recall below 0.25.
- **Takeaway:** The LLM beat regex on most classes but under-detects findings. Some "errors" are genuine annotator disagreement.

### Stage 3 - DICOM geometry
- **Goal:** Recover the true slice order and knee laterality from DICOM geometry.
- **What I did:** Computed slice position from orientation vectors (cross product, then dot product with position) and inferred left/right from the image center.
- **Result:** File-name order did not match physical order (Spearman rho 0.13, p=0.55, one 22-slice series). Laterality matched the convention in 12 of 14 sampled series that had a tag (86%).
- **Takeaway:** File names carry no anatomical order. Geometry must be used. Sample sizes here were small.

### Stage 3b - Slot selection and physical scale
- **Goal:** Pick one series per slot (plane x fat-suppression) and set a physically meaningful resize.
- **What I did:** Implemented `pick_slots` (up to 6 slots per study, tie-break by slice count) and a fixed physical crop with a Nyquist check.
- **Result:** Effective resolution of 0.580 mm/px at 224 px and 0.387 mm/px at 336 px, matching the reference notebook. Sampled studies filled 5-6 slots.
- **Takeaway:** 224 px falls below the Nyquist limit for a 1 mm tear. 336 px is adequate.

### Stage 4 - Multiple-instance learning (MIL) model
- **Goal:** Build a model that treats a study as a bag of slots with one attention query per class.
- **What I did:** Implemented single-query then 12-query attention pooling with masking for missing slots (automated tests), then plugged in DINOv2 (frozen, then last block unfrozen) on real data.
- **Result:** Tests passed (attention sums to 1, masked slots get about 0, gradients flow). Fine-tuning loss on the 58 studies went from 0.7168 (epoch 1) to 0.6096 (epoch 10). No validation split yet.
- **Takeaway:** The mechanism works end to end. This stage verified plumbing, not accuracy.

### Stage 5 - Leak-free validation (5a)
- **Goal:** Validate without fooling myself.
- **What I did:** Checked for duplicate reports, split 46 train / 12 holdout grouped by report hash, computed `pos_weight` on train only, and trained 20 epochs tracking holdout AUC.
- **Result:** Best holdout mean AUC 0.5967 at epoch 10, then it drifted down to about 0.57-0.58 while training loss kept falling. Multimodal fusion with report text (5b) was paused and never resumed.
- **Takeaway:** With 46 training studies the model memorizes quickly, so select the best epoch rather than the last one.

### Stage 6 - Not run
- **Goal:** Originally reserved for advanced specialists (PCA, Synovitis specialist) from the reference notebook.
- **What I did:** Nothing. The number was left unused in the roadmap. My notes show it was never started. The ensembling work became Stage 8.
- **Result:** None.
- **Takeaway:** Plans changed once ensembling of image backbones looked like the bigger lever.

### Stage 7 - First real submission (7a)
- **Goal:** Prove the submission mechanics: train notebook, checkpoint as a Kaggle Dataset, offline inference, valid CSV.
- **What I did:** Trained the Stage 5 model, saved the best-epoch checkpoint (epoch 1, holdout AUC 0.5767), and fixed five bugs (dataset path, `torch.load` with numpy scalars, offline DINOv2 weights, wrong `train_series` vs `test_series` in inference, and a stale saved notebook version).
- **Result:** Real leaderboard (public) score 0.517, close to chance.
- **Takeaway:** The goal was the pipeline, not the number. "Save Version" is mandatory before submitting.

### Stage 8 - Minimal ensemble (8a)
- **Goal:** Test whether ensembling seeds helps a weak model.
- **What I did:** Trained 3 DINOv2 seeds and combined them with rank averaging and plain probability blending.
- **Result:** Holdout AUC: seeds 0.5732 / 0.6030 / 0.5832, rank averaging 0.6392, blending 0.6254.
- **Takeaway:** Rank averaging beat blending and every single seed, as expected for an ordering metric like AUC.

### Stage 9 - More data from weak labels
- **Goal:** Add training volume using LLM-extracted labels.
- **What I did:** Extracted 400 weak labels and trained on 46 gold (weight 1.0) plus 400 weak (weight 0.3), masking classes not extracted.
- **Result:** Holdout AUC 0.5951, inside the single-seed range (0.5732-0.6030) and below the Stage 8 ensemble (0.6392).
- **Takeaway:** Noisy extra data did not beat a clean-data ensemble.

### Stage 10 - Ensemble on the enlarged data
- **Goal:** Combine the ensemble with the weak-label data.
- **What I did:** Trained 3 seeds on the 446-study set and rank-averaged them.
- **Result:** Holdout AUC: seeds 0.5951 / 0.6257 / 0.4785, rank average 0.6000, below Stage 8 (0.6392).
- **Takeaway:** Weak labels increased variance between seeds instead of reducing it. I closed this line.

### Stage 11 - Study of a reference ensemble notebook
- **Goal:** Understand why a public notebook reached about 0.94.
- **What I did:** This was a reading exercise, not an experiment. I went through the public notebook cell by cell and listed techniques: multi-architecture ensemble, per-class outer routing, residual-gated pooling, multilevel TTA, post-ensemble calibration.
- **Result:** No score. Output was a prioritized list of what to test cheaply.
- **Takeaway:** The score jump came from architecture diversity, not better text extraction.

### Stage 12 - Per-class ensemble weights
- **Goal:** Test a simplified outer routing: weight each seed per class by its holdout AUC.
- **What I did:** Weighted seeds by `clip(AUC - 0.5, 0)` per class and rank-averaged.
- **Result:** Holdout AUC 0.7794, versus 0.6392 for equal weights.
- **Takeaway:** Too good to trust: weights were chosen and scored on the same 12 studies. I flagged it as leakage before validating (see Stage 13).

### Stage 13 - Real submission of per-class weights
- **Goal:** Check whether the Stage 12 gain was real.
- **What I did:** Trained 3 DINOv2 seeds with per-class weights and submitted.
- **Result:** Real public score 0.564 (vs 0.517 in Stage 7).
- **Takeaway:** The idea generalized a little, but holdout 0.7794 became 0.564 real. Holdout is for comparing, not for estimating.

### Stage 14 - Domain prior on attention
- **Goal:** Bias slot attention toward the sequences radiologists use for each finding.
- **What I did:** Added a hand-built prior table to the attention logits and compared 3 seeds with and without it.
- **Result:** Holdout AUC mean 0.5864 (std 0.0124) without the prior vs 0.5798 (std 0.0169) with it.
- **Takeaway:** Negative result: worse mean and larger variance. The table was not validated by a radiologist, so I dropped it.

### Stage 15 - Validating the LLM extraction
- **Goal:** Measure how good the LLM labels actually are and why Stages 9-10 failed.
- **What I did:** Scored the extractor on the 58 gold studies. Tested a refined prompt (v7), then a narrower variant (v7b).
- **Result:** Systematic under-detection (recall: PF OA 0.05, Lateral OA 0.09, Fracture 0.17). Prompt v7 made most classes worse. Effusion recall fell from 0.86 to 0.11. The v7b result is not recorded in the notes.
- **Takeaway:** Averaging more noisy labels cannot fix a systematic bias, so I stopped scaling weak labels.

### Stage 16 - Multi-architecture ensemble
- **Goal:** Replace seed diversity with architecture diversity.
- **What I did:** Added a ResNet50 (ImageNet) and a ResNet50 pretrained on RadImageNet (medical images) to the 3 DINOv2 seeds. Compared against RadImageNet alone.
- **Result:** Real public scores: 3 DINOv2 + ResNet50 0.605, plus RadImageNet at equal weights 0.668, RadImageNet alone 0.741. Holdout: RadImageNet alone 0.7907 vs 0.7287 for the 5-model ensemble.
- **Takeaway:** A domain-matched pretrained model beat any blend with weaker models. This is my best own model.

### Stage 17 - Incremental roadmap (seeds, TTA)
- **Goal:** Squeeze more out of the RadImageNet model cheaply.
- **What I did:** Step 1: 3 RadImageNet seeds with rank averaging. Step 2: horizontal-flip test-time augmentation (TTA). Step 3 (weight tuning) was skipped. The Stage 15 retry (Step 4) is covered above.
- **Result:** Holdout AUC: seed 0 0.7907, seed 1 0.7711, seed 2 0.7830, 3-seed average 0.7889. With flip TTA 0.7698 (without 0.7907).
- **Takeaway:** Both negative. Flipping destroyed information because laterality was already normalized, and averaging diluted the best seed.

### Stage 18 - Reusing public community checkpoints
- **Goal:** Close the gap to about 0.94 after confirming in the competition rules that public models were allowed.
- **What I did:** Ran a public community ensemble (credited in the README), debugged a Notebook Timeout (root cause: accelerator was off, not data volume), then added a second public model and blended 60/40 by rank.
- **Result:** Real public scores: one arm 0.928, four arms 0.932, second model alone 0.930, blend 0.939 (private 0.922).
- **Takeaway:** Reuse of validated public work produced the biggest jump, not my own modeling.

### Stage 19 - Knowledge distillation
- **Goal:** Train my own single model on the soft labels of the 4-arm ensemble.
- **What I did:** Generated ensemble probabilities for 300 studies (pilot), then all 4349 unlabeled studies. Trained RadImageNet on 46 gold (weight 1.0) plus soft-labeled studies (weight 0.5). Fixed a global-variable collision (`SLOTS`) when combining scripts.
- **Result:** Holdout AUC 0.7907 baseline, 0.8109 with the 300-study pilot, 0.7912 at full scale.
- **Takeaway:** The pilot gain vanished at scale. Both fit inside holdout noise, so I did not spend a submission on it.

### Stage 20 - Real 5-fold cross-validation
- **Goal:** Measure how noisy the 12-study holdout really is.
- **What I did:** Ran grouped 5-fold CV (by report hash) over all 58 gold studies with RadImageNet. Fixed a bug where `roc_auc_score` returned NaN instead of raising on single-class folds.
- **Result:** Fold AUCs 0.8387 / 0.8112 / 0.6881 / 0.7460 / 0.7879, out-of-fold AUC 0.6789 (vs holdout 0.7907).
- **Takeaway:** Differences of about 0.02 cannot be told apart from noise at this size. I skipped the planned fold ensemble.

### Stage 21 - Per-class outer routing (ours x community)
- **Goal:** Blend my RadImageNet model with the 0.939 community blend using per-class weights (0.60 default, 0.75-1.00 for five classes).
- **What I did:** Wrote and syntax-checked the script and notebook.
- **Result:** Not recorded. GPU quota ran out and the competition ended before a successful run.
- **Takeaway:** Unvalidated. Reserve compute for the final week.

## Score progression

| Stage | Change | Real score (public unless noted) |
|---|---|---|
| 7 | First working pipeline, weak DINOv2 | 0.517 |
| 13 | 3 DINOv2 seeds + per-class weights | 0.564 |
| 16 | + ResNet50 (ImageNet) | 0.605 |
| 16 | + RadImageNet, equal weights | 0.668 |
| 16 | RadImageNet alone (best own model) | 0.741 |
| 18 | Community ensemble, 1 arm | 0.928 |
| 18 | Second community model alone | 0.930 |
| 18 | Community ensemble, 4 arms | 0.932 |
| 18 | 60/40 blend of both (final, selected) | 0.939 public / 0.922 private |
| 21 | Per-class routing with my model | not recorded |

Final result: 3389th place, private score 0.922.
