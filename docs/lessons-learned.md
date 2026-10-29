# Lessons Learned

Lessons from this project that apply to any competition with a numeric metric. Nothing here is specific to knees, DICOM, or multiple-instance learning.

## Validation

### Trust your CV, and know its noise floor
Build a local validation that mirrors the real train/test relationship (same grouping, same leakage risks). When the local metric and the real score disagree, treat it as a signal to investigate, not to ignore. In this project the local holdout was optimistic every single time (0.7794 became 0.564, and 0.7907 became 0.741), but the *direction* of comparisons (A beats B) stayed reliable.

Before optimizing anything, ask: *what could leak between my train and validation that would not leak between the full train set and the real test set?* Duplicate rows, related groups (same patient, same match, same author), overlapping time windows. Build the split to remove that leak before training the first model.

### Check that your validation can resolve the effect you are measuring
With 12 validation studies, a 5-fold cross-validation showed fold scores from 0.69 to 0.84. A 0.02 difference is invisible at that noise level. A pilot that gained +0.02 and a full-scale run that gained ~0.00 were both consistent with no effect. Estimate the noise floor first, then decide how much compute an experiment deserves.

### Compute class weights on the training split only
Computing `pos_weight` on the full data before splitting leaks the positive rate of the validation set into training.

## Modeling

### Rank-average for ranking metrics
If the metric only depends on ordering (AUC is the classic case), convert each model's output to a percentile before combining. It is robust to models with different confidence scales. This does not apply to metrics that depend on absolute values (RMSE, calibrated log-loss).

### Real diversity beats more seeds
Ensembling different architectures or pretraining sources gave the largest solo gains. Ensembling seeds of one already-strong model did not help (0.7907 alone vs. 0.7889 with three seeds): the seeds converged to similar solutions.

### A textbook trick can hurt if your pipeline already solved the problem
Horizontal-flip test-time augmentation dropped the score from 0.7907 to 0.7698, because the pipeline had already normalized laterality, so flipping destroyed real anatomy.

### Cheap noisy labels do not replace trusted labels
Scaling training with automatically extracted weak labels did not help (the 3-seed ensemble got worse, 0.6392 to 0.6000). And a prompt change that looked like an improvement collapsed one class's recall from 0.86 to 0.11 on the first attempt. Measure the extractor against gold labels after every change.

## Process

### Read the rules on external data and models in week one
Confirming that public models were allowed unlocked the largest jump in the project (0.741 to 0.939), and it was available far earlier than I used it. Always credit the original authors.

### Treat compute and submissions as a budget
The last experiment never got validated because GPU quota ran out near the deadline. Reserve slack for the final week.

### Debug from the real log, not the first hypothesis
Repeated "Notebook Timeout" errors looked like too much data. The log showed the accelerator was off. About four hours were lost to a hypothesis a one-minute log read would have killed.

### Mind silent failures
A helper that never moved tensors to the GPU ran on CPU for weeks without any error. Log the device at startup, and benchmark at small scale against an expected throughput before scaling up.

### Beware generic global names when combining scripts
When several independently written scripts are pasted into one notebook, names like `SLOTS` can be silently overwritten by another script. Saving and restoring a global fails if it was already overwritten before you saved it. Rebuild it from its source of truth, or keep scripts in separate modules.

### Document each experiment as you go
One short notes file per experiment (what was tried, what was measured, what it means) made it possible to write the full retrospective later without reconstructing anything from memory. Negative results belong in it too.
