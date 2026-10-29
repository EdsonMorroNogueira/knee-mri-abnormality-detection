# From Zero to 0.922 AUC: What I Learned in a Medical-Imaging Kaggle Competition

**RSNA Knee Abnormality Detection** | Multi-label classification of knee MRI studies (12 findings) | Metric: mean AUC

> **Transparency note.** I made this project with guidance. I had never trained an image model on my own before this challenge, and I worked with an AI assistant (Claude Code in learning mode) as a tutor and pair-programmer. I made the decisions, ran the experiments, and researched the methods. The assistant explained concepts, wrote and reviewed code with me, and helped me debug. Where a result came from someone else's public work, I say so explicitly below.

---

## TL;DR

|                     |                                                                                                       |
| ------------------- | ----------------------------------------------------------------------------------------------------- |
| Final private score | **0.922** (public 0.939)                                                                              |
| Final rank          | **3389th** of 6,299 teams (top 54%)                                                                   |
| My own best model   | **0.741** (fine-tuned RadImageNet ResNet50)                                                           |
| Best submission     | Blend of two public community ensembles, reused with attribution after checking the competition rules |
| Documented stages   | 21, each with a `NOTES.md` recording what was tried and measured                                      |

The headline number is not the main point. **My own models topped out at 0.741. The jump to 0.939 came from reusing public checkpoints.** What I am proud of is everything that surrounds that jump: the validation work that told me which of my ideas were real, and the negative results I measured instead of assuming.

---

## 1. Context

- **Task:** given a knee MRI study (several series: sagittal, coronal and axial, each with or without fat suppression), predict 12 findings (ACL, meniscus tears, fracture, effusion, osteoarthritis, and others).
- **Data constraint:** only **58 studies** had labels I could trust ("gold"). I built them by extracting findings from free-text radiology reports with a local 3B-parameter LLM, then validating the extraction.
- **Goal I set for myself:** study the top public notebook in depth, then build my own implementation rather than copy it.

## 2. Approach (what I built)

1. **Label extraction** from reports with a local LLM, plus DICOM geometry handling (orientation and laterality normalization).
2. **Multiple-Instance Learning (MIL) model:** a study is a _bag_ of up to 6 slots (plane x fat-suppression). A learned attention query per target class decides which slots matter for which finding, with masking for missing slots.
3. **Leak-free validation:** train/validation split grouped by report hash (no duplicate reports across the split). Class weights (`pos_weight`) computed on the training set only.
4. **Rank-percentile blending** instead of averaging raw probabilities, since AUC only depends on ordering.
5. **Architecture diversity:** DINOv2, ResNet50 (ImageNet), and a ResNet50 pretrained on RadImageNet (medical images).
6. **Reuse with attribution:** after confirming in the official rules that using public models and data was allowed, I extracted two community pipelines (a 4-arm CoAtNet ensemble and a residual-gated CoAtNet) and reproduced their 60/40 blend.

## 3. Results

| Stage | What changed                                    | Real score                       |
| ----- | ----------------------------------------------- | -------------------------------- |
| 7a    | First working submission pipeline (weak DINOv2) | 0.517                            |
| 13    | 3 DINOv2 seeds + per-class weights              | 0.564                            |
| 16    | + ResNet50 (ImageNet)                           | 0.605                            |
| 16    | + RadImageNet, equal weights                    | 0.668                            |
| 16b   | **RadImageNet alone (my best own model)**       | **0.741**                        |
| 18    | Community ensemble A, reused with attribution   | 0.932                            |
| 18    | A + community model B, 60/40 rank blend         | **0.939 public / 0.922 private** |

Three things worth reading carefully:

- **My own work ends at 0.741.** The step from 0.741 to 0.932 is not my modeling. It is the decision to verify the rules, reuse validated public work, and reproduce it correctly (the blend ratio matched the reference notebook's internal name, which was my check that I had reproduced the right pieces).
- **The blend helped on the private set too.** The 4-arm community ensemble alone scored 0.930-0.932 public and 0.910 private. The selected 60/40 blend scored 0.939 public and 0.922 private. (My notes record 0.932 at submission time and 0.930 in the final leaderboard table for the same ensemble, so I treat it as about 0.93.)
- **Public to private dropped 0.017 (0.939 to 0.922).** My rank moved up 51 places because other teams dropped more. I read that as a normal leaderboard shake-up, not as proof of generalization.

## 4. What did not work (and how I knew)

Negative results were measured, not assumed:

| Idea                                                  | Outcome                                                                                                                                   | Why it failed                                                                                                                    |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Weak labels from LLM/regex at scale (Stages 9-10)     | No gain: 0.5951 with weak labels (inside the single-seed range) and 0.6000 for the 3-seed ensemble, below the 0.6392 clean-data ensemble  | Noisy labels increased variance between seeds. The extractor under-detects findings systematically, so more volume cannot fix it |
| Prompt engineering on the 3B extractor (Stage 15)     | First attempt made most classes worse and dropped Effusion recall from **0.86 to 0.11** (the follow-up variant's result was not recorded) | A longer, more complex prompt likely overloaded a 3B model. I stopped scaling weak labels                                        |
| Seed ensembling on an already-strong model (Stage 17) | 0.7907 alone vs 0.7889 with 3 seeds                                                                                                       | Seeds converged to similar solutions. Averaging diluted rather than helped                                                       |
| Horizontal-flip TTA (Stage 17)                        | 0.7907 to **0.7698**                                                                                                                      | My pipeline already normalized laterality, so flipping destroyed real anatomical information (medial became lateral)             |
| Knowledge distillation (Stage 19)                     | +0.02 on a 300-study pilot. **~0.00 at full scale** (0.7912 vs 0.7907 baseline)                                                           | The signal was inside the noise of a 12-study holdout (see below)                                                                |

## 5. The lesson that mattered most: "trust your CV", and know its noise floor

My local validation disagreed with the real score repeatedly: holdout 0.7794 became 0.564 (Stage 13), and holdout 0.7907 became 0.741 (Stage 16).

Instead of ignoring this, I ran a real 5-fold grouped cross-validation over all 58 studies (Stage 20):

- Out-of-fold AUC was **0.679**, well below the single holdout's 0.79.
- Fold scores ranged from **0.69 to 0.84**, an amplitude of about 0.15.

Conclusion: with 12 validation studies, differences of ~0.02 cannot be told apart from noise. That reframed the distillation result. The +0.02 pilot gain and the ~0.00 full-scale result both fit inside the noise, so I could not conclude that distillation helps or that it does not. I stopped, and did not spend a submission on it.

Two details that generalize: the local metric was **optimistically biased every time, but the direction of comparisons (A beats B) stayed reliable**, and I should have verified that the holdout could resolve a 0.02 difference _before_ spending ~14x more GPU time scaling the experiment.

## 6. Engineering lessons (the unglamorous ones)

- **A silent device bug.** A shared forward-pass helper (used since Stage 16) never called `.to(device)`, so my "GPU" training had been running on CPU the whole time. No error, just slowness. It was invisible at 58-346 studies and would have cost ~36+ hours at 4,300. I found it while investigating why a long run was slow, and added a numerical equivalence test (full-batch vs. gradient-accumulated minibatches, matching to ~1e-9) before trusting the fix.
- **Global-name collisions in notebooks.** Pasting independently written scripts into one Kaggle notebook silently overwrote a module-level `SLOTS` variable (a 3-field format vs. a 4-field format). My first fix (save and restore) failed because the value was already corrupted by the time I saved it. Lesson: do not patch shared global state. Rebuild it from its source of truth.
- **Debug from the real log.** Repeated "Notebook Timeout" looked like too much data. The log showed the accelerator was off (`gpus 0`). I lost ~4 hours to a hypothesis I could have killed in a minute.
- **Platform behavior is part of the system.** I found that Kaggle's self-reference of a kernel's output only exposes the last _successful_ version, not a cancelled run's checkpoints, which forced a dataset-based checkpoint/resume design for a run that exceeded the 9-hour session limit.
- **Resources are a budget.** My last experiment (per-class outer routing) never got validated because the GPU quota ran out near the deadline. Reserve slack for the end, not just for exploration.

## 7. Limitations (stated plainly)

- Only 58 gold studies. Labels came from LLM extraction of reports, validated but not clinically adjudicated.
- A 12-study holdout is too small to rank close ideas. I corrected for it only late (Stage 20).
- The top score is dominated by reused public work. My original contribution is the pipeline, the validation discipline, and the verified negative results.
- This is a learning project. It is not a clinical tool and must not be used for diagnosis.

## 8. What I would do differently

1. I would build k-fold CV on day one, not at Stage 20. I had already used 5-fold CV in my bachelor's thesis, and that is where I got the idea to use it here.
2. I would read the "External Data and Tools" rules in week one. Community data and models would have been available much earlier, and the biggest jump (0.741 to 0.939) came from them. Thank you to everyone who shared them!
3. I would check the holdout's resolving power before scaling any experiment.
4. I would reserve GPU quota and submission slots for the final week, not only for exploration. I ran out of quota and could not validate my last idea (per-class routing).

## 9. Skills demonstrated

Multiple-instance learning and attention pooling, transfer learning (ImageNet vs. domain-specific pretraining), leak-free grouped validation, k-fold CV and uncertainty-aware interpretation, rank-based ensembling for AUC, local-LLM information extraction, PyTorch debugging (device placement, gradient accumulation), reproducible experiment logging, and responsible reuse of open-source work.

---

_Stage-by-stage log: [stage-log.md](stage-log.md) | General lessons: [lessons-learned.md](lessons-learned.md)_
https://github.com/liquidpass/squad-agents-slack-chat/pulls
