# Knee MRI Abnormality Detection: From Zero to 0.922 AUC

My first machine-learning competition: detecting 12 findings (ACL tear, meniscus tears, fracture, effusion, osteoarthritis, and others) from knee MRI studies, for the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) Kaggle competition. Metric: mean AUC.

| | |
|---|---|
| Final private score | **0.922** (public 0.939) |
| Final rank | **3389th** of 6,299 teams (top 54%) |
| My own best model | **0.741** |
| Documented stages | 21 |

## Read this first (honest framing)

- **My own models topped out at 0.741.** The jump to 0.939 came from reusing public community checkpoints, after I verified the competition rules allowed it. They are credited below and are **not included** in this repository.
- **I worked with an AI assistant (Claude Code)** as a tutor and pair-programmer. I made the decisions and ran the experiments. It helped me learn faster, review code, and debug.
- What I own and am proud of: the data pipeline, the validation discipline, and a set of **measured negative results** (flip TTA, seed ensembling, weak labels, distillation) that taught me more than the wins.
- This is a learning project. It is **not a clinical tool** and must not be used for diagnosis.

## Why I did this

I started this project because I needed to challenge myself to get better at data science. At the beginning, the thing I understood least was how image recognition works (I still don't understand it well), but image models seem really cool and I want to understand them better. The mistake that taught me the most was trusting a validation set of only 12 studies: it said 0.79, and the real score was 0.74. A 5-fold cross-validation showed me how noisy it was, and that is why I stopped trusting small differences (see [lessons learned](docs/lessons-learned.md)).

I'm sharing this because I want to motivate people who don't take on a challenge because they are afraid to try. Don't fear it, just do it. Athletes forget the shot they missed and try again, and again, and again. I will definitely try another competition. I don't know if it will be vision or image analysis, but I will definitely do it.

## Where to start

| If you want... | Read |
|---|---|
| The story, results, and what failed | [docs/case-study.md](docs/case-study.md) |
| Every stage in order, with numbers | [docs/stage-log.md](docs/stage-log.md) |
| Lessons that transfer to other competitions | [docs/lessons-learned.md](docs/lessons-learned.md) |

## What the model does

A knee MRI study is not one image: it is a *bag* of up to 6 slots (sagittal, coronal and axial, each with or without fat suppression), and not every slot exists for every study. This is a multiple-instance learning problem. A learned attention query per target class decides which slots matter for which finding, with masking for missing slots (see `src/model/multi_query_attention_pool.py`).

## Repository map

```
src/
  data/         DICOM slice ordering, laterality, slot selection, physical-scale crop, pixel cache
  labels/       LLM extraction of 12 findings from radiology reports, validation against gold labels
  model/        multi-query attention pool, backbones (DINOv2, timm ResNet50, RadImageNet)
  training/     training and train/validation loops
  validation/   report-grouped split, train-only class weights, grouped k-fold CV
  ensembling/   rank-percentile averaging, per-class weighted blending
  experiments/  flip TTA (negative result), knowledge distillation (inconclusive)
tests/          unit tests for the attention pool (run locally, no data needed)
docs/           case study, stage log, lessons learned
```

The scripts were written to run inside Kaggle notebooks (each stage was pasted into a notebook session with the competition dataset mounted), so this is a curated study archive, not an installable package.

## Running the tests

```bash
pip install -r requirements.txt
python -m pytest -q tests
```

Only the attention-pool tests run without data. Everything else needs the competition dataset, which is not redistributed here (see the competition rules).

## Credits

The 0.939 submission blended two public community pipelines. Their code and weights are **not** included. Please credit and read the original work:

- [Head and Shoulders, Knees and Toes](https://www.kaggle.com/code/prvsiyan/head-and-shoulders-knees-and-toes) by prvsiyan: the 4-arm CoAtNet ensemble ("Raptor") that I used as one half of the blend.
- [Bend the Knee to the Dinosaurs](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs) by mattiaangeli, with the [CoAtNet checkpoints](https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coat-resgated-ep10-top3) it uses (`coat_resgated`): the other half of the blend. That notebook credits Pilkwang, Sofia Anjenje, Antoine G., prvsiyan, Marwan Mahmoud, dreaddevelopment / Roman Tamrazov, renta.k and Anvith Pothula, and so do I.
- [RSNA Knee Hybrid Raptor: Lateral Meniscus R100](https://www.kaggle.com/code/renta0426/rsna-knee-hybrid-raptor-lateral-meniscus-r100) by renta0426: the per-class routing idea between two models that I studied (my own attempt at it was never validated).
- Also studied: the public notebook "Read the Report, Then the Knee" and a notebook that summarizes the public solutions' progression ("Evolution of Public Solutions"). I did not record their authors, so they are not linked here.

## License

MIT for the code in this repository. See [LICENSE](LICENSE).
