# Datasheet — data used in this submission

## 1. LFW — Labeled Faces in the Wild (real faces: fine-tuning, validation, test)

| | |
|---|---|
| Source | University of Massachusetts Amherst, http://vis-www.cs.umass.edu/lfw/ (archive mirror used by scikit-learn: https://ndownloader.figshare.com/files/5976018) |
| Citation | G. B. Huang, M. Ramesh, T. Berg, E. Learned-Miller. *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments.* UMass Tech. Report 07-49, 2007 |
| Content | 13,233 news photographs of 5,749 public figures, 250×250 JPEG, collected by the dataset authors from news articles |
| Usage conditions | Distributed publicly by the authors as a research benchmark. No formal licence is attached; the authors ask that it be cited, and warn that it is **not** suitable for claiming a face-verification product is ready for commercial use (it has too few images of some groups). Used here only for non-commercial research. |
| Why it is acceptable under the brief | Public research benchmark, not scraped by us, and contains no law-enforcement mugshots or criminal-record information |
| Subset used | Identities with ≥ 4 images: **610 identities, 6,733 images** |
| Pre-processing | YuNet face detection (OpenCV Zoo, Apache-2.0) → subject face (largest, closest to the centre) → our own 5-point similarity (Umeyama) alignment onto the 112×112 ArcFace template → JPEG q95. 6,733 / 6,733 detected, so no fallback crops were needed. Normalised to [-1, 1] at load time. |
| Layout | `data/processed/lfw_aligned/person_001/img_01.jpg …` with `lfw_meta.csv` mapping back to the original LFW names |
| Duplicates | Two-stage check: pHash (64-bit) on the full original photo, Hamming ≤ 8 → candidate. Confirmed if the normalised pixel correlation of 64×64 grey thumbnails is ≥ 0.93. That gives **10 confirmed duplicate pairs, all within one identity** (e.g. Ari_Fleischer_0006 ≈ _0011). Hashing the aligned crops was rejected: it produced cross-identity "duplicates" between different people, checked by eye. |

### Split (identity-disjoint, seed 42)

| Split | Identities | Images used | Purpose |
|---|---|---|---|
| train | 427 | 4,891 | stage-1 (jointly with DigiFace) and stage-2 training, INT8 calibration, distillation |
| val | 61 | 378 | model selection, threshold selection |
| test | 122 | 1,110 (of 1,464; capped at 30 / identity) | every reported number |

* No identity appears in more than one split (`tests/test_splits.py`).
* Duplicate clusters never cross a split boundary (`tests/test_splits.py`).
* Val/test identities are capped at 30 images (seeded sample), so George W. Bush (530 images) or Colin Powell cannot dominate the pairs. The capped-out images are unused. They are never moved into training.

## 2. DigiFace-1M (synthetic faces, stage-1 pre-training only)

| | |
|---|---|
| Source | Microsoft, https://github.com/microsoft/DigiFace1M |
| Citation | G. Bae et al. *DigiFace-1M: 1 Million Digital Face Images for Face Recognition.* WACV 2023 |
| Content | Rendered faces of procedurally generated people. **No real person is depicted**, so the data carries no consent or privacy problem. |
| Licence | Microsoft *Research Use of Data Agreement v1.0*: **non-commercial research only**. Models trained on it count as "Results" and inherit the restriction. |
| Subset used | all five 72-image parts `subjects_0-1999 … subjects_8000-9999`: **10,000 identities × 72 = 720,000 images** |
| Pre-processing | Provided as tightly aligned 112×112 crops. We measured them against our template with our detector: median landmark offset 2.7 px, so they were used as-is (RGBA → RGB, alpha is constant 255). |

## 3. What the data is not

* No images were scraped by us; no law-enforcement or mugshot data; no private individuals.
* The test identities are real public figures from LFW. Results describe performance on press photographs of mostly adult, mostly light-skinned, mostly male public figures and should not be read as performance on other populations.
