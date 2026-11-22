# Model card: ResNet50 face embedder (+ ResNet18 edge student)

| | |
|---|---|
| Task | Face verification (1:1, "same person?") and identification (1:N, gallery search) |
| Input | 112×112 RGB face, aligned with 5 landmarks to the ArcFace template, scaled to [-1, 1] |
| Output | 512-D L2-normalised embedding. Two faces are compared with cosine similarity (= dot product) |
| Architecture | torchvision ResNet50 (ImageNet init), stem max-pool removed, 1×1 conv 2048→512, BN-Dropout-FC-BN head. 37.4 M params. |
| Training | Stage 1: DigiFace-1M (synthetic, 10k identities, 720k images) jointly with the 427 LFW *train* identities. Stage 2: fine-tune on the LFW train split. Loss: ArcFace (s, m) + 0.1 × batch-hard triplet. P×K identity-balanced sampling. |
| Selection | Best epoch by TAR@FAR=1% on LFW **validation** identities (disjoint from train and test) |
| Decision rule | `cosine ≥ threshold → match`. Threshold fixed on validation at a target FAR. See `results/evaluation_results.json` → `verification.final_threshold` |
| Edge variant | ResNet18 student, distilled from the ResNet50 (cosine + relational loss), ONNX → static INT8 (ONNX Runtime). It has its own validation-calibrated threshold (`edge_threshold`). |

## Intended use
Research and demonstration of a face-verification pipeline; a technical-assessment deliverable.

## Out-of-scope use
* Any commercial use: the DigiFace-1M licence is non-commercial, and the trained weights inherit that restriction.
* Surveillance, identifying people without their consent, law-enforcement decisions, or any fully automated decision with legal or similarly significant effect.
* Security-critical access control without a liveness / presentation-attack-detection stage (none is included).

## Evaluation data
LFW test split: 122 identities never used for training or model selection, 5,000 genuine + 5,000 impostor pairs, plus all-vs-all scores. Metrics, with bootstrap 95 % CIs, are in `report.pdf` and `results/evaluation_results.json`.

## Known limitations and bias
* LFW is press photography of public figures, skewed towards adult, light-skinned men in good lighting. Error rates for other groups were **not** measured and are likely higher.
* DigiFace is synthetic. The synthetic-to-real gap is narrowed by fine-tuning on real LFW train identities, but not removed.
* Each FAR is an estimate from a limited number of impostor identities; FAR=0.1% on 122 identities has wide uncertainty.
* Faces that are heavily occluded, in profile or at very low resolution degrade the embedding, and the detector may miss them.

## Privacy by design (API / demo)
Only 512-D embeddings are stored; no face images are kept. Enrolments can be deleted (`DELETE /enroll/{name}`). Embeddings are still biometric data and must be protected as such.
