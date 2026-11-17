# Assignment 3 - M1 Dataset Proposal

## Image-text classification with PairDX

**CO3133 - Deep Learning and Its Applications, Semester 261**  
**Group 09 | Instructor: Lê Thành Sách | Prepared: 18 November 2026**

Lê Phước Vũ (2353341) · Nguyễn Vũ Long (2352696) · Nguyễn Hoàng Quốc (2353027)

**Status: proposal prepared for instructor review; approval has not been recorded.** Main implementation and experiments will begin only after Approved or Approved with conditions, as required by handbook Sections 7.6 and 27. This document contains preliminary metadata analysis and a proposed experimental protocol, with no trained-model results.

## 1. Task and dataset choice

We propose **six-class image-text classification**. Each input is a medical figure paired with its original English caption. The output is its broad document/image type, taken from the metadata field `image_type`. This task studies whether a caption and an image provide complementary evidence for recognizing the type of medical content. It does not predict a diagnosis or establish clinical performance.

The selected source is **PairDX, version 1.0**, published on 27 October 2025, DOI [10.5281/zenodo.17452236](https://doi.org/10.5281/zenodo.17452236) [1]. The publisher lists the creator as “Hosam Lafi, Hamed Aljuhani.” Its `PairDX.zip` is 1,930,257,035 bytes (about 1.93 GB); the advertised MD5 is `d37ddb0324967957fe9d1a3e0fac7109`. This checksum has not yet been verified against a full local download.

**Important discrepancy:** the record describes 22,665 balanced pairs, but the supplied ZIP directory contains **103,384 WEBP entries**, and `image_metadata.json` contains **103,384 newline-delimited records**. The archive does not supply a separate file identifying the advertised 22,665-pair subset. We request approval to use the six-class archive population below, followed by the stated quality checks. We will not describe this population as balanced or claim to reproduce the advertised subset.

| Proposed class (`image_type`) | Candidate pairs | Share |
| --- | ---: | ---: |
| Radiology | 68,020 | 65.79% |
| Medical photograph | 22,068 | 21.35% |
| Endoscopy | 3,830 | 3.70% |
| Chart | 3,830 | 3.70% |
| Ophthalmic imaging | 3,695 | 3.57% |
| Electrography | 1,941 | 1.88% |
| **Total** | **103,384** | **100%*** |

*Rounded shares may not sum to exactly 100%. Counts are before exclusions, deduplication, and splitting. The majority/minority ratio is about 35:1. The candidate population exceeds the handbook's default classification threshold of five classes and about 5,000 pairs; final retained counts will be checked again.*

<!-- pagebreak -->

## 2. Preliminary pair quality and distribution

We inspected the complete ZIP central directory, archive readme, and decompressed metadata using HTTP byte ranges on 18 November 2026. The readme identifies MultiCaRe record **14994046** as the upstream source [2]. It spells the dataset name “PairXD.” The file named `image_metadata.json` must be read as JSON Lines rather than one JSON array.

| Metadata check | Observed result |
| --- | --- |
| Unique image filenames / metadata rows | 103,384 / 103,384 |
| Blank or null captions / filenames / `image_type` | 0 / 0 / 0 |
| Distinct case IDs / source article IDs | 30,219 / 28,426 |
| Distinct original figures (`main_image`) | 62,691 |
| Caption words: median / 95th percentile / maximum | 16 / 48 / 450 |
| Captions of at most three whitespace-separated words | 5,668 (5.48%) |
| Repeated-caption rows after lowercase/whitespace normalization | 14,560 (14.08%) |

Repeated-caption rows count every member of a repeated-text group; there are 94,530 distinct normalized captions and 8,854 rows beyond one occurrence per caption. Generic captions and multiple panels can make text ambiguous. Nonempty metadata does not prove a correct image-caption match. Full filename-to-ZIP matching, image decoding, image hashes, and visual pair review remain to be completed before training. No missing-image or mismatch rate has been measured yet.

**Alignment and annotation:** join `file` to the image basename and rebuild the relative archive path; ignore the original absolute `file_path`. Retain `case_id` and `main_image` for provenance. The shared record links the original figure/panel and caption. Use only the six broad `image_type` values as targets; `image_subtype`, label lists, IDs, filenames, and case narratives are not model inputs. The metadata distinguishes machine-learning labels from a separate semisupervised ground-truth field; 44,012 records have that latter field empty. We therefore treat the supplied targets as potentially noisy dataset labels, not independently verified clinical ground truth.

Before splitting, decode images, remove invalid or empty pairs, and inspect a fixed random sample of 20 pairs per class (seed 42), plus short-caption and multi-panel cases. Report image/caption agreement, panel errors, and uncertain labels separately. Keep valid short captions in the primary dataset; report their performance as a subgroup. A label word in an original caption is legitimate task evidence, but can create shortcuts; include a caption keyword-masking sensitivity check.

## 3. Split unit and leakage prevention

Use the **source PMC article ID** (prefix of `case_id`) as the minimum split unit. All cases, original figures, and panels from one article stay together. Patient identities across different articles are unavailable, so patient-disjoint evaluation cannot be guaranteed.

Before assigning splits, connect article groups that share exact decoded-image hashes or identical normalized captions of at least eight words. Review near-duplicate image candidates and merge confirmed duplicate groups. Generic short captions are not used to connect unrelated articles. Assign these connected groups to approximately **70% training, 15% validation, and 15% test**, balancing class counts as closely as group constraints allow, with split seed 42. Publish actual per-split counts and assert zero article/group/hash overlap. Never use test outcomes to choose the split, filters, or model settings.

<!-- pagebreak -->

## 4. Planned encoders and required models

Use an ImageNet-pretrained **ResNet-18** image encoder [3] with its 512-dimensional pooled representation and **`sentence-transformers/all-MiniLM-L6-v2`** [4] with its 384-dimensional text embedding. Initially freeze both encoders and cache features, so all model comparisons share identical representations. Use the published ResNet preprocessing (resize, center crop to 224 × 224, RGB conversion, ImageNet normalization); visually check cropping of wide figures. Use the text model's tokenizer, masked mean pooling and embedding normalization, with a maximum of 256 wordpieces. Report actual truncation rates; whitespace word counts above are not token counts.

| Model | Inputs and representation | Fusion / prediction |
| --- | --- | --- |
| A: image only | Frozen ResNet-18, 512 dimensions | Trainable MLP, six logits |
| B: text only | Frozen MiniLM, 384 dimensions | Trainable MLP, six logits |
| C: simple fusion | Both embeddings, each projected to 256 dimensions | Intermediate fusion by concatenation; MLP, six logits |
| D: improved fusion | Same embeddings and 256-dimensional projections | Learned feature gate; weighted sum; MLP, six logits |

For D, let `v` and `t` be projected image and text features. Compute a 256-dimensional gate `g = sigmoid(W[v;t] + b)` and fused representation `z = g*v + (1-g)*t`. This lets the classifier vary its reliance on each modality. The hypothesis is that gating helps when captions are generic or visual evidence is unclear; improvement is not assumed.

**Loss and training:** each unimodal model uses class-weighted cross-entropy. Each fusion model uses one class-weighted cross-entropy on its fused prediction, with no auxiliary losses. Compute weights from training counts only, using inverse square-root frequency normalized to mean one. Train heads and projections with AdamW, learning rate 0.001, weight decay 0.0001, batch size 128, dropout 0.2, and a maximum of 30 epochs. Stop after five epochs without validation macro-F1 improvement. Select the best validation macro-F1 checkpoint, breaking ties by lower validation loss. Use run seeds **42, 43, 44** on the same fixed split. These are proposed settings, not tuned results.

No encoder fine-tuning is required for the initial comparison. Any later fine-tuning must be reported as a separate experiment with the same data protocol and explicit compute cost. Primary evaluation uses complete pairs; incomplete inputs are rejected rather than silently replaced. A and B explicitly cover single-modality prediction.

## 5. Evaluation, controls, and ablations

**Primary metric: macro-F1.** Also report accuracy, per-class precision/recall/F1, confusion matrices, and mean/standard deviation over three seeds. Include a training-majority-class dummy baseline; always predicting radiology would achieve 65.79% on the unsplit candidate population, illustrating why accuracy alone is inadequate. Compute its actual held-out accuracy after splitting.

Compare A, B, C and D with the same split, cached encoders, loss weights, optimizer settings and checkpoint rule. Report head parameter counts and inference cost because fusion capacity differs. For the controlled gate ablation, change only D's learned gate to a fixed 0.5 gate, keeping the projections, classifier, loss, seeds, and training budget fixed. Separately mask an explicitly documented list of modality keywords in captions at evaluation to assess shortcut sensitivity; report this as a stress test, not a replacement test set.

Judge gating by paired per-seed macro-F1 differences and a source-group bootstrap confidence interval. If the interval includes zero, report no clear improvement. Show support cases where fusion corrects a unimodal error, conflict cases where modalities disagree, and fusion errors involving generic captions, panel mismatch, minority classes, or noisy labels. Do not require fusion to outperform the unimodal baselines.

<!-- pagebreak -->

## 6. Compute estimate and subset rules

The local GPU is a verified **NVIDIA GeForce GTX 1660 SUPER with 6,144 MiB VRAM**. Extract frozen features one encoder at a time, initially with batch size 32 and float32; reduce the batch if needed. The two feature arrays require approximately **370.5 MB** for 103,384 pairs, excluding IDs and framework overhead.

Budget **2-6 GPU-hours** for feature extraction, **1-4 hours** for 12 main head-training runs and three gate-ablation runs, and **1-2 hours** for evaluation. Reserve **4-12 hours** overall; these are unmeasured estimates. Calibrate with 1,000 training examples after approval. Allow roughly 10 GB free disk space plus Python/package storage. Measure peak GPU memory against a target below 5 GiB.

Use **all eligible pairs** by default. If measured cost prevents completion, request approval for a training-only cap of 4,000 pairs per class, selected with seed 42 after splitting. Keep validation/test unchanged and use the same subset for all models. Report retained counts and verify the course size threshold.

## 7. License, limitations, and approval request

The PairDX record declares MIT, but its per-image metadata contains nine license values, including CC BY, CC BY-NC, CC BY-NC-SA, CC BY-NC-ND and unspecified entries. The upstream MultiCaRe record describes the collection as CC BY-NC-SA with item-specific licenses [2]. We will preserve per-item attribution/license metadata, check intended use against those terms, and link to the original download rather than redistribute medical images as MIT. Unclear-permission items will be excluded if their terms cannot be established. Any exclusions will be counted by class before splitting.

Limitations include the description/archive mismatch, imbalance, noisy labels, caption shortcuts, multi-panel figures, duplicated text and unknown patient overlap. Visual pair correctness and image integrity remain unverified. This study supports no diagnostic or patient-care claim.

**Approval requested:** PairDX record 17452236, six-class `image_type` classification on the supplied archive, article/duplicate-group splits, four models, macro-F1 and accuracy. Please confirm acceptance of the population differing from the advertised 22,665 pairs and any label/license conditions. The group will record the decision before main experiments.

## 8. Milestones and responsibilities

| Milestone (23:59, GMT+7) | Planned deliverable |
| --- | --- |
| M1 - 18 November 2026 | Dataset proposal, preliminary metadata EDA, split/model/metric/compute plan |
| M2 - 25 November 2026 | After approval: validated pairs, frozen group split, A/B/C quantitative results |
| M3 - 2 December 2026 | D, controlled ablation, error/compute analysis, report, slides, video, page and checkpoints |

Lê Phước Vũ: pair validation, preprocessing and EDA. Nguyễn Vũ Long: encoders, unimodal models and fusion. Nguyễn Hoàng Quốc: controlled evaluation, error analysis and report. All members contribute to the final presentation and review the submission.

## References

[1] [PairDX v1.0, Zenodo record and download](https://zenodo.org/records/17452236). Preliminary counts above were computed directly from `PairDX.zip`, not copied from its description.

[2] [MultiCaRe, source version identified by the archive readme](https://zenodo.org/records/14994046).

[3] [Torchvision ResNet-18 documentation and pretrained transforms](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

[4] [Sentence Transformers all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

[5] CO3133 Project Handbook, revision 14 September 2026, Sections 7.6 and 24-31 (course-provided PDF).
