# ClaimGuard: Detecting AI-Generated and AI-Edited Insurance Claim Photos

ClaimGuard is a computer-vision classifier that estimates the probability that a claim photo
(e.g. car or property damage) was **generated or edited with AI**. It is designed as a
**triage tool**: flagged photos go to a human adjuster, and nothing is auto-rejected.

> Status: 🚧 in progress (Step 1 of 11: problem framing, research basis and repo setup)

## 1. Problem framing

| Item | Decision |
| --- | --- |
| Task | Binary image classification |
| Positive class (1) | Fake: fully AI-generated or AI-inpainted |
| Negative class (0) | Real camera photo |
| Model output | Probability that the image is fake (0 to 1) |
| Decision | Flag for manual review if probability ≥ threshold (chosen on validation set) |

### Error types and their cost

| Error | Meaning | Business cost |
| --- | --- | --- |
| False positive (FP) | Honest photo flagged as fake | Customer delay, adjuster review time, loss of trust in the tool |
| False negative (FN) | Fake photo passed as real | Fraudulent payout |

### Success criteria

- **Headline metric:** ROC-AUC (the ranking metric of the NTIRE 2026 robust detection challenge)
- **Operating point:** recall ≥ 90% on fakes at a false positive rate ≤ 5%
- **Generalisation:** ROC-AUC on a generator family **never seen in training** (FLUX)
- **Robustness:** < 10-point drop in ROC-AUC after JPEG compression, resizing, blur and cropping

Accuracy is **not** a primary metric: fraud is rare, so a model that predicts "real" for
everything can score high accuracy while catching nothing.

## 2. Research basis (2026)

Every design choice follows a recent paper. Full details and links: [docs/research_basis.md](docs/research_basis.md).

| Design choice | Based on |
| --- | --- |
| Benchmark 5 frozen foundation models (CLIP, DINOv2, DINOv3, SigLIP2, Perception Encoder) | *TAP into the Patch Tokens* (2026) |
| Attention head over patch tokens, not only the global token | *TAP into the Patch Tokens* (2026) |
| Hybrid: semantic branch + forensic noise-residual branch with a learned gate | *FUSED* (Aug 2026) |
| Generators chosen for architectural diversity; FLUX held out | *PROBE: Where Detectors Fail* (May 2026) |
| Real-world transformations for augmentation and robustness tests; ROC-AUC as headline | *NTIRE 2026 Challenge* (CVPR 2026 workshop) |
| Inpainted region pasted back onto the original photo | *DiffusionPrint* (Apr 2026) |
| Optional texture-masking ablation | *MPFT* (Jan 2026) |

## 3. Repository structure

```
claimguard/
  configs/config.yaml       # all paths, seeds, backbones, generators, transforms
  data/                     # images (git-ignored) + metadata.csv
  docs/
    problem_framing.md      # Step 1 decisions in detail
    research_basis.md       # which paper drives which decision
  notebooks/                # EDA, experiments, evaluation
  src/
    utils.py                # seeding, config loading
    build_metadata.py       # scans image folders -> metadata.csv with leakage-safe splits
    check_setup.py          # checks libraries and that all 5 backbones exist
  scripts/make_dummy_data.py  # tiny fake dataset for testing the pipeline
  tests/                    # pytest checks: no leakage, hold-out respected, reproducible
  reports/figures/          # saved plots
```

## 4. Setup

```bash
conda create -n claimguard python=3.11 -y
conda activate claimguard
pip install -r requirements.txt
python -m src.check_setup   # verify environment
pytest -q                   # verify split logic (5 tests)
```

## 5. Roadmap

- [x] Step 1: Problem framing, research basis and repo setup
- [ ] Step 2: Dataset — real photos, diverse generators, inpainted edits, leakage-safe splits
- [ ] Step 3: EDA — shortcuts (format, size, EXIF), frequency spectra, embedding plots
- [ ] Step 4: Preprocessing and NTIRE-style augmentation
- [ ] Step 5: Baselines — ResNet-50 fine-tune + linear probes on 5 frozen foundation models
- [ ] Step 6: Hybrid model — patch-token attention head + forensic branch + gated fusion, with ablation
- [ ] Step 7: Evaluation — confusion matrix, ROC/PR, threshold at 5% FPR, bootstrap CIs
- [ ] Step 8: Generalisation (held-out FLUX) and robustness tests
- [ ] Step 9: Grad-CAM and error analysis
- [ ] Step 10 (optional): Localisation of edited regions
- [ ] Step 11: Streamlit demo

## 6. Ethics and limitations

This model is a decision-support tool. Detectors degrade on unseen generators and heavily
compressed images, and false positives fall on honest customers, so final decisions must
stay with a human.
