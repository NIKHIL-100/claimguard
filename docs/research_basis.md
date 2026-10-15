# Research basis: which 2026 papers ClaimGuard is built on

Each design decision in this project comes from a specific recent paper. This file is the
"why" behind the code, and the answer to the interview question
"how did you decide on this approach?".

## Core papers (2026)

| # | Paper | Venue / date | Main finding | What we take from it |
|---|---|---|---|---|
| 1 | **TAP into the Patch Tokens: Leveraging Vision Foundation Model Features for AI-Generated Image Detection** (Abdullah, Ebert, Wasenmüller) — https://arxiv.org/html/2604.26772v1 | arXiv, 2026 | Newer frozen foundation models beat the original CLIP by >12% accuracy. An attention-pooling head over the **patch tokens** (not only the CLS token) improves generalisation; state of the art on Chameleon (95.6%) | **Step 5:** benchmark five frozen backbones (CLIP, DINOv2, DINOv3, SigLIP2, Perception Encoder) with a linear probe. **Step 6:** add a TAP-style attention head over patch tokens |
| 2 | **FUSED: Forensic-Semantic Mixture-of-Experts for AI Inpainting Detection and Localization** (Nuzhdin, Worring, Najdenkoska) — https://arxiv.org/abs/2608.28302 | arXiv, Aug 2026 | Inpainting changes only a small region. Combining low-level forensic cues with high-level semantic features, with a learned gate choosing between them, gives the best detection and localisation on unseen generators. Also finds detectors partly rely on a global generator artifact | **Step 6:** hybrid model = semantic branch (frozen backbone) + forensic branch (noise residual), fused with a learned gate. **Step 10:** image score + pixel mask. **Step 9:** test whether our model relies on that global artifact |
| 3 | **Where Detectors Fail: Probing Generative Space for Generalizable AI-Generated Image Detection (PROBE)** (Cao et al.) — https://arxiv.org/abs/2605.24906 | arXiv, May 2026 | Detectors fail on new generators because training data lacks *variety* of generative processes, not because it lacks volume | **Step 2:** choose generators from **different architecture families** (UNet diffusion, diffusion transformer, rectified flow) instead of many images from one model |
| 4 | **NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild** (Gushchin, Abud et al.) — https://arxiv.org/abs/2604.11487 | CVPR 2026 workshop | 42 generators, 36 real-world transformations (crop, resize, compression, blur). Detectors are ranked by **ROC-AUC** on transformed + clean images | **Steps 4 and 8:** our augmentation and robustness test use the same transformation families; ROC-AUC is our headline metric |
| 5 | **Detecting AI-Generated Images via Distributional Deviations from Real Images (MPFT)** (Niu, Chen, Zhang) — https://arxiv.org/abs/2601.03586 | arXiv, Jan 2026 | Frozen CLIP groups real images together but cannot separate fakes on its own. Masking textured regions during fine-tuning forces the model to learn "deviation from real" rather than one generator's traces | **Step 6 (ablation):** optional texture-masking during training; also explains why a frozen linear probe has a ceiling |
| 6 | **DiffusionPrint: Learning Generative Fingerprints for Diffusion-Based Inpainting Localization** (Giakoumoglou, Papadopoulos) — https://arxiv.org/abs/2604.12443 | arXiv, Apr 2026 | Inpainting re-generates the whole image through the VAE decoder, destroying camera noise; each generator leaves a learnable fingerprint | **Steps 2 and 10:** we paste the inpainted region back onto the original photo so only the edited area carries generator traces (otherwise the task becomes too easy and unrealistic) |

## Supporting evidence

| Paper | Why it matters |
|---|---|
| **Navigating the Challenges of AI-Generated Image Detection in the Wild (ITW-SM)** (Konstantinidou et al., 2026) | Preprocessing choices matter more than more data. Supports careful cropping instead of resizing in Step 4 |
| **Is AI-Generated Image Detection a Solved Problem? (AIGIBench)** (2025) | Detectors with >95% lab accuracy fail on new generators. Motivates the held-out generator test |
| **Image-Adaptive Prompt Learning (IAPL)** (Li et al., CVPR 2026) | Current state of the art on GenImage (96.7%). Cited as a reference point, not implemented |

## Foundations (the building blocks)

ResNet (He et al., 2016) · CLIP (Radford et al., 2021) · DINOv2 (Oquab et al., 2023) ·
LoRA (Hu et al., 2022) · Grad-CAM (Selvaraju et al., 2017) · U-Net (Ronneberger et al., 2015)

## What ClaimGuard adds that these papers don't

- A dataset in the **insurance domain** (car and property damage), where the realistic attack is a small inpainted edit on a real photo.
- Evaluation at a **business operating point** (recall at 5% false positive rate) rather than accuracy alone.
- A transparent comparison of five 2025–2026 foundation models on the same small, domain-specific dataset.
