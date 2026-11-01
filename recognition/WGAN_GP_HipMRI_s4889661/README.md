# WGAN-GP vs DCGAN for Synthetic Prostate MRI Generation (HipMRI 2D)

**Author:** Jayden Niddrie (s4889661) · **Course:** COMP3710 Pattern Recognition · **Difficulty:** Normal

## 1. Problem and Algorithm

Medical AI models for prostate MRI, such as organ segmentation or cancer detection, need large training sets, but real scans are scarce, expensive to acquire and protected by patient-privacy law. The HipMRI 2D dataset used here contains only 38 patients. This project builds a prototype generative model that produces synthetic 2D pelvic MRI slices (128×64) as a possible supplement for pretraining, method development and data sharing. Synthetic data is only useful if it looks anatomically realistic, and only safe to share if it is genuinely new rather than a copy of a real patient’s scan. The project therefore investigates the domain’s fidelity–memorisation–privacy dilemma: it compares a DCGAN baseline with a WGAN-GP model, evaluating both on anatomical fidelity, diversity, memorisation of training patients and computational cost.

## 2. How It Works

A generative adversarial network (GAN) trains two networks against each other. A generator turns a 100-dimensional random noise vector into a fake slice, and a discriminator compares slices against real training data; its feedback drives the generator to improve. Both models here share an identical DCGAN-style generator (a 4×2 feature map upsampled by five stride-2 transposed convolutions to a 1×128×64 image, with BatchNorm, ReLU and a tanh output), so any difference in results comes from the training objective alone. The DCGAN baseline’s discriminator is a real/fake classifier trained with binary cross-entropy. Once it becomes confident, its output saturates and the generator receives weak gradients, which can cause unstable training and mode collapse. WGAN-GP replaces it with a critic that outputs an unbounded score; the gap between its scores for real and fake images estimates the Wasserstein (earth-mover’s) distance, which gives the generator useful gradients even when it is losing. A gradient penalty keeps the critic smooth by penalising its gradient norm at points blended between real and fake images. The critic uses LayerNorm instead of BatchNorm, because the penalty requires each image to be scored independently, and it is updated five times per generator step.

## 3. Feasibility Review

COMP3710 (PR) – s4889661 

Feasibility Review

1. User Need, Scope & Acceptance Criteria

Intended user: Medical AI researchers training models for prostate MRI tasks who are limited by scarce, privacy restricted patient scans.

Prototype purpose:  To generate realistic synthetic 2D prostate MRI slices that could supplement real training data and be shared without exposing patients. This addresses the open dilemmas of fidelity vs memorisation vs privacy. Where these synthetic slices are only useful if they look anatomically realistic, and only safe to share if they are not copies of real patients' scans.

Scope: Unconditional 2D slice generation at 128×64 from the HipMRI 2D dataset (no segmentation masks, no 3D volumes).

The 5 Acceptance criteria:
    Fidelity: mean SSIM (similarity structural index measure) > 0.6 between generated slices and their closest real slice. 
    Memorisation / privacy: generated slices are not closer (nearest- neighbour distance) to the training set than real unseen test patients are.
    Diversity: uncurated sample grids across several random seeds show no mode collapse.
    Resource constraint: each model trains in under 2 hours with under 8 GB peak VRAM on a single Colab GPU.
    Baseline comparison: WGAN-GP is compared against the DCGAN baseline on criteria 1–4 using the same data split.

2. Model Choice & Course Concepts

Proposed dataset configuration: We were provided HipMRI 2D slices (keras_slices_data), using the provided patient-level split: train (31 patients) for training only; validate/test (7 unseen patients) as the reference set for the memorisation audit. Slices are scaled per-slice to [-1, 1] and downsampled
from 256×128 to 128×64 (2:1 aspect ratio preserved).

Model architecture (Claude helped me produce architecture):

Generator (Implemented and will be shared by both models): 
- Input: 100-dimensional noise vector
- First Layer turns it into a small 512 channel feature map of size 4×2
- Five “upsampling” layers (transposed convolutions) each double the height and width: 4×2 → 8×4 → 16×8 → 32×16 → 64×32 → 128×64.
- Channels shrink as the image grows: 512 → 256 → 128 → 64 → 32 → 1 (greyscale).
- BatchNorm + ReLU between layers. tanh at the end, so pixel values land in [-1, 1], matching how we scaled the real data.
- 3.20M parameters.
- Status: built, shape and range tested, and used in the smoke test.

Baseline discriminator (DCGAN): Implemented, tested and trained in the smoke test
DCGAN = Deep Convolutional GAN

- Input: a 1x128x64 slice real of fake
- Five strided convolutions each halve height and width: 128×64 → … → 4×2.
- Channels grow: 1 → 32 → 64 → 128 → 256 → 512.
- No normalisation on the first layer, then BatchNorm + LeakyReLU (DCGAN convention).
- A final layer squashes 4×2 into one number: a logit, i.e. “how real does this look”.
- Trained with binary cross-entropy (real = 1, fake = 0).
- 2.79M parameters.
- Status: built, tested, and trained for 2 epochs in the smoke test.

WGAN-GP Critic: (not implemented yet) 
WGAN-GP = Wasserstein GAN with Gradient Penalty
- What’s built: the critic network is the same discriminator class with norm="layer". It swaps BatchNorm for LayerNorm, because the gradient penalty needs each image scored independently. It’s shape-tested (outputs one score per image) and has the same 2.79M parameters.
- What’s not built yet (the WGAN-GP training):
- The Wasserstein loss: the critic tries to push real scores up and fake scores down, with no sigmoid and no probabilities.
- The gradient penalty: the “keep scores smooth” rule.
- The critic loop: training the critic ~5 times per generator update.
- WGAN-GP optimiser settings.

Chosen difficulty band: Normal since DCGAN and WGAN-GP on HipMRI 2D are both normal in the specification.
 
3. Preliminary Feasibility Evidence

Initial data audit: After downloading theHipMRI 2D slices from Rangpur and audited them with shell commands and self test in dataset.py. There was 12,660 slices from 211 scans (60 slices per scan) of 38 patients. The provided split is patient-level (31/4/3 patients), and I verified that no patient appears in more than one split. Slices are 256×128 (one scan is 256×144 and is center-cropped to 256×128).

Baseline smoke-test experiment: I trained a DCGAN baseline for 2 epochs on 256 training slices on my laptop to confirm the full pipeline and confirm it works. The pipeline ran end to end in 6.6 s with finite losses, and checkpoints, sample grids and loss plots were saved. It was observed that the discriminator dominated quickly  (mean D(real) = 0.98, D(fake) = 0.05) which is an early sign of the DCGAN instability that for when we provide our WGAN-GP we expect to fix this instability.

Measured resource estimate: My model self-tests reports 3.20M generator parameters and 2.79M discriminator parameters. The smoke test showed that on epoch on 256 slices about ~0.9 s per epoch on 256 slices. I’m yet to measure peak VRAM or full dataset epoch time.

4. Risks, Budget & Fallback

Training instability: My smoke test showed an early dominating discriminator which can starve the generator of gradients, causing stalled training or mode collapse. To monitor this my training script saves sample grid from the same fixed nose every epoch 
Memorisation: My audit showed only 31 training patients with about 6 scans each, and my data preview showed that neighbouring slices within a scan look almost identical. This repetition makes it easier for a GAN to reproduce real slices, which would defeat the privacy purpose. This will be checked with a nearest-neighbour audit comparing generated slices against real slices from unseen test patients.

Metric ambiguity: A generated slice that is very similar to a training image could reflect good quality or copying, so I will always report SSIM alongside the memorisation audit rather than on its own.

Computing budget: about 2 GPU-hours per model on a single Colab GPU (~4 GPU-hours total, plus short evaluation runs).
The next planned experiment: full DCGAN baseline run on to Colab to measure time per epoch and peak VRAM. I will then implement and train WGAN-GP on the same split.

Workable fallback: lets say WGAN-GP is unstable, I will tune the number of critic steps per generator step, the gradient-penalty weight and the learning rate. And if it underperforms DCGAN, I will report this as a finding with analysis. Both models are normal difficulty so wont affect the mark ceiling 

## 4. Dataset, Pre-processing and Splits

TODO: describe the dataset (HipMRI 2D slices, 12,660 slices, 211 scans × 60
slices, 38 patients), pre-processing (per-slice min-max to [-1, 1];
centre-crop one 256×144 scan to 256×128; bilinear downsample to 128×64
keeping the 2:1 aspect ratio), and justify the split (provided patient-level
split 31/4/3, verified no overlap; GANs train on train only; test patients
used as the unseen-real reference for the memorisation audit).

## 5. Dependencies and Reproducibility

| Package | Version |
|---|---|
| Python | TODO |
| torch | TODO |
| torchvision | TODO |
| nibabel | TODO |
| numpy | TODO |
| matplotlib | TODO |

TODO: reproducibility notes — fixed seeds (training seed 42, evaluation seeds
0–3 and 1234); two DCGAN smoke tests produced identical losses; all
hyperparameters are command-line arguments; hardware used (Rangpur A100 for
training, Apple MPS / CPU for smoke tests and evaluation).

## 6. Usage

```bash
# Data audit + preview
python dataset.py --root /path/to/keras_slices_data

# Train (local smoke test)
python train.py --root /path/to/keras_slices_data --max-files 256 --epochs 2 --batch-size 32 --run-name dcgan_smoke

# Train (full runs on Rangpur)
sbatch slurm/train_dcgan.sh
sbatch slurm/train_wgan_gp.sh

# Evaluate both trained generators
python predict.py --root /path/to/keras_slices_data \
    --dcgan runs/dcgan_baseline/generator.pt --wgan runs/wgan_gp/generator.pt
```

TODO: one or two sentences on what each script outputs.

## 7. Results

### 7.1 Example inputs and outputs

Real (unseen test patients):
![Real test slices](images/real_test.png)

DCGAN (uncurated):
![DCGAN samples](images/uncurated_dcgan.png)

WGAN-GP (uncurated):
![WGAN-GP samples](images/uncurated_wgangp.png)

TODO: describe what you see (layout, bladder placement, sharpness vs blur,
artefacts, femoral heads).

### 7.2 Training behaviour

![DCGAN loss curves](images/loss_curves_dcgan.png)
![WGAN-GP loss curves](images/loss_curves_wgangp.png)

TODO: DCGAN discriminator dominance (D(x) 0.83→0.95, D(G(z)) 0.19→0.05, loss
spikes ep 27/30) vs WGAN-GP smooth training (GP steady 0.18–0.31;
Wasserstein estimate plateau ~14.7 and why it's ambiguous).

### 7.3 Quantitative benchmarking

512 generated samples per model (seed 1234); reference = 540 real test slices
(3 unseen patients).

| Metric | Real test (reference) | DCGAN | WGAN-GP |
|---|---|---|---|
| Fidelity: best SSIM to 20 nearest train slices (mean ± std) | 0.397 ± 0.035 | 0.363 ± 0.045 | 0.381 ± 0.053 |
| Worst-6 sample SSIM | — | 0.25–0.26 | 0.22–0.25 |
| L2 distance to nearest train slice (median) | 23.28 | 24.59 | 22.68 |
| % samples closer than real 5th pct L2 (expect ~5%) | 5% | 8.8% | 19.1% |
| corr(contrast, L2 distance) | — | +0.74 | +0.65 |
| Max SSIM to a train slice | 0.503 | 0.486 | 0.514 |
| % samples above real 95th pct SSIM (expect ~5%) | 5% | 2.1% | 8.0% |

TODO: interpret — WGAN-GP closer to real-data fidelity on average but with a
worse tail; why the 0.6 SSIM target is not reachable even by real scans
under this definition (~0.40 ceiling).

### 7.4 Resource profiling

| | DCGAN | WGAN-GP |
|---|---|---|
| Generator parameters | 3,197,376 | 3,197,376 |
| Discriminator / critic parameters | 2,791,808 | 2,791,808 |
| Training time per epoch (A100) | 1.9 s | 19.5 s |
| Total training time, 30 epochs | 130.6 s (incl. one-off data cache build) | 660.8 s |
| Peak PyTorch-allocated VRAM | 0.27 GB | 0.34 GB |

TODO: interpret — ~10× slower per epoch (5 critic steps + gradient-penalty
backward pass), ~25% more memory (create_graph), both far within budget.

### 7.5 Memorisation audit

![Memorisation histogram](images/memorisation_histogram.png)
![WGAN-GP nearest pairs](images/nearest_pairs_wgangp.png)
![DCGAN nearest pairs](images/nearest_pairs_dcgan.png)

TODO: the L2 flag on WGAN-GP, the contrast confound, the SSIM near-copy
check, visual inspection of pairs, conclusion (no evidence of memorisation;
L2 audits confounded by blur), caveat (reference from only 3 patients).

### 7.6 Latent interpolation

![Latent interpolation](images/latent_interpolation.png)

TODO: smooth transitions in both (continuous latent space); DCGAN pair 2
local collapse; WGAN-GP pair 2 mid-frame blur.

## 8. Failure Case Autopsy

![Failure candidates](images/failure_candidates.png)

TODO: 5 cases, each with what happened / expected / likely cause / model
difference:
1. Misplaced organ / global incoherence (DCGAN #456)
2. High-frequency artefacts and background speckle (DCGAN #221, #368)
3. Local mode collapse (DCGAN interpolation pair 2)
4. Blur and washed-out slices (WGAN-GP #205, #192, #129)
5. Missing femoral heads (both)

## 9. Acceptance Criteria Outcome

| Criterion (from feasibility review) | Outcome |
|---|---|
| 1. Fidelity: SSIM > 0.6 | TODO |
| 2. Memorisation / privacy | TODO |
| 3. Diversity | TODO |
| 4. Resource: < 2 h, < 8 GB | TODO |
| 5. Baseline comparison | TODO |

## 10. Engineering Recommendation

TODO: your verdict to the project manager — is the synthetic cohort safe
and useful? Which model, and why? Is WGAN-GP's extra cost justified? What
must happen before sharing (quality filtering of the bad tail, structural
memorisation checks, more training, clinical review)?

## 11. Limitations and Future Work

TODO: e.g. unconditional (no masks/labels), 128×64 resolution, 30 epochs,
reference set of only 3 patients, SSIM/L2 limitations, no FID.

## 12. References

1. A. Radford, L. Metz, S. Chintala, "Unsupervised representation learning with deep convolutional generative adversarial networks," ICLR, 2016.
2. M. Arjovsky, S. Chintala, L. Bottou, "Wasserstein generative adversarial networks," ICML, 2017.
3. I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, A. Courville, "Improved training of Wasserstein GANs," NeurIPS, 2017.
4. Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," IEEE TIP, 2004.
5. I. Goodfellow et al., "Generative adversarial nets," NeurIPS, 2014.

## 13. Artificial Intelligence Usage Disclosure

TODO: tools used, tasks assisted, how you verified (build from ai_notes.md).