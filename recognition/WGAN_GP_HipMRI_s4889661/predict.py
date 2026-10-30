"""
predict.py — Example usage and evaluation of the trained HipMRI GANs

Loads the trained DCGAN (baseline) and WGAN-GP generators and produces the
evidence for the fidelity / memorisation / privacy investigation:
    1. Uncurated sample grids across several random seeds (diversity)
    2. Fidelity: SSIM between generated slices and their closest real slice
    3. Memorisation audit: nearest-neighbour distances vs unseen test patients
    4. Latent interpolation between random noise vectors

Usage
-----
    python predict.py --root ~/keras_slices_data \
        --dcgan ~/Desktop/dcgan_baseline/generator.pt \
        --wgan  ~/Desktop/wgan_gp/generator.pt

Outputs are written to figures/ (gitignored); selected figures are copied
into images/ for the README.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")                 # save figures to file; no display needed
import matplotlib.pyplot as plt
import numpy as np                    # allowed in predict.py for visualisation
import torch

from modules import Generator, Z_DIM, count_params


# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained HipMRI GANs")
    p.add_argument("--root", default=os.path.expanduser("~/keras_slices_data"),
                   help="path to keras_slices_data (for real reference slices)")
    p.add_argument("--dcgan", required=True, help="path to DCGAN generator.pt")
    p.add_argument("--wgan", required=True, help="path to WGAN-GP generator.pt")
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3],
                   help="random seeds for uncurated sample grids")
    p.add_argument("--n-per-seed", type=int, default=8)
    return p.parse_args()


# =============================================================================
# LOADING TRAINED GENERATORS AND SAMPLING
# =============================================================================

def load_generator(path, device):
    """Rebuild the Generator and load trained weights (eval mode: BatchNorm
    uses the running statistics learned during training)."""
    G = Generator()
    G.load_state_dict(torch.load(os.path.expanduser(path), map_location=device))
    return G.to(device).eval()


@torch.no_grad()
def generate(G, n, seed, device):
    """Generate n slices from a fixed seed, so every figure is reproducible."""
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n, Z_DIM, generator=gen).to(device)
    return G(z).cpu()                                  # [n, 1, 128, 64] in [-1, 1]


# =============================================================================
# DISPLAY HELPERS
# =============================================================================


# =============================================================================
# 1. UNCURATED SAMPLE GRIDS (DIVERSITY)
# =============================================================================


# =============================================================================
# 2. FIDELITY: SSIM TO THE CLOSEST REAL SLICE
# =============================================================================


# =============================================================================
# 3. MEMORISATION AUDIT: NEAREST-NEIGHBOUR DISTANCES
# =============================================================================


# =============================================================================
# 4. LATENT INTERPOLATION
# =============================================================================


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    device = torch.device("cpu")      # evaluation is light; CPU keeps it simple
    os.makedirs(args.out_dir, exist_ok=True)

    models = {
        "DCGAN": load_generator(args.dcgan, device),
        "WGAN-GP": load_generator(args.wgan, device),
    }
    for name, G in models.items():
        sample = generate(G, 2, seed=0, device=device)
        print(f"Loaded {name}: {count_params(G):,} params, "
              f"sample {tuple(sample.shape)}, range [{sample.min():.2f}, {sample.max():.2f}]")


if __name__ == "__main__":
    main()