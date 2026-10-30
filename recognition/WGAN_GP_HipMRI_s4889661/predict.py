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
from dataset import load_split


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

def to_display(img):
    """[1, H, W] tensor in [-1, 1] -> H x W array in [0, 1], rotated 90°.

    The NIfTI slices load rotated relative to the usual radiological view;
    rotation is applied for display only and never affects training or metrics.
    """
    return np.rot90(((img[0].numpy() + 1) / 2).clip(0, 1))

# =============================================================================
# 1. UNCURATED SAMPLE GRIDS (DIVERSITY)
# =============================================================================

def plot_uncurated_grid(G, name, seeds, n_per_seed, out_dir, device):
    """One row per seed, n_per_seed samples per row, nothing selected or
    removed, so the grid honestly reflects the model's output diversity."""
    fig, axes = plt.subplots(len(seeds), n_per_seed,
                             figsize=(n_per_seed * 1.6, len(seeds) * 0.9 + 0.6))
    for r, seed in enumerate(seeds):
        imgs = generate(G, n_per_seed, seed, device)
        for c in range(n_per_seed):
            ax = axes[r, c]
            ax.imshow(to_display(imgs[c]), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        axes[r, 0].set_title(f"seed {seed}", fontsize=8, loc="left")

    fig.suptitle(f"{name}: uncurated samples "
                 f"({len(seeds)} seeds × {n_per_seed}, no selection)")
    plt.tight_layout()
    path = os.path.join(out_dir, f"uncurated_{name.lower().replace('-', '')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return 
    
def plot_real_grid(images, split, n_rows, n_cols, out_dir, seed=0):
    """Random (seeded, unselected) real slices in the same layout as the
    generated grids, as a visual reference for fidelity."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(images), generator=gen)[: n_rows * n_cols]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.6, n_rows * 0.9 + 0.6))
    for ax, i in zip(axes.flat, idx):
        ax.imshow(to_display(images[i]), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    fig.suptitle(f"Real {split} slices: random sample "
                 f"({n_rows * n_cols}, no selection)")
    plt.tight_layout()
    path = os.path.join(out_dir, f"real_{split}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path

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

    # --- Real reference data (train = what the GANs learned from;
    #     test = unseen patients, used as the reference for the audit) -------
    real = {split: load_split(args.root, split)["images"]
            for split in ("train", "test")}
    print(f"Real slices: train {tuple(real['train'].shape)}, "
          f"test {tuple(real['test'].shape)}")
    path = plot_real_grid(real["test"], "test", len(args.seeds),
                          args.n_per_seed, args.out_dir)
    print(f"  saved {path}")

    models = {
        "DCGAN": load_generator(args.dcgan, device),
        "WGAN-GP": load_generator(args.wgan, device),
    }
    for name, G in models.items():
        sample = generate(G, 2, seed=0, device=device)
        print(f"Loaded {name}: {count_params(G):,} params, "
              f"sample {tuple(sample.shape)}, range [{sample.min():.2f}, {sample.max():.2f}]")
        path = plot_uncurated_grid(G, name, args.seeds, args.n_per_seed,
                                   args.out_dir, device)
        print(f"  saved {path}")


if __name__ == "__main__":
    main()