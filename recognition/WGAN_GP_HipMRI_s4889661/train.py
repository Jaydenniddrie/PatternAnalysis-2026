"""
train.py — Training script for the HipMRI GANs

Trains the DCGAN baseline on the HipMRI 2D training split, logging losses,
discriminator confidence, fixed-noise sample grids, checkpoints and resource
usage (time, peak VRAM, parameters) for the README.

WGAN-GP training is added in a later milestone via --model wgan-gp.

Usage
-----
    # quick local smoke test
    python train.py --root ~/keras_slices_data --max-files 256 --epochs 2 \
        --batch-size 32 --run-name dcgan_smoke
    # full baseline run (Colab GPU)
    python train.py --root /content/keras_slices_data --epochs 30 --run-name dcgan_baseline

Outputs (in runs/<run-name>/, gitignored)
-----------------------------------------
    samples/epoch_XXX.png   fixed-noise sample grid per epoch
    checkpoint.pt           full state (G, D, optimisers, epoch, args)
    generator.pt            generator weights only (used by predict.py)
    history.json            per-epoch metrics + run summary
    loss_curves.png         training loss / discriminator confidence plots
"""

import argparse
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")                 # save plots to file; no display needed
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.utils import save_image

from dataset import load_split, get_loader
from modules import Generator, Discriminator, weights_init, count_params, Z_DIM


# =============================================================================
# COMMAND-LINE ARGUMENTS (all hyperparameters explicit for reproducibility)
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train a GAN on HipMRI 2D slices")
    p.add_argument("--root", required=True, help="path to keras_slices_data")
    p.add_argument("--model", default="dcgan", choices=["dcgan"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)      # DCGAN paper value
    p.add_argument("--beta1", type=float, default=0.5)    # DCGAN paper value
    p.add_argument("--z-dim", type=int, default=Z_DIM)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-files", type=int, default=None, help="subset for smoke tests")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run-name", default=None)
    return p.parse_args()

# =============================================================================
# REPRODUCIBILITY AND DEVICE
# =============================================================================

def set_seed(seed):
    """Seed Python and PyTorch RNGs (CUDA kernels may still be nondeterministic)."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Prefer CUDA (Colab), then Apple MPS (local Mac), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# =============================================================================
# DCGAN TRAINING STEP (one batch)
# =============================================================================

def dcgan_step(G, D, real, opt_G, opt_D, bce, z_dim, device):
    """One DCGAN update: discriminator step, then generator step.

    Returns per-batch statistics for logging.
    """
    b = real.size(0)
    ones = torch.ones(b, 1, device=device)    # "real" targets
    zeros = torch.zeros(b, 1, device=device)  # "fake" targets

    # --- 1. Discriminator: real -> 1, fake -> 0 -------------------------------
    z = torch.randn(b, z_dim, device=device)
    fake = G(z)
    d_real = D(real)
    d_fake = D(fake.detach())     # detach: this step must not update G
    loss_D = bce(d_real, ones) + bce(d_fake, zeros)

    opt_D.zero_grad(set_to_none=True)
    loss_D.backward()
    opt_D.step()

    # --- 2. Generator: make D output 1 on fakes (non-saturating loss) ---------
    # Re-score the same fakes with the just-updated D; gradients flow into G.
    loss_G = bce(D(fake), ones)

    opt_G.zero_grad(set_to_none=True)
    loss_G.backward()   # also writes grads into D, but opt_D zeroes them next step
    opt_G.step()

    return {
        "loss_D": loss_D.item(),
        "loss_G": loss_G.item(),
        # Mean probability D assigns: ideal equilibrium is ~0.5 for both
        "D_real": torch.sigmoid(d_real).mean().item(),
        "D_fake": torch.sigmoid(d_fake).mean().item(),
    }

# =============================================================================
# OUTPUTS: SAMPLE GRIDS, CHECKPOINTS, PLOTS
# =============================================================================

@torch.no_grad()
def save_samples(G, fixed_z, path):
    """Save an 8x8 grid generated from the SAME noise every epoch."""
    G.eval()
    imgs = G(fixed_z)
    save_image(imgs, path, nrow=8, normalize=True, value_range=(-1, 1))
    G.train()


def save_checkpoint(G, D, opt_G, opt_D, epoch, args, out):
    """Full state for resuming, plus generator-only weights for predict.py."""
    torch.save({
        "epoch": epoch,
        "G": G.state_dict(), "D": D.state_dict(),
        "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
        "args": vars(args),
    }, os.path.join(out, "checkpoint.pt"))
    torch.save(G.state_dict(), os.path.join(out, "generator.pt"))


def plot_losses(history, out):
    """Plot losses and discriminator confidence over epochs."""
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, [h["loss_D"] for h in history], label="D loss")
    ax1.plot(epochs, [h["loss_G"] for h in history], label="G loss")
    ax1.set_xlabel("epoch"); ax1.set_title("Training losses"); ax1.legend()

    ax2.plot(epochs, [h["D_real"] for h in history], label="D(real)")
    ax2.plot(epochs, [h["D_fake"] for h in history], label="D(G(z))")
    ax2.axhline(0.5, ls="--", c="grey", lw=0.8)
    ax2.set_xlabel("epoch"); ax2.set_title("Discriminator confidence"); ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out, "loss_curves.png"), dpi=120)
    plt.close()

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")


if __name__ == "__main__":
    main()