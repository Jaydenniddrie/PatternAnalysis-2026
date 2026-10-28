"""
modules.py — Network components for the HipMRI GAN project

Contains
--------
* Generator      : noise vector z -> synthetic 1x128x64 MRI slice (shared by
                   the DCGAN baseline and the WGAN-GP model)
* Discriminator  : 1x128x64 slice -> real/fake score (logit). With
                   norm="batch" it is the DCGAN discriminator; with
                   norm="layer" it becomes the WGAN-GP critic.
* weights_init   : DCGAN weight initialisation

Architecture follows DCGAN (Radford, Metz & Chintala, ICLR 2016), adapted
from square 64x64 images to the 2:1 HipMRI aspect ratio (128x64).

Per the assessment spec, this file uses PyTorch only (no NumPy).
"""

import torch
import torch.nn as nn


# =============================================================================
# CONFIGURATION
# =============================================================================

Z_DIM = 100          # length of the input noise vector (DCGAN default)
IMG_CHANNELS = 1     # greyscale MRI
BASE_CH = 64         # channel width multiplier for G and D

# Smallest feature map is 4x2 (height x width). Five 2x upsamples give
# 4x2 -> 8x4 -> 16x8 -> 32x16 -> 64x32 -> 128x64, matching our data.
START_HW = (4, 2)


# =============================================================================
# GENERATOR
# =============================================================================

def _up_block(in_ch, out_ch):
    """Transposed conv that doubles H and W, then BatchNorm + ReLU."""
    return nn.Sequential(
        # kernel 4, stride 2, padding 1 -> output size = 2 x input size
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Generator(nn.Module):
    """Maps noise z [B, Z_DIM] to an image [B, 1, 128, 64] in [-1, 1]."""

    def __init__(self, z_dim=Z_DIM, base=BASE_CH):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            # --- project noise to a 4x2 feature map ---------------------------
            # z is reshaped to [B, z_dim, 1, 1]; a (4,2) kernel with stride 1
            # expands the 1x1 input to 4x2
            nn.ConvTranspose2d(z_dim, base * 8, kernel_size=START_HW, stride=1,
                               padding=0, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),                      # [B, 512, 4, 2]

            # --- upsample, halving channels each time --------------------------
            _up_block(base * 8, base * 4),              # [B, 256, 8, 4]
            _up_block(base * 4, base * 2),              # [B, 128, 16, 8]
            _up_block(base * 2, base),                  # [B, 64, 32, 16]
            _up_block(base, base // 2),                 # [B, 32, 64, 32]

            # --- final upsample to image, no BatchNorm on the output ----------
            nn.ConvTranspose2d(base // 2, IMG_CHANNELS, kernel_size=4, stride=2,
                               padding=1, bias=False),  # [B, 1, 128, 64]
            nn.Tanh(),   # output range [-1, 1], matching dataset.py scaling
        )

    def forward(self, z):
        # Accept z as [B, z_dim] and add the two spatial dims the convs expect
        return self.net(z.view(z.size(0), self.z_dim, 1, 1))

# =============================================================================
# DISCRIMINATOR / CRITIC
# =============================================================================


# =============================================================================
# WEIGHT INITIALISATION
# =============================================================================


# =============================================================================
# HELPERS
# =============================================================================


# =============================================================================
# SELF-TEST: SHAPES, OUTPUT RANGE, PARAMETER COUNTS
# =============================================================================