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