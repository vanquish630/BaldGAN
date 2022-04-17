# python 3.7
"""Contains basic configurations for models used in this project."""

import os.path


MODEL_DIR = os.path.realpath("pretrained_models")

MODEL_POOL = {
    'styleganinv_ffhq256': {
        'resolution': 256,
        'repeat_w': False,
        'final_tanh': True,
        'use_bn': True,
    },
}

# Settings for StyleGAN.
STYLEGAN_TRUNCATION_PSI = 0.7  # 1.0 means no truncation
STYLEGAN_TRUNCATION_LAYERS = 8  # 0 means no truncation
STYLEGAN_RANDOMIZE_NOISE = False

# Settings for model running.
USE_CUDA = False

MAX_IMAGES_ON_DEVICE = 2

MAX_IMAGES_ON_RAM = 800


def get_weight_path(weight_name):
  """Gets weight path from `MODEL_DIR`."""

  assert isinstance(weight_name, str)
  if weight_name == '':
    return ''
  if weight_name[-4:] != '.pth':
    weight_name += '.pth'
  return os.path.join(MODEL_DIR, weight_name)
