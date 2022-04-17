import os
import sys
sys.path.append('..')

import numpy as np
from PIL import Image
import cv2
import face_alignment

from .inverter import StyleGANInverter
from  models.invert_model.helper import build_generator
import scipy.ndimage



def build_inverter(model_name,pretrained_weights,logger = None, iteration=300, regularization_loss_weight=0.6 , loss_weight_ssim=3 ):
  """Builds inverter"""
  inverter = StyleGANInverter(
      model_name,
      pretrained_weights = pretrained_weights,
      learning_rate=0.01,
      iteration=iteration,
      reconstruction_loss_weight=1.0,
      perceptual_loss_weight=5e-5,
      regularization_loss_weight=regularization_loss_weight,
      loss_weight_ssim = loss_weight_ssim,
      logger = logger,
      
      )
  return inverter



def get_generator(model_name):
  """Gets model by name"""
  return build_generator(model_name)



def invert(inverter, image):
  """Inverts an image which has been preprocessed."""
  latent_code, reconstruction , ssim_loss = inverter.easy_invert(np.array(image), num_viz=1)
  return latent_code, reconstruction, ssim_loss


def flatten(t):
    return [item for sublist in t for item in sublist]


def smartResize(image):

  image = np.array(image)

  min_indx = np.argmin((image.shape[0], image.shape[1]))
  if image.shape[min_indx] > 512:

    dim1 = min(512, image.shape[min_indx])
    wpercent = image.shape[int(not min_indx)] / image.shape[int(min_indx)]
    dim2 = int(512 * wpercent)

    if min_indx == 0:
      image = np.array(Image.fromarray(image).resize((dim2, dim1), Image.ANTIALIAS, ))
    else:
      image = np.array(Image.fromarray(image).resize((dim1, dim2), Image.ANTIALIAS, ))

  return image

