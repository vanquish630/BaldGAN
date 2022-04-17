import sys
sys.path.append("..")

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import tensorflow as tf
from models.hypergan_models.model import Model as hyperganModel
from .segment import return_hair_mask, return_face_mask, returnfacebbox



class Inpainter(object):
    def __init__(self , model_dir = "./pretrained_models"):
        self.model_dir = model_dir
        self.load()

    def load(self):
        print('Loading models for hypergan inpainting.')
        self.hypergan_model = hyperganModel()
        self.hypergan_generator = self.hypergan_model.build_generator()
        checkpoint = tf.train.Checkpoint(generator=self.hypergan_generator)
        checkpoint.restore(os.path.join(self.model_dir, "ckpt-25"))

    def inpainting(self, image, size=256):
        balds = []

        gt_image = np.array(image)

        gt_image_shape = gt_image.shape

        gt_image = cv2.resize(gt_image, (size, size)) / 255

        gt_image = np.array(gt_image, dtype=np.float32)

        mask = return_hair_mask(gt_image)

        mask = cv2.resize(mask.astype('float32'), (size, size))
        gt_image = np.expand_dims(gt_image, axis=0)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(np.array(mask, dtype=np.uint8), kernel, iterations=2)
        if mask.max() > 1:
            mask = mask / 255
        mask = mask[None, ..., None]

        input_image = np.where(mask == 1, 1, gt_image)
        input_image = input_image[0][None, ...]

        prediction_coarse, prediction_refine = self.hypergan_generator([input_image, mask], training=False)

        normalized = (np.array(prediction_refine[0]) - np.array(prediction_refine[0]).min()) / \
                     ((np.array(prediction_refine[0]).max()) - (np.array(prediction_refine[0]).min()))
        balds.append(prediction_refine)

        normalized = np.array(normalized * 255, dtype=np.uint8)

        # smartCropped = returnfacebbox(normalized)

        normalized = cv2.resize(normalized, (gt_image_shape[1], gt_image_shape[0]), interpolation=cv2.INTER_CUBIC)

        return normalized
