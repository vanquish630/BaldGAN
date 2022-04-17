import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import tensorflow as tf
import torch
import cv2

from pylab import *
from glob import glob

from utils.FaceCropper import FaceCropper
from utils.inpainting import Inpainter
from utils import util


def parse_args():
    """Parses arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument('--test_dir', type=str, default='./test_data',
                        help='directory of images to invert.')

    parser.add_argument('--save_align', dest='save_align', action='store_true',
                        help='save alignment and crop of input images.')

    parser.add_argument('--use_mmod', dest='use_mmod', action='store_true',
                        help=' Use MMOD CNN face detection instead of HOG + SVM')

    parser.add_argument('-o', '--output_dir', type=str, default='./results',
                        help='Directory to save the results. If not specified, '
                             '`./results/'
                             'will be used by default.')

    parser.add_argument('--pretrained_dir', type=str, default='./pretrained_models',
                        help='Directory tof pretraied models. If not specified, '
                             '`./pretrained_models/'
                             'will be used by default.')

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for optimization. (default: 0.01)')

    parser.add_argument('--num_iterations', type=int, default=300,
                        help='Number of optimization iterations for inversion. (default: 300)')

    parser.add_argument('--loss_weight_perceptual', type=float, default=5e-5,
                        help='The perceptual loss scale for optimization. '
                             '(default: 5e-5)')

    parser.add_argument('--loss_weight_ssim', type=float, default=3.0,
                        help='The perceptual loss scale for optimization. '
                             '(default: 3.0)')

    parser.add_argument('--loss_weight_regularization', type=float, default=0.2,
                        help='The regularization loss weight for optimization. '
                             '(default: 0.2)')

    parser.add_argument('--crop_size', type=int, default=256,
                        help='Image size for crop. (default: 256)')

    parser.add_argument('--gpu_id', type=str, default='0',
                        help='Which GPU(s) to use. (default: `0`)')

    parser.add_argument('--invert', dest='invert', action='store_true',
                        help='need inversion.')

    parser.add_argument('--inversion_threshold', type=float, default=0.8,
                        help='Threshold for determining sucessfull inversion. (default: 0.8)')

    parser.add_argument('--device', type=str, default='cuda',
                        help='device')

    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device(args.device)

    assert os.path.exists(args.test_dir)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    inpainting_save_path = os.path.join(output_dir, 'inpainting/')
    align_save_path = os.path.join(output_dir, 'aligned_images/')
    latent_codes_save_path = os.path.join(output_dir, 'inversion/')

    if not os.path.exists(align_save_path):
        os.makedirs(align_save_path)

    if not os.path.exists(inpainting_save_path):
        os.makedirs(inpainting_save_path)

    if args.invert:

        model_name = 'styleganinv_ffhq256'
        model_dir = os.path.join(args.pretrained_dir)

        if not os.path.exists(latent_codes_save_path):
            os.makedirs(latent_codes_save_path)

        print('Loading models for stylegan inversion.')

        print('Loading inverter model.')
        inverter = util.build_inverter(model_name=model_name, iteration=args.num_iterations,
                                       pretrained_weights=os.path.join(model_dir, "vgg16.pth"))

        print('Loading generator model.')
        latent_code_to_image_generator = util.get_generator(model_name=model_name)

    inpainter = Inpainter(args.pretrained_dir)

    faceCropper = FaceCropper(args.crop_size)
    image_list = os.listdir(args.test_dir)
    print(f"found {len(image_list)} images in the folder")

    for img in image_list:
        img_name = os.path.basename(img)
        image = Image.open(os.path.join(args.test_dir, img)).convert("RGB")
        image = util.smartResize(image)

        detected_bboxs = faceCropper.faceDetector(image, args.use_mmod)
        numberDetectedFaces = len(detected_bboxs)

        if numberDetectedFaces == 1:
            face_crop_course, face_bbox_course, lmk68 = faceCropper.detect_face_simple(image)
            face_crop_refine, face_bbox_refine = faceCropper.refineCrop(face_crop_course, face_bbox_course)

            if args.save_align:
                plt.imsave(os.path.join(align_save_path, img_name), face_crop_refine)

            inpainting_result = inpainter.inpainting(face_crop_refine, args.crop_size)
            plt.imsave(os.path.join(inpainting_save_path, img_name), inpainting_result)

            resized_image = cv2.resize(inpainting_result, (args.crop_size, args.crop_size),
                                       interpolation=cv2.INTER_CUBIC)

            if args.invert:
                print("starting inversion!")
                latent_code, reconstruction, ssim = util.invert(inverter=inverter, image=resized_image)
                if ssim > args.inversion_threshold:
                    np.save(os.path.join(latent_codes_save_path, f"{img_name.split('.')[0]}_latent_codes.npy"),
                            latent_code[0])
                    generated_images = latent_code_to_image_generator.easy_synthesize(latent_code, **{'latent_space_type': 'wp'})['image']

                    generated_image = cv2.resize(generated_images[0],
                                                 (generated_images[0].shape[1], generated_images[0].shape[0]),
                                                 interpolation=cv2.INTER_CUBIC)
                    plt.imsave(os.path.join(latent_codes_save_path, img_name), generated_image)
                else:
                    print("inversion unsucessfull")


        elif numberDetectedFaces > 1:

            for indx, bbox in enumerate(detected_bboxs):
                if not args.use_hog:
                    bbox = bbox.rect

                face_crop_course, face_bbox_course = faceCropper.detect_face_multiple(image, bbox)
                face_crop_refine, face_bbox_refine = faceCropper.refineCrop(face_crop_course, face_bbox_course)

                if args.save_align:
                    plt.imsave(os.path.join(align_save_path, img_name.split('.')[0] + f"_{indx}.jpg"), face_crop_refine)

                inpainting_result = inpainter.inpainting(face_crop_refine, args.crop_size)
                plt.imsave(os.path.join(inpainting_save_path, img_name.split('.')[0] + f"_{indx}.jpg"),
                           inpainting_result)

                resized_image = cv2.resize(inpainting_result, (args.crop_size, args.crop_size),
                                           interpolation=cv2.INTER_CUBIC)

                if args.invert:
                    print("starting inversion!")
                    latent_code, reconstruction, ssim = util.invert(inverter=inverter, image=resized_image)
                    if ssim > args.inversion_threshold:
                        np.save(
                            os.path.join(latent_codes_save_path, f"{img_name.split('.')[0]}_latent_codes_{indx}.npy"),
                            latent_code[0])
                        generated_images = \
                        latent_code_to_image_generator.easy_synthesize(latent_code, **{'latent_space_type': 'wp'})[
                            'image']

                        generated_image = cv2.resize(generated_images[0],
                                                     (generated_images[0].shape[1], generated_images[0].shape[0]),
                                                     interpolation=cv2.INTER_CUBIC)
                        plt.imsave(os.path.join(latent_codes_save_path, img_name.split('.') + f"_{indx}.jpg"),
                                   generated_image)
                    else:
                        print("inversion unsucessfull")


if __name__ == '__main__':
    main()
