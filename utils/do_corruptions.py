import numpy as np
from PIL import Image
import cv2
# import torchvision.transforms as transforms

common_corruptions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


def add_gaussian_noise(img, level=1):
    mu = 0.0
    sigma = level * 0.02
    noise = np.random.normal(mu, sigma, img.shape)
    processed_img = img + noise
    if processed_img.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    processed_img = np.clip(processed_img, low_clip, 1.0)
    return processed_img


# poisson noise
def add_shot_noise(img, level=1):
    vals = len(np.unique(img))
    vals = 2**np.ceil(np.log2(vals))
    processed_img = np.random.poisson(img * vals) / float(vals)
    return processed_img


# salt & pepper noise
# svp: salt vs pepper ratio, amount:the percentage of modified pixels
def add_impulse_noise(img, level=1, s_vs_p=0.5):
    amount = level * 0.02
    processed_img = np.copy(img)
    # add salt noise
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    processed_img[coords[0], coords[1], :] = [1, 1, 1]
    # add pepper noise
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    processed_img[coords[0], coords[1], :] = [0, 0, 0]
    return processed_img


def add_defocus_blur(img, level=1):
    return img


def add_glass_blur(img, level=1):
    k = 2 * level + 1
    return cv2.GaussianBlur(img, ksize=(k, k), sigmaX=0, sigmaY=0)


def add_motion_blur(img, level=1):
    blur_degree = level * 2
    angle = 45
    M = cv2.getRotationMatrix2D((blur_degree / 2, blur_degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(blur_degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M,
                                        (blur_degree, blur_degree))

    motion_blur_kernel = motion_blur_kernel / blur_degree
    processed_img = cv2.filter2D(img, -1, motion_blur_kernel)
    cv2.normalize(processed_img, processed_img, 0.0, 1.0, cv2.NORM_MINMAX)
    # processed_img = np.array(processed_img, dtype=np.uint8)
    return processed_img


def add_zoom_blur(img, level):
    return img


def add_snow(img, level):
    return img


def add_frost(img, level):
    return img


def add_fog(img, level):
    return img


def do_corruptions(corrupt_type, img, level):
    img = np.array(img)
    # print(img)
    # mean = np.array([0.5, 0.5, 0.5])
    # std = np.array([0.5, 0.5, 0.5])
    # for img in dataset:
    # img = (img - mean) / std
    img = img / 255.0
    if corrupt_type == 'gaussian_noise':
        img = add_gaussian_noise(img, level)
    elif corrupt_type == 'shot_noise':
        img = add_shot_noise(img, level)
    elif corrupt_type == 'impulse_noise':
        img = add_impulse_noise(img, level)
    elif corrupt_type == 'defocus_blur':
        img = add_defocus_blur(img, level)
    elif corrupt_type == 'glass_blur':
        img = add_glass_blur(img, level)
    elif corrupt_type == 'motion_blur':
        img = add_motion_blur(img, level)
    elif corrupt_type == 'zoom_blur':
        img = add_zoom_blur(img, level)
    elif corrupt_type == 'snow':
        img = add_snow(img, level)
    elif corrupt_type == 'frost':
        img = add_frost(img, level)
    elif corrupt_type == 'fog':
        img = add_fog(img, level)
    img *= 255.0

    # return torch.fromarray(new_dataset)
    return img.astype(np.uint8)
    # return Image.fromarray(img.astype(np.uint8))


class Corruption(object):

    def __init__(self, op):
        self.op = op

    def __call__(self, pic, level=1):
        return do_corruptions(self.op, pic, level)