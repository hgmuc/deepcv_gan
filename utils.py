# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/utils.py
#   + License: MIT
from __future__ import division
import math
import random
import scipy.misc
import numpy as np
import os
from skimage.transform import resize

def get_image(image_path, image_size, is_crop=True):
    return transform(imread(image_path), image_size, is_crop)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def imsave(images, size, path):
    img = merge(images, size)
    if not os.path.exists(path):
        os.mkdir(path)
    return scipy.misc.imsave(path, (255*img).astype(np.uint8))

def transform(image, npx=64, is_crop=True):
    return np.array(image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.
    
def get_aligned_image(image_path, image_size=(160,160), is_crop=True):
    img = imread(image_path)
    return crop_and_resize(img, image_size, crop=is_crop)

def crop_and_resize(image, image_shape, resize_height=64, resize_width=64, crop=True):
    crop_height, crop_width = image_shape

    if crop:
        x = image.copy()
        h, w = x.shape[:2]
        j = int(round((h - crop_height)/2.))
        i = int(round((w - crop_width)/2.))
        cropped_image = resize(x[j:j+crop_height, i:i+crop_width], [resize_height, resize_width])        
    else:
        cropped_image = resize(image, [resize_height, resize_width])
        
    if np.max(cropped_image <= 1):
        return np.array(cropped_image)/0.5 - 1.
    else:   
        return np.array(cropped_image)/127.5 - 1.

def print_output(f, msg):
    print(msg)
    f.write(msg + str("\n"))
