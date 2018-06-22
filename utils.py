import gc
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image


def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


class train_data():
    def __init__(self, filepath='./data/image_clean_pat.npy'):
        self.filepath = filepath
        assert '.npy' in filepath
        if not os.path.exists(filepath):
            print("[!] Data file not exists")
            sys.exit(1)

    def __enter__(self):
        print("[*] Loading data...")
        self.data = np.load(self.filepath)
        np.random.shuffle(self.data)
        print("[*] Load successfully...")
        return self.data

    def __exit__(self, type, value, trace):
        del self.data
        gc.collect()
        print("In __exit__()")

def find_match_file(denoise_files, noise_files):
    noise = []
    for name1 in denoise_files:
        id = name1.split('/')[-1].split('_')[0]
        for name2 in noise_files:
            if name2.find(id) >= 0:
                noise.append(name2)
    return noise

def load_data(filepath='./data/image_clean_pat.npy'):
    return train_data(filepath=filepath)

def load_conv_images(filelist, width, height):
    if not isinstance(filelist, list):
        im = Image.open(filelist)
        return np.array(im).reshape(1, im.size[1], im.size[0], 3)
    output=np.zeros((len(filelist), height, width, 3), dtype="uint8")
    index = 0
    for file in filelist:
        img = Image.open(file)
        output[index, :, :, :] = np.array(img).reshape(1, img.size[1], img.size[0], 3)
        index+=1
    return output
def load_images(filelist):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = Image.open(filelist)
        return np.array(im).reshape(1, im.size[1], im.size[0], 3)
    data = []
    for file in filelist:
        im = Image.open(file)
        data.append(np.array(im).reshape(1, im.size[1], im.size[0], 3))
    return data


def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
    im = Image.fromarray(cat_image.astype('uint8'))
    im.save(filepath, 'JPEG')


def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def tf_psnr(im1, im2):
    # assert pixel value range is 0-1
    mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))
