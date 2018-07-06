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
    def __init__(self, filepath='./data/image_clean_pat.npy', rand=True):
        self.filepath = filepath
        self.rand = rand
        assert '.npy' in filepath
        if not os.path.exists(filepath):
            print("[!] Data file not exists")
            sys.exit(1)

    def __enter__(self):
        print("[*] Loading data...")
        self.data = np.load(self.filepath)
        if self.rand:
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

def load_data(filepath='./data/image_clean_pat.npy', rand=True):
    return train_data(filepath=filepath, rand=rand)


def load_images_patches(filelist, patch_size = 40, stride = 40, batch_size=128):
    # only support file list is list
    if not isinstance(filelist, list):
        return None
    for file in filelist:
        img = Image.open(file)  # convert RGB to gray
        im_h, im_w = img.size
        count = (im_h - patch_size) * (im_w - patch_size) / (stride * stride)
    if count % batch_size != 0:
        numPatches = int((count / batch_size + 1) * batch_size)
    else:
        numPatches = int(count + 0.5)
    inputs = np.zeros((numPatches, patch_size, patch_size, 3), dtype="uint8")
    count = 0
    # generate patches
    for file in filelist:
        img = Image.open(file)
        img_s = np.reshape(np.array(img, dtype="uint8"), (img.size[1], img.size[0], 3))  # extend one dimension
        
        im_h, im_w, _ = img_s.shape
        for x in range(0, im_h - patch_size, stride):
            for y in range(0, im_w - patch_size, stride):
                inputs[count, :, :, :] = img_s[x:x + patch_size, y:y + patch_size, :]
                count += 1
    # pad the batch
    if batch_size == None:
        return inputs
    if count < numPatches:
        to_pad = numPatches - count
        inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]
    return inputs

def load_image_patches(file, patch_size = 40):
    img = Image.open(file)  # convert RGB to gray
    im_w, im_h = img.size
    width_count = int((im_w)/patch_size + 0.5)
    height_count = int((im_h)/patch_size + 0.5)
    count = width_count * height_count
    inputs = np.zeros((count, patch_size, patch_size, 3), dtype="uint8")
    count = 0
    # generate patches
    img = Image.open(file)
    img_s = np.reshape(np.array(img, dtype="uint8"), (img.size[1], img.size[0], 3))  # extend one dimension
    print (img_s.shape)
    im_h, im_w, _ = img_s.shape
    for x in range(0, im_h, patch_size):
        for y in range(0, im_w, patch_size):
            inputs[count, :, :, :] = img_s[x:x + patch_size, y:y + patch_size, :]
            count += 1
    return inputs, width_count, height_count

def save_patches_to_image(patches, patch_size, width, height, filepath):
    outputimage = np.zeros((height * patch_size, width * patch_size, 3), dtype="uint8")
    for x in range(0, height):
        for y in range(0, width):
            count = x * width + y
            x_start = x * patch_size
            y_start = y * patch_size
            outputimage[x_start:x_start + patch_size, y_start:y_start + patch_size, :] = patches[count]
    # for i in range(0, patches.shape[0]):
    #     save_images("%s/%d.jpg" % (filepath, i), patches[i:i+1])
    print (outputimage.shape)
    save_images(filepath, outputimage)

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
