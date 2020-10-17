import os
import cv2
import tensorflow as tf

Input_dir = 'S:/UCSD_ped2/Test256/Unet_Multi_test/'
Label_dir = 'S:/UCSD_ped2'

Input_name = os.listdir(Input_dir)


def read_and_load(path):
    img = cv2.imread(path, 0)
    img = img / 255.
    return img


def train():
    for name in Input_name:
        Input_path = Input_dir + name
        image = read_and_load(Input_path)
        psnr = tf.image.psnr()