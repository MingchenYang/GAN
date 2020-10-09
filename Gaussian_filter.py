import os
import cv2
import numpy as np

Kernel_size = 11
Sigma = 9

Input_dir = 'S:/pix2pix optflow/Generator256/training/'
Output_dir = 'S:/pix2pix optflow/Generator256/training_Gaussian/'


def read_and_load(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def train():
    for name in os.listdir(Input_dir):
        print(name)
        Input_path = Input_dir + name
        Image = read_and_load(Input_path)
        Image = cv2.GaussianBlur(Image, (Kernel_size, Kernel_size), Sigma)
        Output_path = Output_dir + name
        cv2.imwrite(Output_path, Image)


def main(argv=None):
    train()


if __name__=='__main__':
    main()