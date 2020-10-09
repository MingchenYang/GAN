import os
import cv2
import numpy as np
import tensorflow as tf

L_node = 256
W_node = 256
Channel = 1

Input_dir = 'S:/pix2pix optflow/Test256/UGAN3.0_test/'
Training_dir = 'S:/pix2pix optflow/Test256/training/'
Output_dir = 'S:/pix2pix optflow/Test256/UGAN3.0_test_difference/'


def read_and_load(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img / 255.
    return img


def train():
    k = 1

    for name in os.listdir(Input_dir):
        Input_path = Input_dir + name  # Label path
        Training_path = Training_dir + name  # Training path
        print(name)

        Generate_image = read_and_load(Input_path)
        Image = read_and_load(Training_path)

        Difference = np.abs(Generate_image - Image)

        Output = Difference * 255.
        Output_path = Output_dir + name
        cv2.imwrite(Output_path, Output)

        k = k + 1


def main(argv=None):
    train()


if __name__=='__main__':
    main()