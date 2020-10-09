#  This file is used to generate edge pictures using small motion kernels
import os
import cv2
import numpy as np
import tensorflow as tf

Input_dir = 'S:/UCSD_ped2/Test256/training/'
Output_dir = 'S:/UCSD_ped2/Test256/training_edge/'


def get_weights():
    layers_num = 24
    weights = np.zeros((3, 3, layers_num))
    for layer in range(layers_num):
        if layer < 8:
            weights[1, 1, layer] = 0.8
        elif layer < 16:
            weights[1, 1, layer] = 0.5
        elif layer < 24:
            weights[1, 1, layer] = 0.2
    k = 0
    while k < layers_num:
        for a in range(3):
            for b in range(3):
                requirement = (a == 1) and (b == 1)
                if not requirement:
                    if k < 8:
                        weights[a, b, k] = 0.2
                    elif k < 16:
                        weights[a, b, k] = 0.5
                    elif k < 24:
                        weights[a, b, k] = 0.8
                    k = k + 1

    weights = np.reshape(weights, newshape=[3, 3, 1, layers_num])
    return weights


def recover(img):
    img = (255./np.max(img)) * img
    return img


def train():
    Image_R = np.zeros([256, 256, 24])

    edge_weights = get_weights()

    Input_name = os.listdir(Input_dir)

    for i in range(len(Input_name)):
        Input_path = Input_dir + Input_name[i]
        Input_file = cv2.imread(Input_path, cv2.IMREAD_GRAYSCALE)
        Input_file = np.reshape(Input_file, newshape=[1, 256, 256, 1])
        Input_file = tf.cast(Input_file, tf.float32)

        Image = tf.nn.conv2d(Input_file, edge_weights, 1, 'SAME')  # (1, 256, 256, 24)
        Image = np.reshape(Image, newshape=[256, 256, 24])
        Input_file = np.reshape(Input_file, newshape=[256, 256])
        for j in range(24):
            Image_R[:, :, j] = np.abs(Image[:, :, j] - Input_file)
        Result = np.mean(Image_R, 2)

        Result = recover(Result)
        print(Input_name[i])
        cv2.imwrite(Output_dir + Input_name[i], Result)


def main(argv=None):
    train()


if __name__=='__main__':
    main()