# Get the threhold in whole sequence 199 images
import os
import cv2
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

L_node = 256
W_node = 256
L_times = 8  # the number of boxes in L_node length
W_times = 16  # the number of boxes in W_node length
Channel = 1
Starting_point = 1
Num_video = 36
Num_image_per = 199

Result_dir = 'S:/UCSD_ped2/Test256/Unet_Mosaic_test/'
Label_dir = 'S:/UCSD_ped2/Test256/training/'
Output_dir = 'S:/UCSD_ped2/Test256/Unet_Mosaic_test_mask/'


def read_and_load(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = np.reshape(img, newshape=[L_node, W_node, Channel])
    img = img
    return img


def train():
    Patch_diff = np.zeros(shape=[L_times, W_times, Num_image_per])

    Name = [i for i in os.listdir(Result_dir)]
    for num_sequence in range(Num_video):
        for num_image in range(Num_image_per):
            num = num_sequence * Num_image_per + num_image
            print('num: ', num)

            Result_name = Name[Starting_point + num - 1]
            Result_path = Result_dir + Result_name
            Label_path = Label_dir + Result_name

            Result = read_and_load(Result_path)
            Label = read_and_load(Label_path)

            Result = Result / 255.
            Label = Label / 255.
            Result = np.reshape(Result, newshape=[L_node, W_node])
            Label = np.reshape(Label, newshape=[L_node, W_node])

            L_size = int(L_node / L_times)  # The length of one box
            W_size = int(W_node / W_times)  # The width of one box

            for L in range(L_times):
                # print('L: ', L)
                for W in range(W_times):
                    # print('W: ', W)
                    L_start = L * L_size  # 0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240
                    L_end = ((L + 1) * L_size)  # 15, 31, 47, 63, 79, 95, 111, 127, 143, 159, 175, 191, 207, 223, 239, 255+1
                    W_start = W * W_size
                    W_end = ((W + 1) * W_size)

                    Result_patch = Result[L_start: L_end, W_start: W_end]  # [0, 16)
                    Label_patch = Label[L_start: L_end, W_start: W_end]
                    # Patch_diff[L, W, num_image] = np.mean(np.abs(Result_patch - Label_patch))
                    Patch_diff[L, W, num_image] = mean_squared_error(Result_patch, Label_patch)
        # Get the whole video sequence threshold
        Patch_diffs = np.reshape(Patch_diff, newshape=(1, -1))  # Patch difference vector (1, 256 * 199)
        Patch_diffs_sort = np.argsort(Patch_diffs)  # Patch difference vector sorted index (1, 256 * 199)
        index = np.int(L_times * W_times * Num_image_per * (1 - math.pow(0.5, 7)))  # Get the patch difference vector index
        print(index)
        threshold_index = Patch_diffs_sort[0, index]  # Get the threshold index
        threshold = Patch_diffs[0, threshold_index]  # Get the threshold
        print(threshold)
        # Add mask on image
        for num_image in range(Num_image_per):
            num = num_sequence * Num_image_per + num_image

            Result_name = Name[Starting_point + num - 1]
            Result_path = Result_dir + Result_name
            Label_path = Label_dir + Result_name

            Result = read_and_load(Result_path)
            Label = read_and_load(Label_path)

            # 3 Channels grayscale image
            Result_3C = np.concatenate([Result, Result, Result], -1)
            Label_3C = np.concatenate([Label, Label, Label], -1)

            L_size = int(L_node / L_times)  # The length of one box
            W_size = int(W_node / W_times)  # The width of one box

            mask = np.zeros(Result_3C.shape, dtype=np.uint8)  # Initialize mask matrix

            for L in range(L_times):
                for W in range(W_times):
                    zero_mask = np.zeros(Result_3C.shape, dtype=np.uint8)  # Initialize zero mask matrix
                    if (Patch_diff[L, W, num_image] > threshold) & (Patch_diff[L, W, num_image] > 0):
                        # Create mask
                        Y_start = L * L_size
                        Y_end = ((L + 1) * L_size) - 1
                        X_start = W * W_size
                        X_end = ((W + 1) * W_size) - 1
                        mask = mask + cv2.rectangle(zero_mask, (X_start, Y_start), (X_end, Y_end), color=(0, 0, 100), thickness=-1)
            mask = np.array(mask)
            # Add mask to real image
            alpha = 1
            beta = 1.0
            gamma = 0
            img_mask = cv2.addWeighted(Label_3C, alpha, mask, beta, gamma)
            cv2.imwrite(Output_dir + Result_name, img_mask)


def main(argv=None):
    train()


if __name__=='__main__':
    main()