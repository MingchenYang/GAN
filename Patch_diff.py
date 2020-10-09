# Get the threshold through every image
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

L_node = 256
W_node = 256
L_times = 8  # the number of boxes in L_node length
W_times = 32
Channel = 1
Starting_point = 1

Result_dir = 'S:/pix2pix optflow/Test256/Unet_Gaussian_test/'
Label_dir = 'S:/pix2pix optflow/Test256/training/'
Output_dir = 'S:/pix2pix optflow/Test256/Unet_Gaussian_test_mask/'


def read_and_load(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = np.reshape(img, newshape=[L_node, W_node, Channel])
    img = img
    return img


def train():
    Result_batch = 0
    Label_batch = 0
    Patch_diff = np.zeros(shape=[L_times, W_times])

    Name = [i for i in os.listdir(Result_dir)]
    for num in range(7164):
        print('num: ', num)
        Result_name = Name[Starting_point + num - 1]
        Result_path = Result_dir + Result_name
        Label_path = Label_dir + Result_name

        Result = read_and_load(Result_path)
        Label = read_and_load(Label_path)

        # 3 Channels grayscale image
        Result_3C = np.concatenate([Result, Result, Result], -1)
        Label_3C = np.concatenate([Label, Label, Label], -1)

        if num == 0:
            Result_batch = Result
            Label_batch = Label
        else:
            Result_batch = np.concatenate([Result_batch, Result], -1)
            Label_batch = np.concatenate([Label_batch, Label], -1)
        # Result_batch: (256, 256, 199)

        Result = Result / 255.
        Label = Label / 255.
        Result = np.reshape(Result, newshape=[L_node, W_node])
        Label = np.reshape(Label, newshape=[L_node, W_node])

        L_size = int(L_node / L_times)  # The length of one box
        W_size = int(W_node / W_times)

        for L in range(L_times):
            # print('L: ', L)
            for W in range(W_times):
                # print('W: ', W)
                L_start = L * L_size  # 0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240
                L_end = ((L + 1) * L_size)  # 15, 31, 47, 63, 79, 95, 111, 127, 143, 159, 175, 191, 207, 223, 239, 255+1
                W_start = W * W_size
                W_end = ((W + 1) * W_size)

                # plt.figure()
                Result_patch = Result[L_start: L_end, W_start: W_end]  # [0, 16)
                # plt.subplot(1, 2, 1)
                # plt.imshow(Result_patch, cmap='gray')
                Label_patch = Label[L_start: L_end, W_start: W_end]
                # plt.subplot(1, 2, 2)
                # plt.imshow(Label_patch, cmap='gray')
                # plt.show()

                Patch_diff[L, W] = np.mean(np.abs(Result_patch - Label_patch))
                # print(np.mean(np.abs(Result_patch, Label_patch)))
        # print(Patch_diff)
        mask = np.zeros(Result_3C.shape, dtype=np.uint8)
        Patch_diffs = np.reshape(Patch_diff, newshape=(1, -1))  # Patch difference vector
        print(Patch_diffs)
        Patch_diffs_sort = np.argsort(Patch_diffs)  # Patch difference vector sorted index (1, 256)
        index = np.int(L_times * W_times * (1 - 0.03125))  # Get the patch difference vector index
        threshold_index = Patch_diffs_sort[0, index]  # Get the threshold index
        threshold = Patch_diffs[0, threshold_index]  # Get the threshold
        print(threshold)
        for L in range(L_times):
            for W in range(W_times):
                zero_mask = np.zeros(Result_3C.shape, dtype=np.uint8)
                if Patch_diff[L, W] > threshold:
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