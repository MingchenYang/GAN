import os
import cv2
import numpy as np

L_node = 256
W_node = 256
Channel = 1
L_size = 8
W_size = 4

Input_dir = 'S:/UCSD_ped2/Test256/training/'
Output_dir = 'S:/UCSD_ped2/Test256/training_Mosaic/'


def read_and_load(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def train():
    for Name in os.listdir(Input_dir):
        print(Name)
        Input_path = Input_dir + Name
        Image = read_and_load(Input_path)

        L_times = int(L_node / L_size)
        W_times = int(W_node / W_size)

        for L in range(L_times):
            for W in range(W_times):
                L_start = L * L_size
                L_end = (L + 1) * L_size
                W_start = W * W_size
                W_end = (W + 1) * W_size

                Image_patch = Image[L_start:L_end, W_start:W_end]
                mean = np.mean(Image_patch)
                Image[L_start:L_end, W_start:W_end] = mean

        Output_path = Output_dir + Name
        cv2.imwrite(Output_path, Image)


def main(argv=None):
    train()


if __name__=='__main__':
    main()