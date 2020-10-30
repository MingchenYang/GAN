# Based on Evaluate_patch3.1.py
# UCSD ped2 dataset
# No pixel by pixel, no box by box, box next box
# Use SSIM metric
import os
import cv2
import numpy as np
import scipy.io as scio
import tensorflow as tf

L_node = 256
W_node = 256
L_times = 8  # the number of boxes in L_node length
W_times = 8  # the number of boxes in W_node length
Num_video = 12
Num_video_per = [180, 180, 150, 180, 150, 180, 180, 180, 120, 150, 180, 180]
Total_video_frames = sum(Num_video_per) - Num_video

Input_dir = 'S:/UCSD_ped2/Test256/Unet_Mosaic_dis_test/'
Train_dir = 'S:/UCSD_ped2/Test256/training/'
Label_path = 'S:/UCSD_ped2/Test256/Ped2_label.mat'
Output_dir = 'S:/UCSD_ped2/Test256/Unet_Mosaic_est_diff_mask/'

Input_name = os.listdir(Input_dir)
Train_name = os.listdir(Train_dir)
detect = np.zeros(shape=[Total_video_frames, 1])


def reshape_img(img):
    shape = img.shape
    img = tf.reshape(img, shape=[shape[0], shape[1], 1])
    return img


def read_and_load(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img / 255.
    return img


def max_value_per(img1, img2):
    max_val = 0

    L_size = int(L_node / L_times)  # the length of one box in L
    W_size = int(W_node / W_times)  # the length of one box in W

    for L in range(0, L_times):
        for W in range(0, W_times):
            L_start = L * L_size
            L_end = (L + 1) * L_size
            W_start = W * W_size
            W_end = (W + 1) * W_size

            img1_patch = reshape_img(img1[L_start: L_end, W_start: W_end])
            img2_patch = reshape_img(img2[L_start: L_end, W_start: W_end])

            ssim_val = 1 - tf.image.ssim(img1_patch, img2_patch, max_val=1)
            max_val = max(ssim_val, max_val)
    return max_val


def max_value(num_sequence, num_record):
    max_val = 0
    num = num_record

    for num_image in range(Num_video_per[num_sequence] - 1):
        input_name = Input_name[num]
        train_name = Train_name[num]
        num += 1

        Image_path = Input_dir + input_name
        image = read_and_load(Image_path)

        Train_path = Train_dir + train_name
        train = read_and_load(Train_path)

        max_val = max(max_value_per(image, train), max_val)

    return max_val, num


def max_val_col():
    num_record = 0
    max_val = np.zeros(shape=[Num_video, 1])

    for num_sequence in range(Num_video):
        max_val[num_sequence, 0], num_record = max_value(num_sequence, num_record)
        print(num_sequence, max_val[num_sequence, 0])

    return max_val


def abnormal_detect(img1, img2, max_val, patch_threshold, num_threshold):
    num = 0
    L_size = int(L_node / L_times)  # the length of one box in L
    W_size = int(W_node / W_times)  # the length of one box in W

    for L in range(0, L_times):
        for W in range(0, W_times):
            L_start = L * L_size
            L_end = (L + 1) * L_size
            W_start = W * W_size
            W_end = (W + 1) * W_size

            img1_patch = reshape_img(img1[L_start: L_end, W_start: W_end])
            img2_patch = reshape_img(img2[L_start: L_end, W_start: W_end])

            ssim_value = (1 - tf.image.ssim(img1_patch, img2_patch, max_val=1)) / max_val

            if ssim_value > patch_threshold:
                num += 1

            if num >= num_threshold:
                return True
    return False


def normalize_and_detect(num_sequence, num_record, max_val, patch_threshold, num_threshold):
    num = num_record

    for num_image in range(Num_video_per[num_sequence] - 1):
        image_name = Input_name[num]
        train_name = Train_name[num]

        Image_path = Input_dir + image_name
        image = read_and_load(Image_path)

        Train_path = Train_dir + train_name
        train = read_and_load(Train_path)

        detect[num] = abnormal_detect(image, train, max_val, patch_threshold, num_threshold)

        num += 1
    return num


def train(patch_threshold, num_threshold, max_val, TorF):
    num_record = 0

    # Get label from mat file
    label = scio.loadmat(Label_path)['label']

    for num_sequence in range(Num_video):
        num_record = normalize_and_detect(num_sequence, num_record, max_val[num_sequence], patch_threshold, num_threshold)
        # print(num_record)  # 179, 358, 507, 686, 835, 1014, 1193, 1372, 1491, 1640, 1819, 1998

    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(Total_video_frames):
        if (label[i] == 1) and (detect[i] == 1):
            TP = TP + 1
        elif (label[i] == 1) and (detect[i] == 0):
            FN = FN + 1
        elif (label[i] == 0) and (detect[i] == 1):
            FP = FP + 1
        elif (label[i] == 0) and (detect[i] == 0):
            TN = TN + 1
    print(TP, FN, FP, TN)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    ACC = (TP + TN) / Total_video_frames
    print('When pixel threshold = {:.3f}, num threshold = {}, TPR = {:.4f}, FPR = {:.4f}, ACC = {:.4f}'.format(
        patch_threshold, num_threshold, TPR, FPR, ACC))

    return TPR, FPR, ACC


def main(argv=None):
    max_val = max_val_col()

    for i in np.arange(0.11, 1.0, 0.01):
        for j in range(5, 6):
            TPR, FPR, ACC = train(i, j, max_val, 0)
            if ACC < 0.5 or TPR < 0.5 or FPR < 0.05:
                break


if __name__ == '__main__':
    main()
