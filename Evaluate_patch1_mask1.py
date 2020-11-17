# UCSD ped2 dataset
# Add mask on every image in this dataset, all of abnormal patches will be masked on the abnormal images
import os
import cv2
import numpy as np
import scipy.io as scio

L_node = 256
W_node = 256
L_times = 16  # the number of boxes in L_node length
W_times = 32  # the number of boxes in W_node length
Num_video = 12
Num_video_per = [180, 180, 150, 180, 150, 180, 180, 180, 120, 150, 180, 180]
Total_video_frames = sum(Num_video_per) - Num_video
# Mask Parameters
alpha = 1
beta = 1.0
gamma = 0

Input_dir = 'S:/UCSD_ped2/Test256/Unet_Mosaic_Reverse_com_test_diff/'
Back_dir = 'S:/UCSD_ped2/Test256/training/'
Label_path = 'S:/UCSD_ped2/Test256/Ped2_label.mat'
Output_dir = 'S:/UCSD_ped2/Test256/Unet_Mosaic_Reverse_com_test_mask/'

Input_name = os.listdir(Input_dir)
Back_name = os.listdir(Back_dir)
detect = np.zeros(shape=[Total_video_frames, 1])


def read_and_load(path):
    img = cv2.imread(path, 0)
    img = img / 255.
    return img


def read_and_load_3C(path):
    img = cv2.imread(path, 1)
    return img


def max_value(num_sequence, num_record):
    max_val = 0
    num = num_record

    for num_image in range(Num_video_per[num_sequence] - 1):
        name = Input_name[num]
        num += 1

        Image_path = Input_dir + name
        image = read_and_load(Image_path)

        max_val = max(np.max(image), max_val)

    return max_val


def get_mask(back_3C, y_start, x_start, y_end, x_end):
    zero_mask = np.zeros_like(back_3C, dtype=np.uint8)
    mask = cv2.rectangle(zero_mask, (x_start, y_start), (x_end, y_end), color=(0, 0, 100), thickness=-1)
    return mask


def abnormal_detect(img, back_3C, patch_threshold, num_threshold, name):
    num = 0
    detect_status = 0
    L_size = int(L_node / L_times)  # the length of one box in L
    W_size = int(W_node / W_times)  # the length of one box in W

    mask = np.zeros_like(back_3C, dtype=np.uint8)

    for L in range(0, L_node - L_size):
        for W in range(0, W_node - W_size):
            L_start = L
            L_end = L + L_size
            W_start = W
            W_end = W + W_size

            Input_patch = img[L_start: L_end, W_start: W_end]
            Input_patch_avg = np.mean(Input_patch)

            if Input_patch_avg > patch_threshold:
                mask = mask + get_mask(back_3C, L_start, W_start, L_end, W_end)
                num += 1

            if num >= num_threshold:
                detect_status = 1

    if detect_status:
        mask = np.array(mask)
        img_mask = cv2.addWeighted(back_3C, alpha, mask, beta, gamma)
        cv2.imwrite(Output_dir + name[0: 4] + '-' + str(detect_status) + '.jpg', img_mask)
    else:
        cv2.imwrite(Output_dir + name[0: 4] + '-' + str(detect_status) + '.jpg', back_3C)

    return detect_status


def normalize_and_detect(num_sequence, num_record, max_val, patch_threshold, num_threshold):
    num = num_record

    for num_image in range(Num_video_per[num_sequence] - 1):
        input_name = Input_name[num]
        back_name = Back_name[num]

        Image_path = Input_dir + input_name
        image = read_and_load(Image_path)

        back_path = Back_dir + back_name
        back_3C = read_and_load_3C(back_path)

        image = image / max_val
        detect[num] = abnormal_detect(image, back_3C, patch_threshold, num_threshold, back_name)

        num += 1
    return num


def train(patch_threshold, num_threshold):
    num_record = 0

    # Get label from mat file
    label = scio.loadmat(Label_path)['label']

    for num_sequence in range(Num_video):
        max_val = max_value(num_sequence, num_record)
        num_record = normalize_and_detect(num_sequence, num_record, max_val, patch_threshold, num_threshold)
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
    TPR, FPR, ACC = train(0.081, 1)


if __name__ == '__main__':
    main()
