import os
import cv2
import numpy as np
import scipy.io as scio

L_node = 256
W_node = 256
Num_video = 12  # UCSD Ped2 contains 12 videos
Num_video_per = [180, 180, 150, 180, 150, 180, 180, 180, 120, 150, 180, 180]  # UCSD ped2 each video contains 180/150/120 frames
# Num_video_per = [175, 175, 145, 175, 145, 175, 175, 175, 115, 145, 175, 175]
Total_video_frames = sum(Num_video_per) - Num_video  # UCSD ped2 contains 1998 frames

Input_dir = 'S:/UCSD_ped2/Test256/Unet_Mosaic_dis_test_diff_removal/'
# Input_dir = 'S:/UCSD_ped2/Test256/Unet_Reverse_test_diff/'
Label_path = 'S:/UCSD_ped2/Test256/Ped2_label.mat'
Output_dir = 'S:/UCSD_ped2/Test256/Unet_Mosaic_test_diff_mask/'
Input_name = os.listdir(Input_dir)


def read_and_load(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img / 255.
    return img


def abnormal_detect(img, pixel_threshold, num_threshold):
    num = 0

    for L in range(L_node):
        for W in range(W_node):
            if img[L, W] >= pixel_threshold:
                num = num + 1
                #  print('This image is abnormal:', name)

            if num >= num_threshold:
                return True
    return False


def mask(normalize_value, pixel_threhold):
    num = 0
    num_record = 0
    for i in range(Num_video):
        for j in range(Num_video_per[i]):
            num = num_record + j
            name = Input_name[num]
            input_path = Input_dir + name
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            img_reshape = np.reshape(img, newshape=[256, 256, 1])
            img_3C = np.concatenate([img_reshape, img_reshape, img_reshape], -1)
            img = img / 255.
            img_norm = img / normalize_value[i]

            add_mask = np.zeros(shape=img_3C.shape, dtype=np.uint8)

            for L in range(int(L_node*0.3), L_node):
                for W in range(W_node):
                    zero_mask = np.zeros(img_3C.shape, dtype=np.uint8)  # Initialize zero mask matrix
                    if img_norm[L, W] >= pixel_threhold:
                        add_mask = add_mask + cv2.circle(img=zero_mask, center=(W, L), radius=1, color=(0, 0, 255), thickness=-1)
            add_mask = np.array(add_mask)
            # Add mask to real image
            alpha = 1
            beta = 1.0
            gamma = 0
            img_mask = cv2.addWeighted(img_3C, alpha, add_mask, beta, gamma)
            cv2.imwrite(Output_dir + name, img_mask)

        num_record = num + 1


def train(pixel_threshold, num_threshold, TorF):
    num = 0
    num_record = 0

    # Get label from mat file
    label = scio.loadmat(Label_path)['label']

    # Initialize normalization value matrix
    normalize_value = np.zeros(shape=[Num_video, 1])
    # Initialize abnormal detection
    detect = np.zeros(shape=[Total_video_frames, 1])
    # Compute normalization value and detect abnormality
    for num_sequence in range(Num_video):
        # Initialize maximum value in each video
        max_value = np.zeros(shape=[Num_video_per[num_sequence], 1])

        # Compute the maximum value in each video
        for num_image in range(Num_video_per[num_sequence] - 1):
            num = num_record + num_image
            name = Input_name[num]
            # print(name)

            Image_path = Input_dir + name
            image = read_and_load(Image_path)

            max_value[num_image] = np.max(image)

        # Get the maximum value in each video
        normalize_value[num_sequence] = np.max(max_value)

        # Normalize every frame in each video
        for num_image in range(Num_video_per[num_sequence] - 1):
            num = num_record + num_image
            name = Input_name[num]

            Image_path = Input_dir + name
            image = read_and_load(Image_path)

            image = image / normalize_value[num_sequence]

            # Detect whether this frame has pixel intensity over threshold
            detect[num] = abnormal_detect(image, pixel_threshold, num_threshold)

            #if detect[num]:
            #    print(name, ':', 1)
            #else:
            #    print(name, ':', 0)

        num_record = num + 1

    for i in range(Total_video_frames):
        if (i > 0) and (i < (Total_video_frames - 1)):
            if (detect[i - 1] == 1) and (detect[i] == 0) and (detect[i + 1] == 1):
                detect[i] = 1
            elif (detect[i - 1] == 0) and (detect[i] == 1) and (detect[i + 1] == 0):
                detect[i] = 0
    #for i in range(Total_video_frames):
    #    print(detect[i, 0])

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
    print('When pixel threshold = {:.2f}, num threshold = {}, TPR = {:.4f}, FPR = {:.4f}, ACC = {:.4f}'.format(pixel_threshold, num_threshold, TPR, FPR, ACC))

    if TorF:
        mask(normalize_value, pixel_threshold)

    return TPR, FPR, ACC


def main(argv=None):

    for i in np.arange(0.01, 1.0, 0.01):
        for j in range(301, 302):
            TPR, FPR, ACC = train(i, j, 0)
            if ACC < 0.5 or TPR < 0.75:
                break

    # train(0.6, 1, 0)


if __name__ == '__main__':
    main()
