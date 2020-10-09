# This file contains the last frames per video and the remaining part contains the moving objects
import os
import cv2
import glob
import numpy as np

L_node = 240
W_node = 360
Channel = 1
Num_video = 16
Num_video_per = [120, 150, 150, 180, 180, 150, 150, 120, 180, 180, 180, 180, 180, 150, 150, 150]
Total_video_frames = sum(Num_video_per)

Input_dir = 'S:/UCSD_ped2/Train/training_full/'
Removal_dir = 'S:/UCSD_ped2/Train/training_removal_background_full/'
Output_dir = 'S:/UCSD_ped2/Train/training_removal_row_full/'

Input_name = os.listdir(Input_dir)


def motion_mask(index, num_record):
    img = np.zeros(shape=[L_node, W_node])
    num = num_record
    for i in range(Num_video_per[index]):
        name = Input_name[num]
        path = Removal_dir + name
        img += cv2.imread(path, 0)
        num += 1
    return img


def add_mask(mask, img):
    for l in range(L_node):
        for w in range(W_node):
            if mask[l, w] == 0:
                img[l, w] = 0
    return img


def train():
    num_record = 0
    for num_video in range(Num_video):
        mask = motion_mask(num_video, num_record)
        cv2.imshow('mask', mask)
        cv2.waitKey(1)
        for i in range(Num_video_per[num_video]):
            name = Input_name[num_record]
            path = Input_dir + name
            image = cv2.imread(path, 0)

            image_masked = add_mask(mask, image)
            cv2.imwrite(Output_dir + name, image_masked)

            print(num_record)
            num_record += 1


def main(argv=None):
    train()


if __name__ == '__main__':
    main()
