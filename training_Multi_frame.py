# ------
# UCSD Ped2, 119, 149, 179 frames per video, 5 consecutive frames with 1 next frame prediction
# ------
import os
import cv2
import numpy as np

File_path = 'S:/UCSD_ped2/Test256/training/'
Output_path = 'S:/UCSD_ped2/Test256/training_Multi_frame/'
Image_name = os.listdir(File_path)  # 0001.jpg
num_sequence = 5
L_node = 240
W_node = 360
Num_video = 12
# Num_image_per = [120, 150, 150, 180, 180, 150, 150, 120, 180, 180, 180, 180, 180, 150, 150, 150]
Num_image_per = [180, 180, 150, 180, 150, 180, 180, 180, 120, 150, 180, 180]
Total_video_frames = sum(Num_image_per) - Num_video


def read_and_load(file_name):
    image_path = os.path.join(File_path, file_name)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img


def pack_frames(num_record, num_name):
    for j in range(num_sequence + 1):
        name = Image_name[num_record + j]
        print(num_record + j)

        image = read_and_load(name)
        num_save = '%04d' % num_name
        Output_name = Output_path + str(num_save) + '.jpg'
        cv2.imwrite(Output_name, image)


def train():
    num_record = 0
    num_name = 1
    for num_video in range(Num_video):
        for i in range(Num_image_per[num_video] - 1):
            name = Image_name[num_record]
            if (Num_image_per[num_video] - i - 2) < num_sequence:
                print('false', i + 1, name)
                num_record += 1
                continue

            pack_frames(num_record, num_name)
            print('true', i + 1, name)
            print('record', num_record)
            num_record += 1
            num_name += 1


def main(argv=None):
    train()


if __name__ == '__main__':
    main()