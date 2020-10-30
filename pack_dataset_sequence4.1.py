# ------
# UCSD Ped2, 119, 149, 179 frames per video, 2 consecutive frames with 1 next frame prediction
# ------
import os
import cv2
import numpy as np

Image_path = 'S:/UCSD_ped2/Train256/training/'
Label_path = 'S:/UCSD_ped2/Train256/label_dis_removal/'
Output_path = 'S:/UCSD_ped2/Train256/training_sequence1/'
Image_name = os.listdir(Image_path)  # 0001.jpg
Label_name = os.listdir(Label_path)
num_sequence = 2
L_node = 256
W_node = 256
Channel = 1
Num_video = 12
# Num_image_per = [120, 150, 150, 180, 180, 150, 150, 120, 180, 180, 180, 180, 180, 150, 150, 150]
Num_image_per = [180, 180, 150, 180, 150, 180, 180, 180, 120, 150, 180, 180]
Total_video_frames = sum(Num_image_per) - Num_video


def read_and_load_grey(file_name):
    image_path = os.path.join(Image_path, file_name)
    img = cv2.imread(image_path, 0)
    return img


def read_and_load_color(file_name):
    image_path = os.path.join(Label_path, file_name)
    img = cv2.imread(image_path, 1)
    return img


def pack_frames(num_record, num_name):
    num1 = 0  # Compute when number becomes 3
    Sequence_image = np.array([])
    Sequence_label = np.array([])

    for j in range(num_sequence + 1):
        num1 = num1 + 1
        # print(num1)  # Compute num1 from 1 to 6

        image_name = Image_name[num_record + j]
        print(num_record + j)
        # print(name)  # 0001.jpg
        # Read and Load the image
        image = read_and_load_grey(image_name)

        label_name = Label_name[num_record + j]
        label = read_and_load_color(label_name)

        if num1 <= num_sequence:
            Sequence_image = np.append(Sequence_image, image)
            # print(Sequence_image.shape)  # (65536,) (131072.) (196608,) (262144,) (327680,)
        else:
            label_l1 = label[:, :, 0]
            label_l2 = label[:, :, 1]
            label_l3 = label[:, :, 2]
            Sequence_label = np.append(Sequence_label, label_l1)
            Sequence_label = np.append(Sequence_label, label_l2)
            Sequence_label = np.append(Sequence_label, label_l3)
            # print(Sequence_label.shape)  # (65536,)
            # Sequence_image = np.reshape(Sequence_image, newshape=[num_sequence, L_node * W_node])

            num_save = '%04d' % num_name
            # print(num_name)
            Output_name = Output_path + str(num_save) + '.npz'
            # Output_name = Output_path + image_name[i + num_sequence][0: 4]
            np.savez(Output_name, image=Sequence_image, label=Sequence_label)

            num1 = 0
            Sequence_image = np.array([])
            Sequence_label = np.array([])


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
'''          
    for i in range(len(Image_name)):
        num2 = num2 + 1
        # print('num2: ', num2)
        name = Image_name[i]

        # Update num2 when num2 is equal to 200
        if num2 % Num_image_per == 0:
            print('false:', num2)
            print('false:', name)
            num2 = 0
            continue

        if ((num2 % 200) + num_sequence) > Num_image_per:
            print('false:', num2)
            print('false:', name)
            continue

        print('true: ', num2)
        print('true: ', name)  # 0001.jpg
'''


def main(argv=None):
    train()


if __name__ == '__main__':
    main()