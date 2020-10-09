# ------
# UCSD Ped1, 200 images per video
# ------
import os
import cv2
import numpy as np

File_path = 'S:/UCSD_ped1/Test256/training/'
Output_path = 'S:/UCSD_ped1/Test256/training_Multi/'
image_name = os.listdir(File_path)  # 0001.jpg
Sequence_image = np.array([])
Sequence_label = np.array([])
num_sequence = 5
num1 = 0  # Compute when number becomes 6
num2 = 0  # Compute when number becomes 200
num3 = 0  # Added when file is saved
L_node = 256
W_node = 256


def read_and_load(file_name):
    image_path = os.path.join(File_path, file_name)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img


for i in range(len(image_name)):
    num2 = num2 + 1
    name = image_name[i]

    if (num2 + (num_sequence - 1)) % 200 <= (num_sequence - 1):  # num2 = 396, num2 + (num_sequence - 1) = 400
        continue
    # Update num2 when num2 is equal to 200
    if num2 % 200 == 0:
        num2 = 0
    # print(name)  # 0001.jpg

    for j in range(num_sequence + 1):
        num1 = num1 + 1
        print(num1)  # Compute num1 from 1 to 6

        name = image_name[i + j]
        print(name)  # 0001.jpg
        # Read and Load the image
        image = read_and_load(name)

        if num1 <= num_sequence:
            Sequence_image = np.append(Sequence_image, image)
            print(Sequence_image.shape)  # (65536,) (131072.) (196608,) (262144,) (327680,)
        else:
            Sequence_label = np.append(Sequence_label, image)
            print(Sequence_label.shape)  # (65536,)
            # Sequence_image = np.reshape(Sequence_image, newshape=[num_sequence, L_node * W_node])

            num3 = num3 + 1
            num_save = '%04d' % num3
            Output_name = Output_path + str(num_save) + '.npz'
            np.savez(Output_name, image=Sequence_image, label=Sequence_label)

            num1 = 0
            Sequence_image = np.array([])
            Sequence_label = np.array([])