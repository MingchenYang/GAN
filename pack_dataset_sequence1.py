import os
import cv2
import numpy as np

File_path = 'S:/pix2pix optflow/Generator256/training_edge/'
Label_path = 'S:/pix2pix optflow/Generator256/training/'  # Restore normal grayscale image which is the label in ConvLSTM in GAN
Output_path = 'S:/pix2pix optflow/Generator256/sequence_edge_label/'
image_name = os.listdir(File_path)  # 0001.jpg
Sequence_image = np.array([])
Sequence_label = np.array([])
Sequence_edge_label = np.array([])  # This is the label of edge image in sequence
num_sequence = 5
num1 = 0  # Compute when number becomes 6
num2 = 0  # Compute when number becomes 200
num3 = 0  # Added when file is saved
L_node = 256
W_node = 256


def read_and_load(file_name, index):
    if index == 0:
        image_path = os.path.join(File_path, file_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image_path = os.path.join(Label_path, file_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img


for i in range(len(image_name)):
    num2 = num2 + 1
    name = image_name[i]
    # image_path = os.path.join(File_path, name)  # S:/pix2pix optflow/Generator256/training/0001.jpg
    # print(image_path)  # S:/pix2pix optflow/Generator256/training/0001.jpg
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Show the read image
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    if (num2 + (num_sequence - 1)) % 200 <= (num_sequence - 1):
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
        image = read_and_load(name, 0)  # 0 means choose the File path

        if num1 <= num_sequence:
            Sequence_image = np.append(Sequence_image, image)
            print(Sequence_image.shape)  # (65536,) (131072.) (196608,) (262144,) (327680,)
        else:
            Sequence_label = np.append(Sequence_label, image)
            print(Sequence_label.shape)  # (65536,)
            # Sequence_image = np.reshape(Sequence_image, newshape=[num_sequence, L_node * W_node])

            edge_label = read_and_load(name, 1)  # 1 means choose the edge Label path
            Sequence_edge_label = np.append(Sequence_edge_label, edge_label)

            num3 = num3 + 1
            num_save = '%04d' % num3
            Output_name = Output_path + str(num_save) + '.npz'
            np.savez(Output_name, image=Sequence_image, label=Sequence_label, image_label=Sequence_edge_label)  # image_label presents the orginal image of edge image

            num1 = 0
            Sequence_image = np.array([])
            Sequence_label = np.array([])
            Sequence_edge_label = np.array([])