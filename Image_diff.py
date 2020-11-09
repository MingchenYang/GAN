import os
import cv2
import numpy as np

Label_dir = 'S:/UCSD_ped2/Test256/training_Multi_frame/'
Result_dir = 'S:/UCSD_ped2/Test256/Unet_Multi_test/'
Output_dir = 'S:/UCSD_ped2/Test256/Unet_Multi_test_diff/'

# Label_dir = 'S:/UCSD_ped2/Test256/label/'
# Result_dir = 'S:/UCSD_ped2/Test256/Unet_Reverse_test/'
# Output_dir = 'S:/UCSD_ped2/Test256/Unet_Reverse_test_diff/'


def read_and_load(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img / 255.
    return img


def train():
    for name in os.listdir(Result_dir):
        print(name)

        Result_path = Result_dir + name
        result = read_and_load(Result_path)

        Label_path = Label_dir + name
        label = read_and_load(Label_path)

        diff = np.abs(result - label)

        Output_path = Output_dir + name
        cv2.imwrite(Output_path, diff * 255.)


def main(argv=None):
    train()


if __name__=='__main__':
    main()