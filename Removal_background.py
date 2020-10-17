import os
import cv2

L_node = 240
W_node = 360

Input_dir = 'S:/UCSD_ped2/Test/label_dis/'
Back_dir = 'S:/UCSD_ped2/Test/training_removal/'
Output_dir = 'S:/UCSD_ped2/Test/label_dis_removal/'


def train():
    Input_name = os.listdir(Input_dir)
    Back_name = os.listdir(Back_dir)

    for i in range(len(Input_name)):
        input_name = Input_name[i]
        back_name = Back_name[i]
        Input_path = Input_dir + input_name
        Back_path = Back_dir + back_name

        Input_image = cv2.imread(Input_path, 1)
        Back_image = cv2.imread(Back_path, 0)

        for l in range(L_node):
            for w in range(W_node):
                if Back_image[l, w] == 0:
                    Input_image[l, w, :] = 0

        print(Input_path)
        cv2.imwrite(Output_dir + input_name, Input_image)


def main(argv=None):
    train()


if __name__=='__main__':
    main()