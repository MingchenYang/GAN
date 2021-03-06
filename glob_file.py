import os
import cv2
import glob

k = 1

Input_dir = 'S:/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/'
Output_dir = 'S:/UCSD_ped2/Test/mask_full/'

Input_path = glob.glob(Input_dir + 'Test*')
for path in Input_path:
    Input_name = glob.glob(path + '/*.bmp')
    for name in Input_name:
        print(name)
        image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        k_4d = '%04d' % k
        cv2.imwrite(Output_dir + str(k_4d) + '.png', image)
        k += 1