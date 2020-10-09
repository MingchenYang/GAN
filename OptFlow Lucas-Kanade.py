import cv2
import os
import numpy as np
'''
input_dir = 'S:/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/'
output_dir = 'S:/optical flow/Lucas Kanade/'

input_file = os.listdir(input_dir)
input_file = input_file[2:]
input_file.sort(key=lambda x: int(x[-3:]))

output_file = input_file

def opticalflow(file_path, file_name, i):
    x = cv2.imwrite(file_path + file_name[i], cv2.IMREAD_GRAYSCALE)
    x_next = cv2.imwrite(file_path + file_name[i+1], cv2.IMREAD_GRAYSCALE)
    flow = cv2.calcOpticalFlowFarneback(x, x_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow
'''


def recover(img):
    img = (255./np.max(img)) * img
    return img


input_path = 'S:/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test014/001.tif'
next_frame_path = 'S:/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test014/002.tif'
output_path = 'S:/optical flow/1.jpg'

x = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
x.astype(np.float32)
x_next = cv2.imread(next_frame_path, cv2.IMREAD_GRAYSCALE)
x_next.astype(np.float32)
flow = cv2.calcOpticalFlowFarneback(x, x_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
flow = recover(flow[:, :, 0]) + recover(flow[:, :, 1])
print(flow.shape)
cv2.imwrite(output_path, flow)