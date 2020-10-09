import os
import cv2
import glob
import numpy as np
import tensorflow as tf

input_dir = 'S:/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/'
output_dir = 'S:/UCSD_ped2/test/optical/'

L_node = 240
W_node = 360
layers_num = 24
gamma_value = 1
epsilon = 0.01
lam = 0.5
layer1_residual = np.zeros([L_node, W_node, layers_num])

input_file = os.listdir(input_dir)
#input_file = input_file[2:]
# input_file.sort(key=lambda x: int(x[-3:]))

output_file = input_file


def get_weights():
    weights = np.zeros((3, 3, layers_num))
    for layer in range(layers_num):
        if layer < 8:
            weights[1, 1, layer] = 0.8
        elif layer < 16:
            weights[1, 1, layer] = 0.5
        elif layer < 24:
            weights[1, 1, layer] = 0.2
    k = 0
    while k < layers_num:
        for a in range(3):
            for b in range(3):
                requirement = (a == 1) and (b == 1)
                if not requirement:
                    if k < 8:
                        weights[a, b, k] = 0.2
                    elif k < 16:
                        weights[a, b, k] = 0.5
                    elif k < 24:
                        weights[a, b, k] = 0.8
                    k = k + 1
    return weights


def get_movements():
    movements = np.ones(layers_num)
    movements[0] = movements[2] = movements[5] = movements[7] = movements[8] = movements[10] = movements[13] = \
        movements[15] = movements[16] = movements[18] = movements[21] = movements[23] = 2
    return movements


def opticalflow(file_path, k):
    # Load and initialize the input frames
    x = cv2.imread(file_path[k], cv2.IMREAD_GRAYSCALE)
    x = np.reshape(x, newshape=[1, L_node, W_node, 1])
    x = tf.cast(x, tf.float32)
    x_next = cv2.imread(file_path[k + 1], cv2.IMREAD_GRAYSCALE)
    x_next = np.reshape(x_next, newshape=[L_node, W_node])
    x_next = tf.cast(x_next, tf.float32)
    # Get small pixel movement weights
    weights = get_weights()
    weights_reshape = np.reshape(weights, newshape=[3, 3, 1, layers_num])
    # Convolutional layer
    layer1 = tf.nn.conv2d(x, weights_reshape, 1, 'SAME')
    layer1 = np.reshape(layer1, newshape=[L_node, W_node, layers_num])
    # Get movements matrix
    movements = get_movements()
    # Calculate energy function
    for i in range(layers_num):
        layer1_residual[:, :, i] = np.sqrt((np.power((layer1[:, :, i] - x_next), 2) + epsilon)) + lam * np.sqrt(movements[i])
    # Minimum-pooling
    layer1_minpool = np.mean(layer1_residual, 2)
    # Morphology open and close algorithm
    kernel = np.ones((2, 2), np.float32)
    layer1_open = cv2.morphologyEx(layer1_minpool, cv2.MORPH_OPEN, kernel)
    layer1_close = cv2.morphologyEx(layer1_open, cv2.MORPH_CLOSE, kernel)
    # 5x5 median filter
    layer1_close = layer1_close.astype(np.float32)
    layer1_median = cv2.medianBlur(layer1_close, 5)
    flow = layer1_median
    return flow


# File name is the list of train dataset: Train001...
for file_name in input_file:
    file = glob.glob(os.path.join(input_dir + file_name, '*.tif'))
    file.sort()
    num = len(file) - 1
    # Create new path if file does not exist
    if not os.path.exists(output_dir + file_name):
        os.makedirs(output_dir + file_name)
    # Get optical flow images
    for i in range(num):
        flow = opticalflow(file, i)
        num1 = '%04d' % (i + 1)
        print(output_dir + file_name + '/' + str(num1) + '.jpg')
        cv2.imwrite(output_dir + file_name + '/' + str(num1) + '.jpg', flow)
