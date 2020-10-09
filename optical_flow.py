import tensorflow as tf
import numpy as np
import cv2
from skimage import exposure


def recover(img):
    img = (255./np.max(img)) * img
    return img

input_path = 'S:/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001/001.tif'
next_frame_path = 'S:/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001/002.tif'
#input_path = 'S:/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test014/001.tif'
#next_frame_path = 'S:/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test014/002.tif'
output_path = 'S:/optical flow/'

layers_num = 24
gamma_value = 1
epsilon = 0.01
lam = 0.5
layer1_residual = np.zeros([158, 238, layers_num])

x = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
x = np.reshape(x, newshape=[1, 158, 238, 1])
x = tf.cast(x, tf.float32)
x = x
x_next = cv2.imread(next_frame_path, cv2.IMREAD_GRAYSCALE)
x_next = np.reshape(x_next, newshape=[158, 238])
x_next = tf.cast(x_next, tf.float32)
x_next = x_next

weights = np.zeros((3, 3, layers_num))
movements = np.ones(layers_num)
movements[0]=movements[2]=movements[5]=movements[7]=movements[8]=movements[10]=movements[13]=movements[15]=movements[16]=movements[18]=movements[21]=movements[23] = 2
sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

for i in range(layers_num):
    if i < 8:
        weights[1, 1, i] = 0.8
    elif i < 16:
        weights[1, 1, i] = 0.5
    elif i < 24:
        weights[1, 1, i] = 0.2
    #elif i < 32:
    #    weights[1, 1, i] = 0

k = 0
while k < layers_num:
    for i in range(3):
        for j in range(3):
            requirement = (i == 1) and (j == 1)
            #print(requirement)
            if not requirement:
                if k < 8:
                    weights[i, j, k] = 0.2
                elif k < 16:
                    weights[i, j, k] = 0.5
                elif k < 24:
                    weights[i, j, k] = 0.8
                #elif k < 32:
                #    weights[i, j, k] = 1
                k = k + 1

#weights[1, 1, 32] = 1

#kernel = tf.random.normal(shape=[3, 3, 9])
#weights[:, :, 23:32] = kernel

weights_reshape = np.reshape(weights, newshape=[3, 3, 1, layers_num])
sobelx_reshape = np.reshape(sobelx, newshape=[3, 3, 1, 1])
sobely_reshape = np.reshape(sobely, newshape=[3, 3, 1, 1])
x_next1 = np.reshape(x_next, newshape=[1, 158, 238, 1])

#weights_reshape = np.concatenate((weights_reshape, sobelx_reshape), axis=3)
#weights_reshape = np.concatenate((weights_reshape, sobely_reshape), axis=3)

layer1 = tf.nn.conv2d(x, weights_reshape, strides=1, padding='SAME')
layer2 = tf.nn.max_pool2d(layer1, 2, 2, 'SAME')
layer2 = tf.nn.conv2d(layer2, weights_reshape, 1, 'SAME')
#x_next2 = tf.nn.max_pool2d(x_next, 2, 2, 'SAME')
#print(x_next2.shape)
layer1 = np.reshape(layer1, newshape=[158, 238, layers_num])

for i in range(layers_num):
    layer1_residual[:, :, i] = np.sqrt((np.power((layer1[:, :, i] - x_next), 2) + epsilon)) + lam * np.sqrt(movements[i])
    #layer1_residual[:, :, i] = recover(layer1_residual[:, :, i])
    cv2.imwrite(output_path + 'layer1.jpg', recover(layer1[:, :, i]))
    cv2.imwrite(output_path + 'residual.jpg', layer1_residual[:, :, i])

layer1_minpool = np.mean(layer1_residual, 2)

#for i in range(158):
#    for j in range(238):
#        if layer1_minpool[i, j] > 0.02:
#            layer1_minpool[i, j] += 0.1

cv2.imwrite(output_path + 'layer1_minpool.jpg', recover(layer1_minpool))

kernel = np.ones((2, 2), np.float32)
layer1_minpool_open = cv2.morphologyEx(layer1_minpool, cv2.MORPH_OPEN, kernel)
cv2.imwrite(output_path + 'layer1_open.jpg', recover(layer1_minpool_open))
layer1_minpool_close = cv2.morphologyEx(layer1_minpool_open, cv2.MORPH_CLOSE, kernel)
cv2.imwrite(output_path + 'layer1_close.jpg', recover(layer1_minpool_close))
layer1_minpool_close = layer1_minpool_close.astype(np.float32)
layer1_minpool = cv2.medianBlur(layer1_minpool_close, 5)

layer1_minpool = recover(layer1_minpool)

layer1_gamma = exposure.adjust_gamma(layer1_minpool, gamma_value)

cv2.imwrite(output_path + 'final.jpg', layer1_gamma)