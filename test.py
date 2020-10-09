import cv2
import numpy as np
import tensorflow as tf

threshold = 0

label_dir = 'S:/pix2pix optflow/Test256/training/7060.jpg'
test_dir = 'S:/pix2pix optflow/Test256/result3/7060.jpg'
Output_dir = 'S:/pix2pix optflow/Test256/result3/7060differ.jpg'

label = cv2.imread(label_dir, cv2.IMREAD_GRAYSCALE)
label = tf.cast(label, tf.float32)
label = label / 255
test = cv2.imread(test_dir, cv2.IMREAD_GRAYSCALE)
test = tf.cast(test, tf.float32)
test = test / 255.

difference = tf.abs(test - label)
difference = np.array(difference)
difference = difference * 255.
max_val = np.max(difference)
print(max_val)
# max_threshold = max_val * threshold
# for i in range(256):
#     for j in range(256):
#         if difference[i, j] <= max_threshold:
#             difference[i, j] = 0
cv2.imwrite(Output_dir, difference)