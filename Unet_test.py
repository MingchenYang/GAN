import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

L_node = 256
W_node = 256
Channel = 1

Test_dir = 'S:/pix2pix optflow/Test256/training/6975.jpg'
Label_dir = 'S:/pix2pix optflow/Test256/label/6975.jpg'
Label_dir_ab = 'S:/pix2pix optflow/Test/result/2731.jpg'
Output_test_dir = 'S:/pix2pix optflow/Test256/result3/6975.jpg'
Save_path = 'S:/pix2pix optflow/Train256/save3/12000Gmodel.h5'


def recover(img):
    max_value = np.max(img)
    img = (255. / max_value) * img
    return img


model = tf.keras.models.load_model(Save_path)
model.summary()

# x is the frame
x = cv2.imread(Test_dir, cv2.IMREAD_GRAYSCALE)
x = tf.reshape(x, shape=[1, L_node, W_node, Channel])
x = tf.cast(x, tf.float32)
x = x / 255.
# label is the optical flow image
label = cv2.imread(Label_dir, cv2.IMREAD_GRAYSCALE)
label = tf.reshape(label, shape=[1, L_node, W_node, Channel])
label = tf.cast(label, tf.float32)
label = label / 255.
# label_ab is the generated optical flow image
label_ab = cv2.imread(Label_dir, cv2.IMREAD_GRAYSCALE)
label_ab = cv2.resize(label_ab, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
label_ab = tf.reshape(label_ab, shape=[1, L_node, W_node, Channel])
label_ab = tf.cast(label_ab, tf.float32)
label_ab = label_ab / 255.

#label = recover(label)

input = label_ab
# input = layers.concatenate([x, label_ab], 3)
output = model(input, training=False)
output = tf.reshape(output, shape=[L_node, W_node, Channel])
l1_loss = tf.reduce_mean(tf.abs(output - x))
tf.print(l1_loss)
# ssim_loss = tf.reduce_mean(1 - tf.image.ssim(output, x, max_val=1))
#tf.print(ssim_loss)
output = output * 255.
output = np.reshape(output, newshape=[L_node, W_node])
# output = cv2.resize(output, dsize=(238, 158), interpolation=cv2.INTER_AREA)
cv2.imwrite(Output_test_dir, output)