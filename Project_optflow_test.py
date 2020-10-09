import cv2
import numpy as np
import tensorflow as tf

L_node = 158
W_node = 238
Channel = 1

Test_dir = 'S:/pix2pix optflow/Test/training/2731.jpg'
Label_dir = 'S:/pix2pix optflow/Test/label/2731.jpg'
Output_test_dir = 'S:/pix2pix optflow/Test/result/2731.jpg'
Save_path = 'S:/pix2pix optflow/Train/save/5000model.h5'


def recover(img):
    max_value = np.max(img)
    img = (255. / max_value) * img
    return img


model = tf.keras.models.load_model(Save_path)
model.summary()

x = cv2.imread(Test_dir, cv2.IMREAD_GRAYSCALE)
x = tf.reshape(x, shape=[1, L_node, W_node, Channel])
x = tf.cast(x, tf.float32)
x = x / 255.
label = cv2.imread(Label_dir, cv2.IMREAD_GRAYSCALE)
label = tf.reshape(label, shape=[L_node, W_node, Channel])
label = tf.cast(label, tf.float32)
label = label / 255.
#label = recover(label)

output = model(x, training=False)
output = tf.reshape(output, shape=[L_node, W_node, Channel])
ssim_loss = tf.reduce_mean(1 - tf.image.ssim(output, label, max_val=1))
tf.print(ssim_loss)
output = output * 255.
output = np.reshape(output, newshape=[L_node, W_node])
cv2.imwrite(Output_test_dir, output)