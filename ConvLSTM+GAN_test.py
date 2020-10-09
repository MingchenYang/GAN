import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

Batch_size = 1
Time_steps = Num_sequence = 5
L_node = 256
W_node = 256
Channel = 1
Output_dim = 100
Learning_rate_gen = 0.0002
Learning_rate_disc = 0.0002
Beta_1 = 0.5
Beta_2 = 0.999
E = 1e-08
Training_steps = 100000
Lambda = 100

File_dir = 'S:/pix2pix optflow/Test256/training/'
Output_dir = 'S:/pix2pix optflow/Test256/CLSTM+GAN_result/'
Save_path = 'S:/pix2pix optflow/Generator256/CLSTM+GAN_save/3000Gmodel.h5'
File_name = 6981


def get_path(name):
    path = File_dir + str(name) + '.jpg'
    return path


def read_and_load(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = np.reshape(img, newshape=[1, 1, L_node, W_node, Channel])  # (1, 1, 256, 256, 1)
    img = tf.cast(img, tf.float32)
    img = img / 255.
    return img


def train():
    image_sequence = []

    for i in range(Num_sequence):
        File_num = File_name + Num_sequence
        File_path = get_path(File_num)
        image = read_and_load(File_path)
        if i == 0:
            image_sequence = image  # (1, 1, 256, 256, 1)
        else:
            image_sequence = layers.concatenate([image_sequence, image], 1)  # (1, 5, 256, 256, 1)

    model = tf.keras.models.load_model(Save_path)
    model.summary()

    output = model(image_sequence, training=False)
    output = np.reshape(output, newshape=[L_node, W_node])

    cv2.imwrite(Output_dir + str(File_name) + '.jpg', output * 255.)


def main(argv=None):
    train()


if __name__ == '__main__':
    main()
