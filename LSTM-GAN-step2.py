import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

Batch_size = 5
Num_sequence = 5
L_node = 256
W_node = 256
Channel = 1
Time_steps = Num_sequence
Output_dim = 100
Training_steps = 100000
Learning_rate = 0.01
Beta_1 = 0.5
Beta_2 = 0.999
E = 1e-08

File_path = 'S:/pix2pix optflow/Generator256/sequence/'
Save_path = 'S:/pix2pix optflow/Generator256/save/11000Gmodel.h5'
Output_dir = 'S:/pix2pix optflow/Generator256/result_step2/'


def read_and_load(name):
    name = '%04d' % name  # 0001
    File_name = File_path + str(name) + '.npz'  # S:/pix2pix optflow/Generator256/sequence/0001.npz

    Sequence_image = np.load(File_name)['image']  # Load image npz files
    Sequence_image = tf.reshape(Sequence_image, shape=[1, Num_sequence, L_node * W_node])  # (5, 256*256)
    Sequence_image = tf.cast(Sequence_image, tf.float32)
    Sequence_image = Sequence_image / 255.
    # print(Sequence_image.shape)  # (5, 65536)

    Sequence_label = np.load(File_name)['label']  # Load label npz files
    Sequence_label = tf.reshape(Sequence_label, shape=[1, L_node * W_node])  # (1, 256*256)
    Sequence_label = tf.cast(Sequence_label, tf.float32)
    Sequence_label = Sequence_label / 255.
    # print(Sequence_label.shape)  # (1, 65536)

    return Sequence_image, Sequence_label


def build_model():
    input = tf.keras.Input(shape=[Time_steps, L_node * W_node])

    l1 = layers.Conv1D(filters=1024, kernel_size=3, strides=1, padding='same')(input)
    # l2 = layers.Conv1D(1000, 3, 1, 'same')(l1)
    # print(input.shape, l2.shape)  # (None, 5, 65536) (None, 5, 1000)

    l3 = layers.LSTM(512, return_sequences=True)(l1)
    # print(l3.shape)  # (None, 1000)
    l4 = layers.LSTM(256, return_sequences=True)(l3)
    l5 = layers.LSTM(128, return_sequences=True)(l4)
    l6 = layers.LSTM(Output_dim)(l5)

    model = tf.keras.Model(input, l6)
    return model


def train():
    # Load generator model
    generator = tf.keras.models.load_model(Save_path)

    Model = build_model()
    Model.build(input_shape=[Batch_size, Num_sequence, L_node * W_node])
    Model.summary()

    generator.summary()

    # optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate, beta_1=Beta_1, beta_2=Beta_2, epsilon=E)
    optimizer = tf.keras.optimizers.SGD(learning_rate=Learning_rate)

    for steps in range(Training_steps):
        # Generate Batch_size random integer lists
        rand = random.sample(range(1, 6630), Batch_size)
        # print(rand)  # [6404, 1552, 622, 5389, 4499]

        for i in range(Batch_size):
            image_sequence, label = read_and_load(rand[i])
            # print(image_sequence.shape, label.shape)  # (5, 65536) (1, 65536)

            if i == 0:
                image_batch = image_sequence
                label_batch = label
            else:
                image_batch = layers.concatenate([image_batch, image_sequence], 0)
                label_batch = layers.concatenate([label_batch, label], 0)
        # print(image_batch.shape, label_batch.shape)  # (5, 5, 65536) (5, 65536) (Batch_size, Time_steps, 65536)

        with tf.GradientTape() as Tape:
            output = Model(image_batch, training=True)
            generator_output = generator(output, training=False)
            # print(generator_output.shape)  # (5, 256, 256, 1)
            generator_output = tf.reshape(generator_output, shape=[Batch_size, L_node, W_node, 1])
            # print(generator_output.shape)  # (5, 65536)
            # loss = tf.reduce_mean(tf.keras.losses.MSE(label_batch, generator_output))
            label_batch = tf.reshape(label_batch, shape=[Batch_size, L_node, W_node, 1])
            loss = tf.reduce_mean(1 - tf.image.ssim(generator_output, label_batch, max_val=1.0))
        gradients = Tape.gradient(loss, Model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, Model.trainable_variables))

        if steps % 10 == 0:
            print("Step:{} Loss:{:.4f}".format(steps, loss))
            # print(output[0])
            generator_output = np.reshape(generator_output, newshape=[Batch_size, L_node, W_node])  # (5, 256, 256)
            # print(generator_output[0].shape)  # (256, 256)
            label_batch = np.reshape(label_batch, newshape=[Batch_size, L_node, W_node])  # (5, 256, 256)
            # print(label_batch[0].shape)  # (256, 256)
            cv2.imwrite(Output_dir + 'Label' + str(steps) + '.jpg', label_batch[0] * 255.)
            cv2.imwrite(Output_dir + str(steps) + '.jpg', generator_output[0] * 255.)


def main():
    train()


if __name__ == '__main__':
    main()