import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

Batch_size = 5
Time_steps = Num_sequence = 5
L_node = 256
W_node = 256
Channel = 1
Output_dim = 100
Learning_rate_gen = 0.0001
Learning_rate_disc = 0.0001
Beta_1 = 0.5
Beta_2 = 0.999
E = 1e-08
training_steps = 100000

File_path = 'S:/pix2pix optflow/Generator256/sequence/'
Output_dir = 'S:/pix2pix optflow/Generator256/result/'


def read_and_load(name):
    name = '%04d' % name
    File_name = File_path + str(name) + '.npz'  # S:/pix2pix optflow/Generator256/sequence/0001.npz

    Sequence_image = np.load(File_name)['image']
    Sequence_image = np.reshape(Sequence_image, newshape=[1, Num_sequence, L_node * W_node])  # (1, 5, 256*256)
    Sequence_image = tf.cast(Sequence_image, tf.float32)
    Sequence_image = Sequence_image / 255.

    Sequence_label = np.load(File_name)['label']
    Sequence_label = np.reshape(Sequence_label, newshape=[1, L_node, W_node, Channel])  # (1, 256, 256, 1)
    Sequence_label = tf.cast(Sequence_label, tf.float32)
    Sequence_label = Sequence_label / 255.

    return Sequence_image, Sequence_label


def Generator():
    input = tf.keras.Input(shape=[Time_steps, L_node * W_node])  # (None, 5, 65536)

    l0 = layers.Conv1D(1024, 3, 1, 'same')(input)  # (None, 5, 1024)
    l0 = layers.LSTM(512, return_sequences=True)(l0)  # (None, 5, 512)
    l0 = layers.LSTM(256, return_sequences=True)(l0)  # (None, 5, 256)
    l0 = layers.LSTM(Output_dim)(l0)  # (None, 100)
    l0 = layers.Activation('tanh')(l0)

    l1 = layers.Dense(1024)(l0)
    l1 = layers.Dense(1024 * 4 * 4)(l1)
    l1 = layers.Reshape([4, 4, 1024])(l1)
    l1 = layers.Activation('relu')(l1)

    l2 = layers.Conv2DTranspose(256, 3, 2, 'same')(l1)  # (8, 8, 256)
    l2_1 = layers.Conv2D(128, 1, 1, 'same')(l2)
    l2_2 = layers.Conv2D(256, 3, 1, 'same')(l2)
    l2_3 = layers.Conv2D(64, 5, 1, 'same')(l2)
    l2 = layers.concatenate([l2_1, l2_2, l2_3], axis=-1)
    l2 = layers.Conv2D(256, 3, 1, 'same')(l2)
    l2 = layers.BatchNormalization()(l2)
    l2 = layers.Activation('relu')(l2)

    l3 = layers.Conv2DTranspose(128, 3, 2, 'same')(l2)  # (16, 16, 128)
    l3_1 = layers.Conv2D(64, 1, 1, 'same')(l3)
    l3_2 = layers.Conv2D(128, 5, 1, 'same')(l3)
    l3_3 = layers.Conv2D(32, 7, 1, 'same')(l3)
    l3 = layers.concatenate([l3_1, l3_2, l3_3], axis=-1)
    l3 = layers.Conv2D(128, 3, 1, 'same')(l3)
    l3 = layers.BatchNormalization()(l3)
    l3 = layers.Activation('relu')(l3)

    l4 = layers.Conv2DTranspose(64, 3, 2, 'same')(l3)  # (32, 32, 64)
    l4_1 = layers.Conv2D(32, 1, 1, 'same')(l4)
    l4_2 = layers.Conv2D(64, 3, 1, 'same')(l4)
    l4_3 = layers.Conv2D(16, 5, 1, 'same')(l4)
    l4 = layers.concatenate([l4_1, l4_2, l4_3], axis=-1)
    l4 = layers.Conv2D(64, 3, 1, 'same')(l4)
    l4 = layers.BatchNormalization()(l4)
    l4 = layers.Activation('relu')(l4)

    l5 = layers.Conv2DTranspose(32, 3, 2, 'same')(l4)  # (64, 64, 32)
    l5_1 = layers.Conv2D(16, 1, 1, 'same')(l5)
    l5_2 = layers.Conv2D(32, 3, 1, 'same')(l5)
    l5_3 = layers.Conv2D(8, 5, 1, 'same')(l5)
    l5 = layers.concatenate([l5_1, l5_2, l5_3], axis=-1)
    l5 = layers.Conv2D(32, 3, 1, 'same')(l5)
    l5 = layers.BatchNormalization()(l5)
    l5 = layers.Activation('relu')(l5)

    l6 = layers.Conv2DTranspose(16, 3, 2, 'same')(l5)  # (128, 128, 16)
    l6_1 = layers.Conv2D(8, 1, 1, 'same')(l6)
    l6_2 = layers.Conv2D(16, 3, 1, 'same')(l6)
    l6_3 = layers.Conv2D(4, 5, 1, 'same')(l6)
    l6 = layers.concatenate([l6_1, l6_2, l6_3], axis=-1)
    l6 = layers.Conv2D(16, 3, 1, 'same')(l6)
    l6 = layers.BatchNormalization()(l6)
    l6 = layers.Activation('relu')(l6)

    l7 = layers.Conv2DTranspose(1, 3, 2, 'same')(l6)  # (256, 256, 1)
    l7 = layers.Conv2D(1, 3, 1, 'same')(l7)
    l7 = layers.Activation('tanh')(l7)

    model = tf.keras.Model(input, l7)

    return model


def Discriminator():
    input = tf.keras.Input(shape=[L_node, W_node, Channel])

    l1 = layers.Conv2D(64, 5, 2, 'same')(input)  # (128, 128, 64)
    l1 = layers.LeakyReLU(0.2)(l1)

    l2 = layers.Conv2D(128, 5, 2, 'same')(l1)  # (64, 64, 128)
    l2 = layers.BatchNormalization()(l2)
    l2 = layers.LeakyReLU(0.2)(l2)

    l3 = layers.Conv2D(256, 5, 2, 'same')(l2)  # (32, 32, 256)
    l3 = layers.BatchNormalization()(l3)
    l3 = layers.LeakyReLU(0.2)(l3)

    l4 = layers.Conv2D(512, 5, 2, 'same')(l3)  # (16, 16, 512)
    l4 = layers.BatchNormalization()(l4)
    l4 = layers.LeakyReLU(0.2)(l4)

    l5 = layers.Flatten()(l4)
    l5 = layers.Dense(1)(l5)  # (None, 1)

    output = l5
    model = tf.keras.Model(input, output)
    return model


def train():
    image_batch = []
    label_batch = []
    k = 0

    generator = Generator()
    generator.summary()
    generator.build(input_shape=[Batch_size, Time_steps, L_node * W_node])
    discriminator = Discriminator()
    discriminator.summary()
    discriminator.build(input_shape=[Batch_size, L_node, W_node, Channel])

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate_gen, beta_1=Beta_1, beta_2=Beta_2, epsilon=E)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate_disc, beta_1=Beta_1, beta_2=Beta_2, epsilon=E)

    cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    for steps in range(training_steps):
        rand = random.sample(range(1, 6630), Batch_size)

        for i in range(Batch_size):
            image_sequence, label = read_and_load(rand[i])

            if i == 0:
                image_batch = image_sequence
                label_batch = label
            else:
                image_batch = layers.concatenate([image_batch, image_sequence], 0)  # (5, 5, 65536)
                label_batch = layers.concatenate([label_batch, label], 0)  # (5, 256, 256, 1)

        with tf.GradientTape() as Tape:
            gen_output = generator(image_batch, training=True)
            disc_real = discriminator(label_batch, training=True)
            disc_fake = discriminator(gen_output, training=True)
            d_loss_real = cross_entropy(tf.ones_like(disc_real) * 0.9, disc_real)
            d_loss_fake = cross_entropy(tf.zeros_like(disc_fake) + 0.1, disc_fake)
            d_loss = d_loss_fake + d_loss_real
        d_gradients = Tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

        with tf.GradientTape() as Tape:
            gen_output = generator(image_batch, training=True)
            disc_fake = discriminator(gen_output, training=True)
            g_loss = cross_entropy(tf.ones_like(disc_fake) * 0.9, disc_fake)
        g_gradients = Tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

        if k % 10 == 0:
            print("Step:{} Generator Loss:{:.4f} Discriminator Loss:{:.4f}".format(k, g_loss, d_loss))
            output_save = np.reshape(gen_output[1], newshape=[L_node, W_node])
            cv2.imwrite(Output_dir + str(k) + '.jpg', output_save * 255.)

        k = k + 1


def main(argv=None):
    train()


if __name__ == '__main__':
    main()