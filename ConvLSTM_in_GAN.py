import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

Training_steps = 50000
Batch_size = 10
Num_sequence = Time_steps = 5
L_node = 256
W_node = 256
Channel = 1
Learning_rate_gen = 0.0002
Learning_rate_disc = 0.002
Beta_1 = 0.5
Beta_2 = 0.999
E = 1e-08
Lambda = 1

File_dir = 'S:/pix2pix optflow/Generator256/sequence_edge_label/'
Output_dir = 'S:/pix2pix optflow/Generator256/CLSTM_in_GAN_result/'
Save_dir = 'S:/pix2pix optflow/Generator256/CLSTM_in_GAN_save/'


def read_and_load(name):
    name = '%04d' % name
    File_name = File_dir + str(name) + '.npz'  # S:/pix2pix optflow/Generator256/sequence/0001.npz

    Sequence_image = np.load(File_name)['image']
    Sequence_image = np.reshape(Sequence_image,
                                newshape=[1, Num_sequence, L_node, W_node, Channel])  # (1, 5, 256, 256, 1)
    Sequence_image = tf.cast(Sequence_image, tf.float32)
    Sequence_image = Sequence_image / 255.

    Sequence_label = np.load(File_name)['label']
    Sequence_label = np.reshape(Sequence_label, newshape=[1, L_node, W_node, Channel])  # (1, 256, 256, 1)
    Sequence_label = tf.cast(Sequence_label, tf.float32)
    Sequence_label = Sequence_label / 255.

    Sequence_edge_label = np.load(File_name)['image_label']
    Sequence_edge_label = np.reshape(Sequence_edge_label, newshape=[1, L_node, W_node, Channel])
    Sequence_edge_label = tf.cast(Sequence_edge_label, tf.float32)
    Sequence_edge_label = Sequence_edge_label / 255.

    return Sequence_image, Sequence_label, Sequence_edge_label


'''
def encoder():
    input = tf.keras.Input(shape=[L_node, W_node, Channel])  # (None, 256, 256, 1)

    e1 = layers.Conv2D(64, 5, 2, 'same')(input)  # (None, 128, 128, 64)
    e1 = layers.LeakyReLU(0.2)(e1)
    e1 = layers.Conv2D(64, 3, 1, 'same', activation='relu')(e1)  # (None, 128, 128, 64)

    e2 = layers.Conv2D(128, 5, 2, 'same')(e1)  # (None, 64, 64, 128)
    e2 = layers.BatchNormalization()(e2)
    e2 = layers.LeakyReLU(0.2)(e2)
    e2 = layers.Conv2D(128, 3, 1, 'same', activation='relu')(e2)  # (None, 64, 64, 128)

    e3 = layers.Conv2D(256, 5, 2, 'same')(e2)  # (None, 32, 32, 256)
    e3 = layers.BatchNormalization()(e3)
    e3 = layers.LeakyReLU(0.2)(e3)
    e3 = layers.Conv2D(256, 3, 1, 'same', activation='relu')(e3)  # (None, 32, 32, 256)

    Model = tf.keras.Model(input, e3, name='encoder')
    return Model


def convLSTM():
    input = tf.keras.Input(shape=[Time_steps, 32, 32, 256])  # (None, 5, 32, 32, 256)

    c1 = layers.ConvLSTM2D(10, 3, 1, 'same', return_sequences=True)(input)
    c1 = layers.ConvLSTM2D(20, 3, 1, 'same', return_sequences=True)(c1)
    c1 = layers.ConvLSTM2D(10, 3, 1, 'same', return_sequences=False)(c1)

    c2 = layers.Conv2D(256, 3, 1, 'same', activation='relu')(c1)  # (None, 32, 32, 256)

    Model = tf.keras.Model(input, c2, name='convLSTM')
    return Model


def decoder():
    input = tf.keras.Input(shape=[32, 32, 256])  # (None, 32, 32, 256)

    d1 = layers.Conv2DTranspose(128, 5, 2, 'same', activation='relu')(input)  # (None, 64, 64, 128)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.Dropout(0.5)(d1)
    d1 = layers.Conv2D(128, 3, 1, 'same', activation='relu')(d1)

    d2 = layers.Conv2DTranspose(64, 5, 2, 'same', activation='relu')(d1)  # (None, 128, 128, 64)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.Dropout(0.5)(d2)
    d2 = layers.Conv2D(64, 3, 1, 'same', activation='relu')(d2)

    d3 = layers.Conv2DTranspose(64, 5, 2, 'same', activation='relu')(d2)  # (None, 256, 256, 32)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.Dropout(0.5)(d3)
    d3 = layers.Conv2D(1, 3, 1, 'same', activation='tanh')(d3)  # (None, 256, 256, 1)

    Model = tf.keras.Model(input, d3, name='decoder')
    return Model
'''


def generator():
    input = tf.keras.Input(shape=[L_node, W_node, Channel])  # (None, 256, 256, 1)

    e0 = layers.Conv2D(32, 3, 1, 'same', activation='relu')(input)  # (None, 256, 256, 32)

    e1 = layers.Conv2D(64, 5, 2, 'same')(e0)  # (None, 128, 128, 64)
    e1_1 = layers.LeakyReLU(0.2)(e1)
    e1_1 = layers.Conv2D(64, 3, 1, 'same', activation='relu')(e1_1)  # (None, 128, 128, 64)

    e2 = layers.Conv2D(128, 5, 2, 'same')(e1_1)  # (None, 64, 64, 128)
    e2 = layers.BatchNormalization()(e2)
    e2_1 = layers.LeakyReLU(0.2)(e2)
    e2_1 = layers.Conv2D(128, 3, 1, 'same', activation='relu')(e2_1)  # (None, 64, 64, 128)

    e3 = layers.Conv2D(256, 5, 2, 'same')(e2_1)  # (None, 32, 32, 256)
    e3 = layers.BatchNormalization()(e3)
    e3 = layers.LeakyReLU(0.2)(e3)
    e3 = layers.Conv2D(256, 3, 1, 'same', activation='relu')(e3)  # (None, 32, 32, 256)
    e3 = tf.reshape(e3, shape=[-1, Num_sequence, 32, 32, 256])

    c1 = layers.ConvLSTM2D(10, 3, 1, 'same', return_sequences=True)(e3)
    c1 = layers.ConvLSTM2D(20, 3, 1, 'same', return_sequences=True)(c1)
    c1 = layers.ConvLSTM2D(10, 3, 1, 'same', return_sequences=False)(c1)

    c2 = layers.Conv2D(256, 3, 1, 'same', activation='relu')(c1)  # (None, 32, 32, 256)

    d1 = layers.Conv2DTranspose(128, 5, 2, 'same', activation='relu')(c2)  # (None, 64, 64, 128)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.Dropout(0.5)(d1)
    d1 = layers.concatenate([d1, e2[Batch_size * (Num_sequence - 1):Batch_size * Num_sequence, ]], 3)  # (None, 64, 64, 256)
    d1 = layers.Activation('relu')(d1)
    d1 = layers.Conv2D(128, 3, 1, 'same', activation='relu')(d1)

    d2 = layers.Conv2DTranspose(64, 5, 2, 'same', activation='relu')(d1)  # (None, 128, 128, 64)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.Dropout(0.5)(d2)
    d2 = layers.concatenate([d2, e1[Batch_size * (Num_sequence - 1):Batch_size * Num_sequence, ]], 3)  # (None, 128, 128, 128)
    d2 = layers.Activation('relu')(d2)
    d2 = layers.Conv2D(64, 3, 1, 'same', activation='relu')(d2)

    d3 = layers.Conv2DTranspose(64, 5, 2, 'same', activation='relu')(d2)  # (None, 256, 256, 32)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.Dropout(0.5)(d3)
    d3 = layers.concatenate([d3, e0[Batch_size * (Num_sequence - 1):Batch_size * Num_sequence, ]], 3)  # (None, 256, 256, 64)
    d3 = layers.Activation('relu')(d3)
    d3 = layers.Conv2D(1, 3, 1, 'same', activation='tanh')(d3)  # (None, 256, 256, 1)

    Model = tf.keras.Model(input, d3)
    return Model


def discriminator():
    input = tf.keras.Input(shape=[L_node, W_node, Channel * 2])
    # input: (None, 256, 256, 2)
    h0 = layers.Conv2D(64, 5, 2, 'same')(input)
    h0 = layers.LeakyReLU(0.2)(h0)
    # h0: (None, 128, 128, 64)
    h1 = layers.Conv2D(128, 5, 2, 'same')(h0)
    h1 = layers.BatchNormalization()(h1)
    h1 = layers.LeakyReLU(0.2)(h1)
    # h1: (None, 64, 64, 128)
    h2 = layers.Conv2D(256, 5, 2, 'same')(h1)
    h2 = layers.BatchNormalization()(h2)
    h2 = layers.LeakyReLU(0.2)(h2)
    # h2: (None, 32, 32, 256)
    h3 = layers.Conv2D(512, 5, 2, 'same')(h2)
    h3 = layers.BatchNormalization()(h3)
    h3 = layers.LeakyReLU(0.2)(h3)
    # h3: (None, 16, 16, 512)
    h4 = layers.Flatten()(h3)
    h4 = layers.Dense(1)(h4)
    # h4 = layers.Activation('sigmoid')(h4)
    # h4: (None, 1)
    Model = tf.keras.Model(input, h4, name='discriminator')
    return Model


def train():
    image_batch = []
    label_batch = []
    edge_batch = []
    k = 0

    '''
    en = encoder()
    en.build(input_shape=[Batch_size, L_node, W_node, Channel])
    en.summary()

    lstm = convLSTM()
    lstm.build(input_shape=[Batch_size, Time_steps, 32, 32, 256])
    lstm.summary()

    de = decoder()
    de.build(input_shape=[Batch_size, 32, 32, 256])
    de.summary()
    '''

    gen = generator()
    gen.build(input_shape=[Batch_size * Num_sequence, L_node, W_node, Channel])
    gen.summary()

    disc = discriminator()
    disc.build(input_shape=[Batch_size, L_node, W_node, Channel * 2])
    disc.summary()

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate_gen, beta_1=Beta_1, beta_2=Beta_2, epsilon=E)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate_disc, beta_1=Beta_1, beta_2=Beta_2, epsilon=E)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    for steps in range(Training_steps):
        rand = random.sample(range(1, 6630), Batch_size)

        for i in range(Batch_size):
            image_sequence, label, edge_label = read_and_load(rand[i])

            if i == 0:
                image_batch = image_sequence
                label_batch = label
                edge_batch = edge_label
            else:
                image_batch = layers.concatenate([image_batch, image_sequence], 0)  # (None, 5, 256, 256, 1)
                label_batch = layers.concatenate([label_batch, label], 0)  # (None, 256, 256, 1)
                edge_batch = layers.concatenate([edge_batch, edge_label], 0)  # (None, 256, 256, 1)
        image_batch = tf.reshape(image_batch, shape=[Batch_size * Num_sequence, L_node, W_node, Channel])

        with tf.GradientTape() as Tape:
            ''''
            # Encoder
            for layer_num in range(Num_sequence):
                en_output = en(image_batch[:, layer_num, ], training=True)  # (None, 32, 32, 256)
                en_output = np.reshape(en_output, newshape=[Batch_size, 1, 32, 32, 256])  # (None, 1, 32, 32, 256)

                if layer_num == 0:
                    en_output_list = en_output
                else:
                    en_output_list = layers.concatenate([en_output_list, en_output], 1)  # (None, 5, 32, 32, 256)
            # ConvLSTM
            lstm_output = lstm(en_output_list, training=True)  # (None, 32, 32, 256)
            # Decoder
            de_output = de(lstm_output, training=True)  # (None, 256, 256, 1)
            '''

            # Generator
            gen_output = gen(image_batch, training=True)
            # Discriminator
            disc_input_real = layers.concatenate([edge_batch, label_batch], 3)
            disc_input_fake = layers.concatenate([gen_output, label_batch], 3)

            disc_real = disc(disc_input_real, training=True)
            disc_fake = disc(disc_input_fake, training=False)
            #Loss
            d_loss_real = cross_entropy(tf.ones_like(disc_real) * 0.9, disc_real)
            d_loss_fake = cross_entropy(tf.zeros_like(disc_fake) + 0.1, disc_fake)
            d_loss = d_loss_real + d_loss_fake

        d_gradients = Tape.gradient(d_loss, disc.trainable_variables)
        d_optimizer.apply_gradients(zip(d_gradients, disc.trainable_variables))

        with tf.GradientTape() as Tape:
            # Generator
            gen_output = gen(image_batch, training=True)
            # Discriminator
            disc_input_fake = layers.concatenate([gen_output, label_batch], 3)
            disc_fake = disc(disc_input_fake, training=True)
            # Loss
            g_loss_entropy = cross_entropy(tf.ones_like(disc_fake) * 0.9, disc_fake)
            g_loss_l1 = Lambda * tf.reduce_mean(tf.abs(gen_output - edge_batch))
            g_loss_ssim = tf.reduce_mean(1 - tf.image.ssim(gen_output, edge_batch, max_val=1))
            g_loss = g_loss_entropy + g_loss_l1

        g_gradients = Tape.gradient(g_loss, gen.trainable_variables)
        g_optimizer.apply_gradients(zip(g_gradients, gen.trainable_variables))

        if k % 10 == 0:
            print("Step:{} Generator Loss:{:.4f} {:.4f} Discriminator Loss:{:.4f}".format(k, g_loss, g_loss_l1, d_loss))
            edge_batch_save = np.reshape(edge_batch[0], newshape=[L_node, W_node])
            cv2.imwrite(Output_dir + str(k) + 'edge.jpg', edge_batch_save * 255.)
            output_save = np.reshape(gen_output[0], newshape=[L_node, W_node])
            cv2.imwrite(Output_dir + str(k) + '.jpg', output_save * 255.)
        if k % 1000 == 0:
            gen.save(Save_dir + str(k) + 'Gmodel' + '.h5')

        k = k + 1


def main(argv=None):
    train()


if __name__ == '__main__':
    main()
