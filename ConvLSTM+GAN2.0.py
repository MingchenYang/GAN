# Image is a sequence of edge images, label is the prediction real frame, image_label is the prediction edge image.
import cv2
import random
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
Lambda = 10

File_dir = 'S:/pix2pix optflow/Generator256/sequence_edge_label/'
Output_dir = 'S:/pix2pix optflow/Generator256/CLSTM+GAN_result3.0/'
Save_dir = 'S:/pix2pix optflow/Generator256/CLSTM+GAN_save3.0/'


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
    Sequence_edge_label = np.reshape(Sequence_edge_label, newshape=[1, L_node, W_node, Channel])  # (1, 256, 256, 1)
    Sequence_edge_label = tf.cast(Sequence_edge_label, tf.float32)
    Sequence_edge_label = Sequence_edge_label / 255.

    return Sequence_image, Sequence_label, Sequence_edge_label


def lstm_gan():
    input = tf.keras.Input(shape=[Time_steps, L_node, W_node, Channel])  # （None, 5, 256, 256, 1)

    l1 = layers.ConvLSTM2D(10, 3, 1, 'same', return_sequences=True)(input)  # (None, 5, 256, 256, 10)
    l1 = layers.BatchNormalization()(l1)

    l2 = layers.ConvLSTM2D(20, 3, 1, 'same', return_sequences=True)(l1)  # (None, 5, 256, 256, 20)
    l2 = layers.BatchNormalization()(l2)

    l3 = layers.ConvLSTM2D(20, 3, 1, 'same', return_sequences=True)(l2)  # (None, 5, 256, 256, 20)
    l3 = layers.BatchNormalization()(l3)

    l4 = layers.ConvLSTM2D(10, 3, 1, 'same', return_sequences=False)(l3)  # (None, 256, 256, 10)
    l4 = layers.BatchNormalization()(l4)

    l5 = layers.Conv2D(1, 3, 1, 'same', activation='sigmoid')(l4)  # (None, 256, 256, 1)

    e1 = layers.Conv2D(64, 5, 2, 'same')(l5)  # (None, 128, 128, 64)
    # e1: (None, 128, 128, 64)
    e2 = layers.LeakyReLU(0.2)(e1)
    e2 = layers.Conv2D(64, 3, 1, 'same', activation='relu')(e2)
    e2 = layers.Conv2D(128, 5, 2, 'same')(e2)
    e2 = layers.BatchNormalization()(e2)
    # e2: (None, 64, 64, 128)
    e3 = layers.LeakyReLU(0.2)(e2)
    e3 = layers.Conv2D(128, 3, 1, 'same', activation='relu')(e3)
    e3 = layers.Conv2D(256, 5, 2, 'same')(e3)
    e3 = layers.BatchNormalization()(e3)
    # e3: (None, 32, 32, 256)
    e4 = layers.LeakyReLU(0.2)(e3)
    e4 = layers.Conv2D(256, 3, 1, 'same', activation='relu')(e4)
    e4 = layers.Conv2D(512, 5, 2, 'same')(e4)
    e4 = layers.BatchNormalization()(e4)
    # e4: (None, 16, 16, 512)
    e5 = layers.LeakyReLU(0.2)(e4)
    e5 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(e5)
    e5 = layers.Conv2D(512, 5, 2, 'same')(e5)
    e5 = layers.BatchNormalization()(e5)
    # e5: (None, 8, 8, 512)
    e6 = layers.LeakyReLU(0.2)(e5)
    e6 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(e6)
    e6 = layers.Conv2D(512, 5, 2, 'same')(e6)
    e6 = layers.BatchNormalization()(e6)
    # e6: (None, 4, 4, 512)
    e7 = layers.LeakyReLU(0.2)(e6)
    e7 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(e7)
    e7 = layers.Conv2D(512, 5, 2, 'same')(e7)
    e7 = layers.BatchNormalization()(e7)
    # e7: (None, 2, 2, 512)
    e8 = layers.LeakyReLU(0.2)(e7)
    e8 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(e8)
    e8 = layers.Conv2D(512, 5, 2, 'same')(e8)
    e8 = layers.BatchNormalization()(e8)
    # e8: (None, 1, 1, 512)
    # Decoder:
    d1 = layers.Activation('relu')(e8)
    d1 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(d1)
    d1 = layers.Conv2DTranspose(512, 5, 2, 'same')(d1)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.Dropout(0.5)(d1)
    d1 = layers.concatenate([d1, e7], 3)
    # d1: (None, 2, 2, 512*2)
    d2 = layers.Activation('relu')(d1)
    d2 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(d2)
    d2 = layers.Conv2DTranspose(512, 5, 2, 'same')(d2)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.Dropout(0.5)(d2)
    d2 = layers.concatenate([d2, e6], 3)
    # d2: (None, 4, 4, 512*2)
    d3 = layers.Activation('relu')(d2)
    d3 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(d3)
    d3 = layers.Conv2DTranspose(512, 5, 2, 'same')(d3)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.Dropout(0.5)(d3)
    d3 = layers.concatenate([d3, e5], 3)
    # d3: (None, 8, 8, 512*2)
    d4 = layers.Activation('relu')(d3)
    d4 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(d4)
    d4 = layers.Conv2DTranspose(512, 5, 2, 'same')(d4)
    d4 = layers.BatchNormalization()(d4)
    d4 = layers.Dropout(0.5)(d4)
    d4 = layers.concatenate([d4, e4], 3)
    # d4: (None, 16, 16, 512*2)
    d5 = layers.Activation('relu')(d4)
    d5 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(d5)
    d5 = layers.Conv2DTranspose(256, 5, 2, 'same')(d5)
    d5 = layers.BatchNormalization()(d5)
    d5 = layers.Dropout(0.5)(d5)
    d5 = layers.concatenate([d5, e3], 3)
    # d5: (None, 32, 32, 256*2)
    d6 = layers.Activation('relu')(d5)
    d6 = layers.Conv2D(256, 3, 1, 'same', activation='relu')(d6)
    d6 = layers.Conv2DTranspose(128, 5, 2, 'same')(d6)
    d6 = layers.BatchNormalization()(d6)
    d6 = layers.Dropout(0.5)(d6)
    d6 = layers.concatenate([d6, e2], 3)
    # d6: (None, 64, 64, 128*2)
    d7 = layers.Activation('relu')(d6)
    d7 = layers.Conv2D(128, 3, 1, 'same', activation='relu')(d7)
    d7 = layers.Conv2DTranspose(64, 5, 2, 'same')(d7)
    d7 = layers.BatchNormalization()(d7)
    d7 = layers.Dropout(0.5)(d7)
    d7 = layers.concatenate([d7, e1], 3)
    # d7: (None, 128, 128, 64*2)
    d8 = layers.Activation('relu')(d7)
    d8 = layers.Conv2D(64, 3, 1, 'same', activation='relu')(d8)
    d8 = layers.Conv2DTranspose(1, 5, 2, 'same')(d8)
    d8 = layers.Activation('tanh')(d8)

    Model = tf.keras.Model(input, d8, name='generator')
    return Model


def discriminator():
    input = tf.keras.Input(shape=[L_node, W_node, Channel])
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

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate_gen, beta_1=Beta_1, beta_2=Beta_2, epsilon=E)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate_disc, beta_1=Beta_1, beta_2=Beta_2, epsilon=E)

    gen = lstm_gan()
    gen.build(input_shape=[Batch_size, Num_sequence, L_node, W_node, Channel])
    gen.summary()

    disc = discriminator()
    disc.build(input_shape=[Batch_size, L_node, W_node, Channel])
    disc.summary()

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

        with tf.GradientTape() as Tape:
            # Generator
            gen_output = gen(image_batch, training=True)
            # Discriminator
            disc_input_real = edge_batch
            disc_input_fake = gen_output

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
            disc_input_fake = gen_output
            disc_fake = disc(disc_input_fake, training=True)
            # Loss
            g_loss_entropy = cross_entropy(tf.ones_like(disc_fake) * 0.9, disc_fake)
            # g_loss_l1 = Lambda * tf.reduce_mean(tf.abs(gen_output - edge_batch))
            g_loss_ssim = Lambda * tf.reduce_mean(1 - tf.image.ssim(gen_output, edge_batch, max_val=1))
            g_loss = g_loss_entropy + g_loss_ssim

        g_gradients = Tape.gradient(g_loss, gen.trainable_variables)
        g_optimizer.apply_gradients(zip(g_gradients, gen.trainable_variables))

        if k % 10 == 0:
            print("Step:{} Generator Loss:{:.4f} {:.4f} Discriminator Loss:{:.4f}".format(k, g_loss, g_loss_ssim, d_loss))

            edge_save = np.reshape(edge_batch[0], newshape=[L_node, W_node])
            cv2.imwrite(Output_dir + str(k) + '-origin.jpg', edge_save * 255.)

            label_save = np.reshape(label_batch[0], newshape=[L_node, W_node])
            cv2.imwrite(Output_dir + str(k) + '-edge.jpg', label_save * 255.)

            output_save = np.reshape(gen_output[0], newshape=[L_node, W_node])
            cv2.imwrite(Output_dir + str(k) + '.jpg', output_save * 255.)

        if k % 1000 == 0:
            gen.save(Save_dir + str(k) + 'Gmodel' + '.h5')

        k = k + 1


def main(argv=None):
    train()


if __name__=='__main__':
    main()