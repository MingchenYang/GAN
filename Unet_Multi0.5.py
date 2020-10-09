# Input image is a sequence of real frames, label is the prediction frame.
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

Batch_size = 4
Time_steps = Num_sequence = 5
L_node = 256
W_node = 256
Channel = 1
Output_dim = 100
Learning_rate_gen = 0.0002
Learning_rate_disc = 0.00002
Beta_1 = 0.5
Beta_2 = 0.999
E = 1e-08
Training_steps = 100000
Lambda = 10
Lambda1 = 10

File_dir = 'S:/pix2pix optflow/Train256/training_Multi/'
Output_dir = 'S:/pix2pix optflow/Train256/result_Multi/'
Save_dir = 'S:/pix2pix optflow/Train256/save_Multi/'


def read_and_load(name):
    name = '%04d' % name
    File_name = File_dir + str(name) + '.npz'  # S:/pix2pix optflow/Generator256/sequence/0001.npz

    Sequence_image = np.load(File_name)['image']  # (5, 256, 256)
    Sequence_image = np.reshape(Sequence_image,
                                newshape=[Num_sequence, L_node, W_node, Channel])  # (5, 256, 256, 1)
    for num in range(Num_sequence):
        if num == 0:
            image = Sequence_image[num, ]
            image_reshape = image
        else:
            image_reshape = np.concatenate([image_reshape, image], 2)  # (256, 256, 5)
    Sequence_image = np.reshape(image_reshape, newshape=[1, L_node, W_node, Time_steps])  # (1, 256, 256, 5)
    Sequence_image = tf.cast(Sequence_image, tf.float32)
    Sequence_image = Sequence_image / 255.

    Sequence_label = np.load(File_name)['label']
    Sequence_label = np.reshape(Sequence_label, newshape=[1, L_node, W_node, Channel])  # (1, 256, 256, 1)
    Sequence_label = tf.cast(Sequence_label, tf.float32)
    Sequence_label = Sequence_label / 255.

    return Sequence_image, Sequence_label


def build_generator():
    input = tf.keras.Input(shape=(L_node, W_node, Time_steps))  # (None, 256, 256, 5)

    l1 = input  # (None, 256, 256, 5)

    l2_1 = layers.Conv2D(64, 1, 1, 'same', activation='relu')(l1)  # (None, 256, 256, 64)
    l2_1 = layers.BatchNormalization()(l2_1)

    l2_2 = layers.Conv2D(48, 1, 1, 'same', activation='relu')(l1)  # (None, 256, 256, 48)
    l2_2 = layers.BatchNormalization()(l2_2)
    l2_2 = layers.Conv2D(64, 3, 1, 'same', activation='relu')(l2_2)  # (None, 256, 256, 64)
    l2_2 = layers.BatchNormalization()(l2_2)

    l2_3 = layers.Conv2D(48, 1, 1, 'same', activation='relu')(l1)  # (None, 256, 256, 48)
    l2_3 = layers.BatchNormalization()(l2_3)
    l2_3 = layers.Conv2D(64, 5, 1, 'same', activation='relu')(l2_3)  # (None, 256, 256, 64)
    l2_3 = layers.BatchNormalization()(l2_3)

    l2_4 = layers.AvgPool2D(3, 1, 'same')(l1)
    l2_4 = layers.Conv2D(64, 1, 1, 'same', activation='relu')(l2_4)  # (None, 256, 256, 64)
    l2_4 = layers.BatchNormalization()(l2_4)

    l2 = layers.concatenate([l2_1, l2_2, l2_3, l2_4], 3)  # (None, 256, 256, 256)

    l3 = layers.Conv2D(64, 3, 1, 'same', activation='relu')(l2)  # (None, 256, 256, 64)

    l4_1 = layers.Conv2D(128, 3, 2, 'same')(l3)  # (None, 128, 128, 128)
    l4_1 = layers.BatchNormalization()(l4_1)
    l4_1 = layers.LeakyReLU(0.2)(l4_1)

    l5_1 = layers.Conv2D(256, 3, 2, 'same')(l4_1)  # (None, 64, 64, 256)
    l5_1 = layers.BatchNormalization()(l5_1)
    l5_1 = layers.LeakyReLU(0.2)(l5_1)

    l6_1 = layers.Conv2D(512, 3, 2, 'same')(l5_1)  # (None, 32, 32, 512)
    l6_1 = layers.BatchNormalization()(l6_1)
    l6_1 = layers.LeakyReLU(0.2)(l6_1)

    l7_1 = layers.Conv2D(512, 3, 2, 'same')(l6_1)  # (None, 16, 16, 512)
    l7_1 = layers.BatchNormalization()(l7_1)
    l7_1 = layers.LeakyReLU(0.2)(l7_1)

    l8_1 = layers.Conv2DTranspose(512, 3, 2, 'same')(l7_1)  # (None, 32, 32, 512)
    l8_1 = layers.BatchNormalization()(l8_1)
    l8_1 = layers.Dropout(0.3)(l8_1)
    l8_1 = layers.Activation('relu')(l8_1)

    l9_1 = layers.Conv2DTranspose(256, 3, 2, 'same')(l8_1)  # (None, 64, 64, 256)
    l9_1 = layers.BatchNormalization()(l9_1)
    l9_1 = layers.Dropout(0.3)(l9_1)
    l9_1 = layers.Activation('relu')(l9_1)

    l10_1 = layers.Conv2DTranspose(128, 3, 2, 'same')(l9_1)  # (None, 128, 128, 128)
    l10_1 = layers.BatchNormalization()(l10_1)
    l10_1 = layers.Dropout(0.3)(l10_1)
    l10_1 = layers.Activation('relu')(l10_1)

    l11_1 = layers.Conv2DTranspose(64, 3, 2, 'same')(l10_1)  # (None, 256, 256, 64)
    l11_1 = layers.BatchNormalization()(l11_1)
    l11_1 = layers.Dropout(0.3)(l11_1)
    l11_1 = layers.Activation('tanh')(l11_1)

    l4_2 = layers.Conv2D(32, 3, 2, 'same')(l3)  # (None, 128, 128, 32)

    l5_2 = layers.Conv2D(5, 3, 2, 'same')(l4_2)  # (None, 64, 64, 5)
    # l5_2_1 = tf.reshape(l5_2(-1, -1, -1, 0), shape=(1, L_node, W_node, Channel))
    # l5_2_2 = tf.reshape(l5_2(-1, -1, -1, 1), shape=(1, L_node, W_node, Channel))
    # l5_2_3 = tf.reshape(l5_2(-1, -1, -1, 2), shape=(1, L_node, W_node, Channel))
    # l5_2_4 = tf.reshape(l5_2(-1, -1, -1, 3), shape=(1, L_node, W_node, Channel))
    # l5_2_5 = tf.reshape(l5_2(-1, -1, -1, 4), shape=(1, L_node, W_node, Channel))
    l5_2 = tf.reshape(l5_2, shape=(-1, 5, 64, 64, 1))  # (None, 5, 64, 64, 1)

    l6_2 = layers.ConvLSTM2D(10, 3, 1, 'same', return_sequences=True)(l5_2)  # (None, 5, 64, 64, 10)

    l7_2 = layers.ConvLSTM2D(20, 3, 1, 'same', return_sequences=True)(l6_2)  # (None, 5, 64, 64, 20)

    l8_2 = layers.ConvLSTM2D(20, 3, 1, 'same', return_sequences=True)(l7_2)  # (None, 5, 64, 64, 20)

    l9_2 = layers.ConvLSTM2D(10, 3, 1, 'same', return_sequences=False)(l8_2)  # (None, 64, 64, 10)

    l10_2 = layers.Conv2DTranspose(32, 3, 2, 'same')(l9_2)  # (None, 128, 128, 32)
    l10_2 = layers.BatchNormalization()(l10_2)
    l10_2 = layers.Dropout(0.3)(l10_2)
    l10_2 = layers.Activation('relu')(l10_2)

    l11_2 = layers.Conv2DTranspose(64, 3, 2, 'same')(l10_2)  # (None, 256, 256, 64)
    l11_2 = layers.BatchNormalization()(l11_2)
    l11_2 = layers.Dropout(0.3)(l11_2)
    l11_2 = layers.Activation('relu')(l11_2)

    l12 = layers.add([l11_1, l11_2])  # (None, 256, 256, 64)

    l13 = layers.Conv2D(1, 3, 1, 'same', activation='tanh')(l12)  # (None, 256, 256, 1)

    Model = tf.keras.Model(input, l13, name='generator')
    return Model


def build_discriminator():
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


def build_combined(g, d):
    Model = tf.keras.Sequential()
    Model.add(g)
    d.trainable = False
    Model.add(d)
    return Model


def train():
    image_batch = []
    label_batch = []
    k = 0

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate_gen, beta_1=Beta_1, beta_2=Beta_2, epsilon=E)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate_disc, beta_1=Beta_1, beta_2=Beta_2, epsilon=E)

    # Build Discriminator
    discriminator = build_discriminator()
    discriminator.compile(optimizer=d_optimizer, loss='binary_crossentropy')

    # Build Generator
    generator = build_generator()
    generator.summary()
    generator.compile(optimizer=g_optimizer, loss='binary_crossentropy')

    # Build Combined Network
    combined = build_combined(generator, discriminator)
    combined.compile(optimizer=g_optimizer, loss='binary_crossentropy')

    for steps in range(Training_steps):
        rand = random.sample(range(1, 6630), Batch_size)

        for i in range(Batch_size):
            image_sequence, label = read_and_load(rand[i])

            if i == 0:
                image_batch = image_sequence
                label_batch = label
            else:
                image_batch = layers.concatenate([image_batch, image_sequence], 0)  # (None, 256, 256, 5)
                label_batch = layers.concatenate([label_batch, label], 0)  # (None, 256, 256, 1)

        # --------------------
        # Train Discriminator
        # --------------------

        valid = np.ones(Batch_size)
        fake = np.zeros(Batch_size)

        # Generate fake images
        gen_output = generator.predict(image_batch)

        # Train discriminator
        discriminator.trainable = True
        disc_loss_real = discriminator.train_on_batch(label_batch, valid)
        disc_loss_fake = discriminator.train_on_batch(gen_output, fake)
        d_loss = 0.5 * (disc_loss_real + disc_loss_fake)

        # ---------------
        # Train Generator
        # ---------------

        g_loss = combined.train_on_batch(image_batch, valid)

        if k % 10 == 0:
            print("Step:{}, Generator Loss:{:.4f}, Discriminator Loss:{:.4f}".format(k, g_loss, d_loss))

            output_save = np.reshape(gen_output[0], newshape=[L_node, W_node])
            cv2.imwrite(Output_dir + str(k) + '.jpg', output_save * 255.)

        if k % 1000 == 0:
            generator.save(Save_dir + str(k) + 'Gmodel' + '.h5')
            discriminator.save(Save_dir + str(k) + 'Dmodel' + '.h5')

        k = k + 1


def main(argv=None):
    train()


if __name__=='__main__':
    main()