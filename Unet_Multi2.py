# Input image is a sequence of real frames, label is the prediction frame.
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

Batch_size = 8
Time_steps = Num_sequence = 5
L_node = 256
W_node = 256
Channel = 1
Output_dim = 100
Learning_rate_gen = 0.0001
Learning_rate_disc = 0.0002
Beta_1 = 0.5
Beta_2 = 0.999
E = 1e-08
Training_steps = 100000
Lambda = 5
Lambda1 = 15
LSTM_layer = 64

File_dir = 'S:/UCSD_ped2/Train256/training_Multi/'
Output_dir = 'S:/UCSD_ped2/Train256/result_Multi/'
Save_dir = 'S:/UCSD_ped2/Train256/save_Multi/'


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

    l3 = layers.Conv2D(64, 3, 1, 'same', activation='relu')(l1)  # (None, 256, 256, 64)

    l4_1 = layers.Conv2D(128, 5, 2, 'same')(l3)  # (None, 128, 128, 128)

    l5_1 = layers.LeakyReLU(0.2)(l4_1)
    l5_1 = layers.Conv2D(128, 3, 1, 'same', activation='relu')(l5_1)
    l5_1 = layers.Conv2D(256, 5, 2, 'same')(l5_1)  # (None, 64, 64, 256)
    l5_1 = layers.BatchNormalization()(l5_1)

    l6_1 = layers.LeakyReLU(0.2)(l5_1)
    l6_1 = layers.Conv2D(256, 3, 1, 'same', activation='relu')(l6_1)
    l6_1 = layers.Conv2D(512, 5, 2, 'same')(l6_1)  # (None, 32, 32, 512)
    l6_1 = layers.BatchNormalization()(l6_1)

    l7_1 = layers.LeakyReLU(0.2)(l6_1)
    l7_1 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(l7_1)
    l7_1 = layers.Conv2D(512, 5, 2, 'same')(l7_1)  # (None, 16, 16, 512)
    l7_1 = layers.BatchNormalization()(l7_1)

    l8_1 = layers.Activation('relu')(l7_1)
    l8_1 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(l8_1)
    l8_1 = layers.Conv2DTranspose(512, 5, 2, 'same')(l8_1)  # (None, 32, 32, 512)
    l8_1 = layers.BatchNormalization()(l8_1)
    l8_1 = layers.Dropout(0.3)(l8_1)
    # l8_1 = layers.concatenate([l8_1, l6_1], 3)  # (None, 32, 32, 512*2)

    l9_1 = layers.Activation('relu')(l8_1)
    l9_1 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(l9_1)
    l9_1 = layers.Conv2DTranspose(256, 5, 2, 'same')(l9_1)  # (None, 64, 64, 256)
    l9_1 = layers.BatchNormalization()(l9_1)
    l9_1 = layers.Dropout(0.3)(l9_1)
    # l9_1 = layers.concatenate([l9_1, l5_1], 3)  # (None, 64, 64, 256*2)

    l10_1 = layers.Activation('relu')(l9_1)
    l10_1 = layers.Conv2D(256, 3, 1, 'same', activation='relu')(l10_1)
    l10_1 = layers.Conv2DTranspose(128, 5, 2, 'same')(l10_1)  # (None, 128, 128, 128)
    l10_1 = layers.BatchNormalization()(l10_1)
    l10_1 = layers.Dropout(0.3)(l10_1)
    # l10_1 = layers.concatenate([l10_1, l4_1], 3)  # (None, 128, 128, 128*2)

    l11_1 = layers.Activation('relu')(l10_1)
    l11_1 = layers.Conv2D(128, 3, 1, 'same', activation='relu')(l11_1)
    l11_1 = layers.Conv2DTranspose(64, 5, 2, 'same')(l11_1)  # (None, 256, 256, 64)
    l11_1 = layers.Activation('relu')(l11_1)

    l4_2 = layers.Conv2D(32, 3, 2, 'same')(l3)  # (None, 128, 128, 32)

    l5_2 = layers.Conv2D(5, 3, 2, 'same')(l4_2)  # (None, 64, 64, 5)
    l5_2_1 = tf.reshape(l5_2[:, :, :, 0], shape=(-1, 1, LSTM_layer, LSTM_layer, Channel))
    l5_2_2 = tf.reshape(l5_2[:, :, :, 1], shape=(-1, 1, LSTM_layer, LSTM_layer, Channel))
    l5_2_3 = tf.reshape(l5_2[:, :, :, 2], shape=(-1, 1, LSTM_layer, LSTM_layer, Channel))
    l5_2_4 = tf.reshape(l5_2[:, :, :, 3], shape=(-1, 1, LSTM_layer, LSTM_layer, Channel))
    l5_2_5 = tf.reshape(l5_2[:, :, :, 4], shape=(-1, 1, LSTM_layer, LSTM_layer, Channel))
    l5_2 = layers.concatenate([l5_2_1, l5_2_2, l5_2_3, l5_2_4, l5_2_5], 1)  # (None, 5, 64, 64, 1)

    l6_2 = layers.ConvLSTM2D(10, 3, 1, 'same', return_sequences=True)(l5_2)  # (None, 5, 64, 64, 10)

    l7_2 = layers.ConvLSTM2D(20, 3, 1, 'same', return_sequences=True)(l6_2)  # (None, 5, 64, 64, 20)

    l8_2 = layers.ConvLSTM2D(20, 3, 1, 'same', return_sequences=True)(l7_2)  # (None, 5, 64, 64, 20)

    l9_2 = layers.ConvLSTM2D(10, 3, 1, 'same', return_sequences=False)(l8_2)  # (None, 64, 64, 10)

    l10_2 = layers.Conv2DTranspose(32, 3, 2, 'same')(l9_2)  # (None, 128, 128, 32)
    l10_2 = layers.BatchNormalization()(l10_2)
    l10_2 = layers.Dropout(0.3)(l10_2)
    l10_2 = layers.Activation('relu')(l10_2)

    l11_2 = layers.Conv2DTranspose(64, 3, 2, 'same')(l10_2)  # (None, 256, 256, 64)
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
    discriminator.build(input_shape=(Batch_size, L_node, W_node, Channel))

    # Build Generator
    generator = build_generator()
    generator.build(input_shape=(L_node, W_node, Time_steps))
    generator.summary()

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    for steps in range(Training_steps):
        rand = random.sample(range(1, 2454), Batch_size)

        for i in range(Batch_size):
            image_sequence, label = read_and_load(rand[i])

            if i == 0:
                image_batch = image_sequence
                label_batch = label
            else:
                image_batch = layers.concatenate([image_batch, image_sequence], 0)  # (None, 256, 256, 5)
                label_batch = layers.concatenate([label_batch, label], 0)  # (None, 256, 256, 1)

        with tf.GradientTape() as Tape:
            # Generator
            gen_output = generator(image_batch, training=False)
            # Discriminator
            disc_input_real = label_batch
            disc_input_fake = gen_output

            disc_real = discriminator(disc_input_real, training=True)
            disc_fake = discriminator(disc_input_fake, training=True)
            #Loss
            d_loss_real = cross_entropy(tf.ones_like(disc_real) * 0.9, disc_real)
            d_loss_fake = cross_entropy(tf.zeros_like(disc_fake) + 0.1, disc_fake)
            d_loss = d_loss_real + d_loss_fake

        d_gradients = Tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

        with tf.GradientTape() as Tape:
            # Generator
            gen_output = generator(image_batch, training=True)
            # Discriminator
            disc_input_fake = gen_output
            disc_fake = discriminator(disc_input_fake, training=False)
            # Loss
            g_loss_entropy = cross_entropy(tf.ones_like(disc_fake) * 0.9, disc_fake)
            g_loss_l1 = Lambda * tf.reduce_mean(tf.abs(gen_output - label_batch))
            g_loss_ssim = Lambda1 * tf.reduce_mean(1 - tf.image.ssim(gen_output, label_batch, max_val=1))
            g_loss = g_loss_entropy + g_loss_l1 + g_loss_ssim

        g_gradients = Tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

        if k % 100 == 0:
            print("Step:{}, Generator Loss:{:.4f}, L1 Loss:{:.4f}, SSIM Loss:{:.4f}, Discriminator Loss:{:.4f}".format(k, g_loss, g_loss_l1 / Lambda, g_loss_ssim / Lambda1, d_loss))

            output_save = np.reshape(gen_output[0], newshape=[L_node, W_node])
            cv2.imwrite(Output_dir + str(k) + '.jpg', output_save * 255.)

        if k % 2000 == 0:
            generator.save(Save_dir + str(k) + 'Gmodel' + '.h5')
            discriminator.save(Save_dir + str(k) + 'Dmodel' + '.h5')

        k = k + 1


def main(argv=None):
    train()


if __name__=='__main__':
    main()