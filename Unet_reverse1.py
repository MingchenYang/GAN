# Input image is a sequence of real frames, label is the prediction frame.
import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

Batch_size = 16
L_node = 256
W_node = 256
Channel = 1
Output_dim = 100
Learning_rate_gen = 0.0001
Learning_rate_disc = 0.0002
Beta_1 = 0.5
Beta_2 = 0.999
E = 1e-08
Training_steps = 5
Buffer_size = 1000
Lambda_entropy = 0.5
Lambda_l1 = 0.2
Lambda_ssim = 0.3

Image_dir = 'S:/UCSD_ped2/Train256/label/'
Label_dir = 'S:/UCSD_ped2/Train256/training/'
Output_dir = 'S:/UCSD_ped2/Train256/result_Reverse/'
Save_dir = 'S:/UCSD_ped2/Train256/save_Reverse/'
Text_dir = 'S:/UCSD_ped2/Train256/'
Image_path = [os.path.join(Image_dir, i) for i in os.listdir(Image_dir)]
Label_path = [os.path.join(Label_dir, i) for i in os.listdir(Label_dir)]


def read_and_load(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = image / 255.
    return image


def recover(img):
    max_value = np.max(img)
    img = (255. / max_value) * img
    return img


def build_generator():
    # Encoder:
    input = tf.keras.Input(shape=[L_node, W_node, Channel * 1])
    # input: (None, 256, 256, 2)
    e1 = layers.Conv2D(64, 5, 2, 'same')(input)
    # e1: (None, 128, 128, 64)
    e2 = layers.LeakyReLU(0.2)(e1)
    e2 = layers.Conv2D(64, 3, 1, 'same', activation='relu')(e2)
    e2 = layers.Conv2D(128, 5, 2, 'same')(e2)  # Downsampling
    e2 = layers.BatchNormalization()(e2)
    # e2: (None, 64, 64, 128)
    e3 = layers.LeakyReLU(0.2)(e2)
    e3 = layers.Conv2D(128, 3, 1, 'same', activation='relu')(e3)
    e3 = layers.Conv2D(256, 5, 2, 'same')(e3)  # Downsampling
    e3 = layers.BatchNormalization()(e3)
    # e3: (None, 32, 32, 256)
    e4 = layers.LeakyReLU(0.2)(e3)
    e4 = layers.Conv2D(256, 3, 1, 'same', activation='relu')(e4)
    e4 = layers.Conv2D(512, 5, 2, 'same')(e4)  # Downsampling
    e4 = layers.BatchNormalization()(e4)
    # e4: (None, 16, 16, 512)
    e5 = layers.LeakyReLU(0.2)(e4)
    e5 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(e5)
    e5 = layers.Conv2D(512, 5, 2, 'same')(e5)  # Downsampling
    e5 = layers.BatchNormalization()(e5)
    # e5: (None, 8, 8, 512)
    e6 = layers.LeakyReLU(0.2)(e5)
    e6 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(e6)
    e6 = layers.Conv2D(512, 5, 2, 'same')(e6)  # Dowmsampling
    e6 = layers.BatchNormalization()(e6)
    # e6: (None, 4, 4, 512)
    e7 = layers.LeakyReLU(0.2)(e6)
    e7 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(e7)
    e7 = layers.Conv2D(512, 5, 2, 'same')(e7)  # Downsampling
    e7 = layers.BatchNormalization()(e7)
    # e7: (None, 2, 2, 512)
    e8 = layers.LeakyReLU(0.2)(e7)
    e8 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(e8)
    e8 = layers.Conv2D(512, 5, 2, 'same')(e8)  # Downsampling
    e8 = layers.BatchNormalization()(e8)
    # e8: (None, 1, 1, 512)
    # Decoder:
    d1 = layers.Activation('relu')(e8)
    d1 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(d1)
    d1 = layers.Conv2DTranspose(512, 5, 2, 'same')(d1)  # Upsampling
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.Dropout(0.5)(d1)
    d1 = layers.concatenate([d1, e7], 3)
    # d1: (None, 2, 2, 512*2)
    d2 = layers.Activation('relu')(d1)
    d2 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(d2)
    d2 = layers.Conv2DTranspose(512, 5, 2, 'same')(d2)  # Upsampling
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.Dropout(0.5)(d2)
    d2 = layers.concatenate([d2, e6], 3)
    # d2: (None, 4, 4, 512*2)
    d3 = layers.Activation('relu')(d2)
    d3 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(d3)
    d3 = layers.Conv2DTranspose(512, 5, 2, 'same')(d3)  # Upsampling
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.Dropout(0.5)(d3)
    d3 = layers.concatenate([d3, e5], 3)
    # d3: (None, 8, 8, 512*2)
    d4 = layers.Activation('relu')(d3)
    d4 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(d4)
    d4 = layers.Conv2DTranspose(512, 5, 2, 'same')(d4)  # Upsampling
    d4 = layers.BatchNormalization()(d4)
    d4 = layers.Dropout(0.5)(d4)
    d4 = layers.concatenate([d4, e4], 3)
    # d4: (None, 16, 16, 512*2)
    d5 = layers.Activation('relu')(d4)
    d5 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(d5)
    d5 = layers.Conv2DTranspose(256, 5, 2, 'same')(d5)  # Upsampling
    d5 = layers.BatchNormalization()(d5)
    d5 = layers.Dropout(0.5)(d5)
    d5 = layers.concatenate([d5, e3], 3)
    # d5: (None, 32, 32, 256*2)
    d6 = layers.Activation('relu')(d5)
    d6 = layers.Conv2D(256, 3, 1, 'same', activation='relu')(d6)
    d6 = layers.Conv2DTranspose(128, 5, 2, 'same')(d6)  # Upsampling
    d6 = layers.BatchNormalization()(d6)
    d6 = layers.Dropout(0.5)(d6)
    d6 = layers.concatenate([d6, e2], 3)
    # d6: (None, 64, 64, 128*2)
    d7 = layers.Activation('relu')(d6)
    d7 = layers.Conv2D(128, 3, 1, 'same', activation='relu')(d7)
    d7 = layers.Conv2DTranspose(64, 5, 2, 'same')(d7)  # Upsampling
    d7 = layers.BatchNormalization()(d7)
    d7 = layers.Dropout(0.5)(d7)
    d7 = layers.concatenate([d7, e1], 3)
    # d7: (None, 128, 128, 64*2)
    d8 = layers.Activation('relu')(d7)
    d8 = layers.Conv2D(64, 3, 1, 'same', activation='relu')(d8)
    d8 = layers.Conv2DTranspose(1, 5, 2, 'same')(d8)  # Upsampling
    d8 = layers.Activation('tanh')(d8)
    # d8: (None, 256, 256, 1)
    output = d8
    Model = tf.keras.Model(input, output)
    return Model


def build_discriminator():
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


def build_combined(g, d):
    Model = tf.keras.Sequential()
    Model.add(g)
    d.trainable = False
    Model.add(d)
    return Model


def train():
    k = 0

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # Slice the input image path string, get a dataset about string
    path_image_ds = tf.data.Dataset.from_tensor_slices(Image_path)
    # Load all images into dataset
    image_ds = path_image_ds.map(read_and_load, num_parallel_calls=AUTOTUNE)
    # Slice the input label path string, get a dataset about string
    path_label_ds = tf.data.Dataset.from_tensor_slices(Label_path)
    # Load all labels into dataset
    label_ds = path_label_ds.map(read_and_load, num_parallel_calls=AUTOTUNE)
    # Pack images and labels together
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    # Shuffle, Repeat, Batch operations
    ds = image_label_ds.shuffle(buffer_size=Buffer_size)
    ds = ds.repeat()
    ds = ds.batch(Batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate_gen, beta_1=Beta_1, beta_2=Beta_2, epsilon=E)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate_disc, beta_1=Beta_1, beta_2=Beta_2, epsilon=E)

    # Build Discriminator
    discriminator = build_discriminator()
    discriminator.build(input_shape=(Batch_size, L_node, W_node, Channel * 1))

    # Build Generator
    generator = build_generator()
    generator.build(input_shape=(Batch_size, L_node, W_node, Channel * 2))
    generator.summary()

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    for epoch in range(Training_steps):
        for data, label in ds:
            with tf.GradientTape() as Tape:
                gen_input = label
                # gen_input = layers.concatenate([data, label], 3)
                gen_output = generator(gen_input, training=False)
                disc_input_real = layers.concatenate([data, label], 3)
                # disc_input_real = data
                disc_input_fake = layers.concatenate([gen_output, label], 3)
                # disc_input_fake = gen_output
                # if k != 0:
                #     print(1, disc_fake[0, ])
                disc_real = discriminator(disc_input_real, training=True)
                disc_fake = discriminator(disc_input_fake, training=True)
                d_loss_real = cross_entropy(tf.ones_like(disc_real) * 0.9, disc_real)
                d_loss_fake = cross_entropy(tf.zeros_like(disc_fake) + 0.1, disc_fake)
                d_loss = d_loss_real + d_loss_fake

            d_gradients = Tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

            with tf.GradientTape() as Tape:
                gen_input = label
                # gen_input = layers.concatenate([data, label], 3)
                gen_output = generator(gen_input, training=True)
                disc_input_fake = layers.concatenate([gen_output, label], 3)
                # disc_input_fake = gen_output
                disc_fake = discriminator(disc_input_fake, training=False)
                # print(2, disc_fake[0, ])
                g_loss_entropy = Lambda_entropy * cross_entropy(tf.ones_like(disc_fake) * 0.9, disc_fake)
                g_loss_l1 = Lambda_l1 * tf.reduce_mean(tf.abs(gen_output - data))
                g_loss_ssim = Lambda_ssim * tf.reduce_mean(1 - tf.image.ssim(gen_output, data, max_val=1))
                # g_loss_l1_c1 = tf.reduce_mean(tf.abs(gen_output[:, :, :, 0] - data[:, :, :, 0]))
                # g_loss_l1_c2 = tf.reduce_mean(tf.abs(gen_output[:, :, :, 1] - data[:, :, :, 1]))
                # g_loss_l1_c3 = tf.reduce_mean(tf.abs(gen_output[:, :, :, 2] - data[:, :, :, 2]))
                # g_loss_l1 = Lambda * (g_loss_l1_c1 + g_loss_l1_c2 + g_loss_l1_c2 + g_loss_l1_c3)
                g_loss = g_loss_entropy + g_loss_l1 + g_loss_ssim

            g_gradients = Tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

            if k % 10 == 0:
                # print("Step:{}, Generator Loss:{:.4f}, L1 Loss:{:.4f}, SSIM Loss:{:.4f}, Discriminator Loss:{:.4f}".format(k, g_loss, g_loss_l1 / Lambda, g_loss_ssim / Lambda1, d_loss))
                print("Step:{} Generator Loss:{:.4f} L1 Loss:{:.4f} SSIM Loss:{:.4f} Discriminator Loss:{:.4f}".format(k, g_loss, g_loss_l1 / Lambda_l1, g_loss_ssim / Lambda_ssim, d_loss))
                # print("Step:{} Generator Loss:{:.4f} SSIM Loss:{:.4f} Discriminator Loss:{:.4f}".format(l, g_loss, g_loss_ssim / Lambda_ssim, d_loss)
                output_save = np.reshape(gen_output[0], newshape=[L_node, W_node])
                cv2.imwrite(Output_dir + str(k) + '.jpg', output_save * 255.)

            if k % 100 == 0:
                generator.save(Save_dir + str(k) + 'Gmodel' + '.h5')
                discriminator.save(Save_dir + str(k) + 'Dmodel' + '.h5')

            k = k + 1


def main(argv=None):
    train()


if __name__ == '__main__':
    main()
