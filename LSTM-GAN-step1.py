import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

Buffer_size = 1000
Batch_size = 20
L_node = 256
W_node = 256
Channel = 1
Noise_node = 1000
Learning_rate_gen = 0.0001
Learning_rate_disc = 0.0001
Beta_1 = 0.5
Beta_2 = 0.999
E = 1e-08
Epoches = 2

Image_dir = 'S:/pix2pix optflow/Train256/training/'
Output_dir = 'S:/pix2pix optflow/Generator256/result2/'
Save_dir = 'S:/pix2pix optflow/Generator256/save2/'
Image_path = [os.path.join(Image_dir, i) for i in os.listdir(Image_dir)]


def read_and_load(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=Channel)
    image = tf.cast(image, tf.float32)
    image = image / 255.
    return image


def Generator():
    input = tf.keras.Input(shape=[Noise_node])

    l1 = layers.Dense(1024)(input)
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
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    path_image_ds = tf.data.Dataset.from_tensor_slices(Image_path)
    image_ds = path_image_ds.map(read_and_load, num_parallel_calls=AUTOTUNE)

    ds = image_ds.shuffle(buffer_size=Buffer_size)
    ds = ds.repeat()
    ds = ds.batch(Batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    generator = Generator()
    generator.summary()
    generator.build(input_shape=[Batch_size, Noise_node])
    discriminator = Discriminator()
    discriminator.summary()
    discriminator.build(input_shape=[Batch_size, L_node, W_node, Channel])

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate_gen, beta_1=Beta_1, beta_2=Beta_2, epsilon=E)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate_disc, beta_1=Beta_1, beta_2=Beta_2, epsilon=E)

    k = 0
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    for epoch in range(Epoches):
        for data in ds:
            noise = tf.random.normal(shape=[Batch_size, Noise_node], mean=0.0, stddev=1.0)

            with tf.GradientTape() as Tape:
                gen_output = generator(noise, training=True)
                disc_real = discriminator(data, training=True)
                disc_fake = discriminator(gen_output, training=True)
                d_loss_real = cross_entropy(tf.ones_like(disc_real * 0.9), disc_real)
                d_loss_fake = cross_entropy(tf.zeros_like(disc_fake) + 0.1, disc_fake)
                d_loss = d_loss_real + d_loss_fake
            d_gradients = Tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

            with tf.GradientTape() as Tape:
                gen_output = generator(noise, training=True)
                disc_fake = discriminator(gen_output, training=True)
                g_loss_entropy = cross_entropy(tf.ones_like(disc_fake * 0.9), disc_fake)
                g_loss_l1 = 100 * tf.reduce_mean(tf.abs(gen_output - data))
                g_loss = g_loss_entropy + g_loss_l1
            g_gradients = Tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

            if k % 100 == 0:
                print("Step:{} Generator Loss:{:.4f} {:.4f} Discriminator Loss:{:.4f}".format(k, g_loss, g_loss_entropy, d_loss))
                output_save = np.reshape(gen_output[1], newshape=[L_node, W_node])
                cv2.imwrite(Output_dir + str(k) + '.jpg', output_save * 255.)

            if k % 1000 == 0:
                generator.save(Save_dir + str(k) + 'Gmodel' + '.h5')

            k = k + 1


def main(argv=None):
    train()


if __name__ == '__main__':
    main()