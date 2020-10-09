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
Learning_rate_gen = 0.0002
Learning_rate_disc = 0.0002
Beta_1 = 0.5
Beta_2 = 0.999
E = 1e-08
Lambda = 10
Lambda1 = 30
Training_steps = 5

Image_dir = 'S:/UCSD_ped2/Train256/training/'
Mosaic_dir = 'S:/UCSD_ped2/Train256/training_Mosaic/'
Label_dir = 'S:/UCSD_ped2/Train256/label/'
Output_dir = 'S:/UCSD_ped2/Train256/result_Mosaic/'
Save_path = 'S:/UCSD_ped2/Train256/save_Mosaic/'
Image_path = [os.path.join(Image_dir, i) for i in os.listdir(Image_dir)]
Mosaic_path = [os.path.join(Mosaic_dir, i) for i in os.listdir(Mosaic_dir)]
Label_path = [os.path.join(Label_dir, i) for i in os.listdir(Label_dir)]


def read_and_load(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=Channel)
    image = tf.cast(image, tf.float32)
    image = image / 255.
    return image


def Generator():
    # Encoder:
    input = tf.keras.Input(shape=[L_node, W_node, Channel * 2])
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
    model = tf.keras.Model(input, output)
    return model


def Discriminator():
    input = tf.keras.Input(shape=[L_node, W_node, Channel*3])
    # input: (None, 256, 256, 1)
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
    output = h4
    model = tf.keras.Model(input, output)
    return model


def train():
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Slice the input image path string, get a dataset about string
    path_image_ds = tf.data.Dataset.from_tensor_slices(Image_path)
    # Load all images into dataset
    image_ds = path_image_ds.map(read_and_load, num_parallel_calls=AUTOTUNE)

    # Slice the input label path string, get a dataset about string
    path_label_ds = tf.data.Dataset.from_tensor_slices(Label_path)
    # Load all labels into dataset
    label_ds = path_label_ds.map(read_and_load, num_parallel_calls=AUTOTUNE)

    # Slice the input mosaic path string, get a datset about string
    path_mosaic_ds = tf.data.Dataset.from_tensor_slices(Mosaic_path)
    # Load all Mosaic images into dataset
    mosaic_ds = path_mosaic_ds.map(read_and_load, num_parallel_calls=AUTOTUNE)

    # Pack images and labels together
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds, mosaic_ds))
    # Shuffle, Repeat, Batch operations
    ds = image_label_ds.shuffle(buffer_size=Buffer_size)
    ds = ds.repeat()
    ds = ds.batch(Batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    generator = Generator()
    generator.build(input_shape=(Batch_size, L_node, W_node, Channel * 2))
    generator.summary()

    discriminator = Discriminator()
    discriminator.build(input_shape=(Batch_size, L_node, W_node, Channel * 3))
    discriminator.summary()

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate_gen, beta_1=Beta_1, beta_2=Beta_2, epsilon=E)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate_disc, beta_1=Beta_1, beta_2=Beta_2, epsilon=E)

    k = 0
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    for epoch in range(Training_steps):
        for data, label, mosaic in ds:
            with tf.GradientTape() as Tape:
                # gen_input = label
                gen_input = layers.concatenate([mosaic, label], 3)
                gen_output = generator(gen_input, training=False)
                disc_input_real = layers.concatenate([data, mosaic, label], 3)
                disc_input_fake = layers.concatenate([gen_output, mosaic, label], 3)
                disc_real = discriminator(disc_input_real, training=True)
                disc_fake = discriminator(disc_input_fake, training=True)
                d_loss_real = cross_entropy(tf.ones_like(disc_real) * 0.9, disc_real)
                d_loss_fake = cross_entropy(tf.zeros_like(disc_fake) + 0.1, disc_fake)
                d_loss = d_loss_real + d_loss_fake
            d_gradients = Tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

            with tf.GradientTape() as Tape:
                # gen_input = label
                gen_input = layers.concatenate([mosaic, label], 3)
                gen_output = generator(gen_input, training=True)
                disc_input_fake = layers.concatenate([gen_output, mosaic, label], 3)
                disc_fake = discriminator(disc_input_fake, training=False)
                g_loss_entropy = cross_entropy(tf.ones_like(disc_fake) * 0.9, disc_fake)
                g_loss_l1 = Lambda * tf.reduce_mean(tf.abs(gen_output - data))
                g_loss_ssim = Lambda1 * tf.reduce_mean(1 - tf.image.ssim(gen_output, data, max_val=1))
                g_loss = g_loss_entropy + g_loss_l1 + g_loss_ssim
            g_gradients = Tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

            if k % 100 == 0:
                # print("Step:{} Generator Loss:{:.4f} Discriminator Loss:{:.4f}".format(k, g_loss, d_loss))
                # print("Step:{} Generator Loss:{:.4f} L1 Loss:{:.4f} Discriminator Loss:{:.4f}".format(k, g_loss, g_loss_l1 / Lambda, d_loss))
                print("Step:{} Generator Loss:{:.4f} L1 Loss:{:.4f} SSIM Loss:{:.4f} Discriminator Loss:{:.4f}".format(k, g_loss, g_loss_l1 / Lambda, g_loss_ssim / Lambda1, d_loss))
                output_save = np.reshape(gen_output[0], newshape=[L_node, W_node])
                cv2.imwrite(Output_dir + str(k) + '.jpg', output_save * 255.)
            if k % 1000 == 0:
                generator.save(Save_path + str(k) + 'Gmodel' + '.h5')
                discriminator.save(Save_path + str(k) + 'Dmodel' + '.h5')

            k = k + 1


def main(argv=None):
    train()


if __name__=='__main__':
    main()