import tensorflow as tf
import numpy as np
import os
import cv2

Training_steps = 10000

Batch_size = 32
Noise_node = 100
Input_node = 64

Learning_rate_gen = 0.02
Learning_rate_disc = 0.02

Input_dir = 'S:/face/faces_64/'
Input_path = [os.path.join(Input_dir, i) for i in os.listdir(Input_dir)]


def read_img(path):
    img = np.reshape(cv2.imread(path), newshape = [Input_node, Input_node, 3])
    return img


def get_batch(batch_size):
    n_batches = len(Input_path) // batch_size
    img_list = Input_path[:n_batches*batch_size]

    for i in range(n_batches):
        img_batch = img_list[i*batch_size:(i+1)*batch_size]
        img_output = np.zeros(shape = [batch_size, Input_node, Input_node, 3])
        for j, img in enumerate(img_batch):
            img_output[j] = read_img(img)
        yield img_output

def generator_model():
    model = tf.keras.Sequential()
    #(100)
    model.add(tf.keras.layers.Dense(1024*4*4, input_dim=100))
    #(1024*4*4)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Reshape(target_shape=(4, 4, 1024)))
    #(4, 4, 1024)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    #(8, 8, 256)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    #(16, 16, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3),strides=(2, 2),padding='same'))
    #(32, 32, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    #(64, 64, 3)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('tanh'))
    #model.summary()
    return model


def discriminator_model():
    model = tf.keras.Sequential()
    #(64, 64)
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))
    #model.summary()
    return model


def read_and_load(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = image/127.5 - 1
    return image


def train():
    generator = generator_model()
    discriminator = discriminator_model()
    generator.build(input_shape=(Batch_size, Noise_node))
    generator.summary()
    discriminator.build(input_shape=(Batch_size, Input_node, Input_node, 3))
    discriminator.summary()

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate_gen, beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate_disc, beta_1=0.5)

    k = 0
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    for epoch in range(Training_steps):
        for data in get_batch(Batch_size):
            k = k+1
            noise = tf.random.normal(shape=(Batch_size, Noise_node))

            with tf.GradientTape() as Tape:
                gen_output = generator(noise, training=True)
                disc_fake = discriminator(gen_output, training=True)
                disc_real = discriminator(data, training=True)
                d_loss_fake = cross_entropy(tf.zeros_like(disc_fake), disc_fake)
                d_loss_real = cross_entropy(tf.ones_like(disc_real)*0.9, disc_real)
                d_loss = d_loss_real + d_loss_fake
            d_gradients = Tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

            with tf.GradientTape() as Tape:
                gen_output = generator(noise, training=True)
                disc_fake = discriminator(gen_output, training=True)
                g_loss = cross_entropy(tf.ones_like(disc_fake*0.9), disc_fake)
            g_gradients = Tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

            if k%10 == 0:
                print("Step:{} Generator Loss:{:.4f} Discriminator Loss:{:.4f}".format(k, g_loss, d_loss))


def main(argv=None):
    train()

if __name__ == '__main__':
    main()