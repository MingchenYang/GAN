import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras

Training_steps = 10000

Batch_size = 64
Noise_node = 100
Input_node = 28

Learning_rate_gen = 0.0002
Learning_rate_disc = 0.0002

Output_dir = 'S:/face/mnist_result/'

'''
def generator_model():
    model = tf.keras.Sequential()
    #(100)
    model.add(tf.keras.layers.Dense(1024, input_dim=100))
    #(1024)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Reshape(target_shape=(1, 1, 1024)))
    #(1, 1, 1024)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(7, 7), padding='same'))
    #(7, 7, 256)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    #(14, 14, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3),strides=(2, 2),padding='same'))
    #(28, 28, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(5, 5), strides=(1, 1), padding='same'))
    #(28, 28, 1)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('tanh'))
    #model.summary()
    return model


def discriminator_model():
    model = tf.keras.Sequential()
    #(28, 28)
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    #(28, 28, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    #(14, 14, 64)
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    #(14, 14, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    #(7, 7, 128)
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    #(7, 7, 256)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(7, 7), strides=(7, 7), padding='same'))
    #(1, 1, 256)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(rate=0.99))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))
    #model.summary()
    return model
'''


class Generator(keras.Model):

    def __init__(self):
        super(Generator, self).__init__()

        self.n_f = 512
        self.n_k = 4

        # input z vector is [None, 100]
        self.dense1 = keras.layers.Dense(3 * 3 * self.n_f)
        self.conv2 = keras.layers.Conv2DTranspose(self.n_f // 2, 3, 2, 'valid')
        self.bn2 = keras.layers.BatchNormalization()
        self.conv3 = keras.layers.Conv2DTranspose(self.n_f // 4, self.n_k, 2, 'same')
        self.bn3 = keras.layers.BatchNormalization()
        self.conv4 = keras.layers.Conv2DTranspose(1, self.n_k, 2, 'same')
        return

    def call(self, inputs, training=None):
        # [b, 100] => [b, 3, 3, 512]
        x = tf.nn.leaky_relu(tf.reshape(self.dense1(inputs), shape=[-1, 3, 3, self.n_f]))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        x = tf.tanh(self.conv4(x))
        return x


class Discriminator(keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.n_f = 64
        self.n_k = 4

        # input image is [-1, 28, 28, 1]
        self.conv1 = keras.layers.Conv2D(self.n_f, self.n_k, 2, 'same')
        self.conv2 = keras.layers.Conv2D(self.n_f * 2, self.n_k, 2, 'same')
        self.bn2 = keras.layers.BatchNormalization()
        self.conv3 = keras.layers.Conv2D(self.n_f * 4, self.n_k, 2, 'same')
        self.bn3 = keras.layers.BatchNormalization()
        self.flatten4 = keras.layers.Flatten()
        self.dense4 = keras.layers.Dense(1)
        return

    def call(self, inputs, training=None):
        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        x = self.dense4(self.flatten4(x))
        return x


def train():
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32)/255.
    ds = tf.data.Dataset.from_tensor_slices(x_train)
    ds = ds.shuffle(buffer_size=10000)
    ds = ds.repeat()
    ds = ds.batch(batch_size=Batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    #generator = generator_model()
    #discriminator = discriminator_model()
    generator = Generator()
    discriminator = Discriminator()
    generator.build(input_shape=(Batch_size, Noise_node))
    generator.summary()
    discriminator.build(input_shape=(Batch_size, Input_node, Input_node, 1))
    discriminator.summary()

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate_gen)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate_disc)

    k = 0
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    for epoch in range(Training_steps):
        for data in ds:
            data = data*2.0 - 1.0
            data = tf.reshape(data, shape=(Batch_size, Input_node, Input_node, 1))
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

            if k%100 == 0:
                print("Step:{} Generator Loss:{:.4f} Discriminator Loss:{:.4f}".format(k, g_loss, d_loss))

            if k%1000 == 0:
                test_noise = tf.random.normal([Batch_size, Noise_node])
                test_output = generator(test_noise, training=False)
                test_output = (test_output + 1)*0.5
                test_output = tf.reshape(test_output, shape=(Batch_size, Input_node, Input_node))
                for i in range(Batch_size):
                    plt.subplot(8, 8, i+1)
                    plt.imshow(test_output[i])
                Output_path = os.path.join(Output_dir, str(k))
                plt.savefig(Output_path + '.jpg')
                plt.close()


def main(argv=None):
    train()

if __name__ == '__main__':
    main()