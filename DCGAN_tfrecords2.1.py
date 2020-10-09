import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

Training_steps = 500

Batch_size = 128
Noise_node = 100
Input_node = 64

Learning_rate_gen = 0.0002
Learning_rate_disc = 0.0002

Input_dir = 'S:/face/faces_64/'
Input_path = [os.path.join(Input_dir, i) for i in os.listdir(Input_dir)]
Output_dir = 'S:/face/result/'


def Generator():
    model = tf.keras.Sequential()
    # (100)
    model.add(tf.keras.layers.Dense(1024 * 4 * 4, input_dim=100))
    # (1024*4*4)
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Reshape(target_shape=(4, 4, 1024)))
    # (4, 4, 1024)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2DTranspose(256, 3, 2, 'same'))
    # (8, 8, 256)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2DTranspose(128, 3, 2, 'same'))
    # (16, 16, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2DTranspose(64, 3, 2, 'same'))
    # (32, 32, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2DTranspose(3, 3, 2, 'same'))
    # (64, 64, 3)
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('tanh'))
    model.summary()
    return model


def Discriminator():
    model = tf.keras.Sequential()
    #(64, 64)
    model.add(tf.keras.layers.Conv2D(64, 3, 2, 'same', input_shape=[64, 64, 3]))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Conv2D(128, 3, 2, 'same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Conv2D(256, 3, 2, 'same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Conv2D(1024, 3, 2, 'same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(1))
    model.summary()
    return model


'''
class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        # z:[b,100]-->[b,3*3*512]-->[b,3,3,512]-->[b,64,64,3]
        self.fc=keras.layers.Dense(3*3*512)

        self.conv1=keras.layers.Conv2DTranspose(256,3,3,'valid')  # 反卷积
        self.bn1=keras.layers.BatchNormalization()

        self.conv2=keras.layers.Conv2DTranspose(128,5,2,'valid')
        self.bn2=keras.layers.BatchNormalization()

        self.conv3=keras.layers.Conv2DTranspose(3,4,3,'valid')

    def call(self, inputs, training=None, mask=None):
        # [z,100]-->[z,3*3*512]
        x=self.fc(inputs)
        x=tf.reshape(x,[-1,3,3,512])
        x=tf.nn.leaky_relu(x)

        x=tf.nn.leaky_relu(self.bn1(self.conv1(x),training=training))
        x=tf.nn.leaky_relu(self.bn2(self.conv2(x),training=training))
        x=self.conv3(x)
        x=tf.tanh(x)

        return x

class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        # [b,64,64,3]-->[b,1]
        self.conv1=keras.layers.Conv2D(64,5,3,'valid')

        self.conv2=keras.layers.Conv2D(128,5,3,'valid')
        self.bn2=keras.layers.BatchNormalization()

        self.conv3=keras.layers.Conv2D(256,5,3,'valid')
        self.bn3=keras.layers.BatchNormalization()

        # [b,h,w,c]-->[b,-1]
        self.flatten=keras.layers.Flatten()
        # [b,-1]-->[b,1]
        self.fc=keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x=tf.nn.leaky_relu(self.conv1(inputs))
        x=tf.nn.leaky_relu(self.bn2(self.conv2(x),training=training))
        x=tf.nn.leaky_relu(self.bn3(self.conv3(x),training=training))

        x=self.flatten(x)
        logits=self.fc(x)

        return logits
'''

'''
def Generator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(3 * 3 * 512, input_dim=100))
    model.add(tf.keras.layers.Reshape(target_shape=[3, 3, 512]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(256, 3, 3, 'valid'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(128, 5, 2, 'valid'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(3, 4, 3, 'valid'))
    model.add(tf.keras.layers.Activation('tanh'))
    model.summary()
    return model


def Discriminator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, 5, 3, 'valid', input_shape=[64, 64, 3]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(128, 5, 3, 'valid'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(256, 5, 3, 'valid'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    model.summary()
    return model
'''


def read_and_load(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = image / 127.5 - 1
    return image


def train():
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices(Input_path)
    image_ds = ds.map(read_and_load)
    ds = image_ds.shuffle(buffer_size=10000)
    ds = ds.repeat()
    ds = ds.batch(Batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    generator = Generator()
    discriminator = Discriminator()
    generator.build(input_shape=(Batch_size, Noise_node))
    # generator.summary()
    discriminator.build(input_shape=(Batch_size, Input_node, Input_node, 3))
    # discriminator.summary()

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate_gen, beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate_disc, beta_1=0.7)

    k = 0
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    for epoch in range(Training_steps):
        for data in ds:
            k = k + 1
            noise = tf.random.normal(shape=(Batch_size, Noise_node), mean=0.0, stddev=1.0)

            with tf.GradientTape() as Tape:
                gen_output = generator(noise, training=True)
                disc_fake = discriminator(gen_output, training=True)
                disc_real = discriminator(data, training=True)
                d_loss_fake = cross_entropy(tf.zeros_like(disc_fake), disc_fake)
                d_loss_real = cross_entropy(tf.ones_like(disc_real) * 0.9, disc_real)
                d_loss = d_loss_real + d_loss_fake
            d_gradients = Tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

            with tf.GradientTape() as Tape:
                gen_output = generator(noise, training=True)
                disc_fake = discriminator(gen_output, training=True)
                g_loss = cross_entropy(tf.ones_like(disc_fake * 0.9), disc_fake)
            g_gradients = Tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

            if k % 100 == 0:
                print("Step:{} Generator Loss:{:.4f} Discriminator Loss:{:.4f}".format(k, g_loss, d_loss))

            if k % 100 == 0:
                test_noise = tf.random.normal([Batch_size, Noise_node])
                test_output = generator(test_noise, training=False)
                test_output = (test_output + 1) * 0.5
                for i in range(64):
                    plt.subplot(8, 8, i + 1)
                    plt.imshow(test_output[i])
                Output_path = os.path.join(Output_dir, str(k))
                plt.savefig(Output_path + '.jpg')
                plt.close()


def main(argv=None):
    train()


if __name__ == '__main__':
    main()
