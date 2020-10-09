import tensorflow as tf
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
import numpy as np

Training_Steps = 10000

Batch_size = 64
Noise_node = 100
Image_node = 64

Learning_rate_gen = 0.0001
Learning_rate_disc = 0.0001
Min_after_dequeue = 1000

Model_path = "S:/face/save"
Model_name = "faces_64.ckpt"
Input_dir = "S:/face/faces_64/"
Input_path = [os.path.join(Input_dir, i) for i in os.listdir(Input_dir)]

#dataset = tf.data.TFRecordDataset(Input_dir)
#feature_description = {
#    'image_raw': tf.io.FixedLenFeature([], tf.string)
#}


def read_and_decode(example_string):
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    image = tf.io.decode_jpeg(feature_dict['image_raw'])
    image = tf.reshape(image, [Image_node, Image_node, 3])
    image = tf.cast(image, dtype = 'float32')
    image = (image - 127.5)/127.5
    return image


def read_and_load(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels = 3)
    image = tf.cast(image, tf.float32)
    image = image/127.5 - 1
    return image


def generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(1024, input_dim = 100))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(1024*4*4))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((4, 4, 1024), input_shape = (1024*4*4, )))
    model.add(layers.UpSampling2D(size = (2, 2)))
    model.add(layers.Conv2D(256, (3, 3), padding = 'same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.UpSampling2D(size = (2, 2)))
    model.add(layers.Conv2D(128, (3, 3), padding = 'same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.UpSampling2D(size = (2, 2)))
    model.add(layers.Conv2D(64, (3, 3), padding = 'same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.UpSampling2D(size = (2, 2)))
    model.add(layers.Conv2D(3, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('tanh'))
    model.summary()
    return model


def discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (3, 3), padding = 'same', input_shape = (64, 64, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPool2D(pool_size = (2, 2)))
    model.add(layers.Conv2D(128, (3, 3), padding = 'same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPool2D(pool_size = (2, 2)))
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(1024, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(rate = 0.1))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))
    model.summary()
    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)


def discriminator_loss(disc_fake, disc_real):
    disc_loss_fake = cross_entropy(tf.zeros_like(disc_fake), disc_fake)
    disc_loss_real = cross_entropy(tf.ones_like(disc_real), disc_real)
    return disc_loss_fake + disc_loss_real


def generator_loss(disc_fake):
    gen_loss = cross_entropy(tf.ones_like(disc_fake), disc_fake)
    return gen_loss


generator = generator_model()
discriminator = discriminator_model()


def train_step(batch):
    noise = tf.random.normal([Batch_size, Noise_node])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        disc_real = discriminator(batch, training = True)

        gen_output = generator(noise, training = True)
        disc_fake = discriminator(gen_output, training = True)

        gen_loss = generator_loss(disc_fake)
        disc_loss = discriminator_loss(disc_fake, disc_real)

    gradient_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradient_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_opt = tf.keras.optimizers.Adam(Learning_rate_gen)
    discriminator_opt = tf.keras.optimizers.Adam(Learning_rate_disc)
    generator_opt.apply_gradients(zip(gradient_gen, generator.trainable_variables))
    discriminator_opt.apply_gradients(zip(gradient_disc, discriminator.trainable_variables))
    return gen_loss, disc_loss

'''
def input_data():
    dataset = tf.data.TFRecordDataset(Input_dir)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size = 512)
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }
    dataset = dataset.map(read_and_decode)
    dataset = dataset.prefetch(buffer_size = 100)
    dataset = dataset.batch(batch_size = Batch_size)
    return dataset
'''
AUTOTUNE = tf.data.experimental.AUTOTUNE
dataset = tf.data.Dataset.from_tensor_slices(Input_path)
image_dataset = dataset.map(read_and_load)
dataset = image_dataset.shuffle(buffer_size = 10000)
dataset = dataset.repeat()
dataset = dataset.batch(Batch_size)
dataset = dataset.prefetch(buffer_size = AUTOTUNE)


k = 0
for steps in range(Training_Steps):
    for data in dataset:
        #for i in range(64):
        #    image = (data + 1)*0.5
        #    plt.subplot(8, 8, i+1)
        #    plt.imshow(image[i])
        #plt.show()
        gen_loss, disc_loss = train_step(data)
        k = k + 1
        if k%1 == 0:
            print("Step:{} Generator Loss:{:.10f} Discriminator Loss:{:.10f}".format(k, gen_loss, disc_loss))
        if k%10 == 0:
            test_noise = tf.random.normal([Batch_size,Noise_node])
            test_output = generator(test_noise)
            test_output = (test_output + 1)*0.5
            for i in range(Batch_size):
                plt.subplot(8, 8, i+1)
                plt.imshow(test_output[i])
            plt.show()