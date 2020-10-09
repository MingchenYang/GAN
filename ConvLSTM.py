import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

Batch_size = 2
Time_steps = Num_sequence = 5
L_node = 256
W_node = 256
Channel = 1
Output_dim = 100
Learning_rate = 0.0001
Beta_1 = 0.5
Beta_2 = 0.999
E = 1e-08
training_steps = 100000

File_path = 'S:/pix2pix optflow/Generator256/sequence/'
Output_dir = 'S:/pix2pix optflow/Generator256/ConvLSTM_result/'


def read_and_load(name):
    name = '%04d' % name
    File_name = File_path + str(name) + '.npz'  # S:/pix2pix optflow/Generator256/sequence/0001.npz

    Sequence_image = np.load(File_name)['image']
    Sequence_image = np.reshape(Sequence_image,
                                newshape=[1, Num_sequence, L_node, W_node, Channel])  # (1, 5, 256, 256, 1)
    Sequence_image = tf.cast(Sequence_image, tf.float32)
    Sequence_image = Sequence_image / 255.

    Sequence_label = np.load(File_name)['label']
    Sequence_label = np.reshape(Sequence_label, newshape=[1, L_node, W_node, Channel])  # (1, 256, 256, 1)
    Sequence_label = tf.cast(Sequence_label, tf.float32)
    Sequence_label = Sequence_label / 255.

    return Sequence_image, Sequence_label


def build_model():
    input = tf.keras.Input(shape=[Time_steps, L_node, W_node, Channel])

    l1 = layers.ConvLSTM2D(10, 3, 1, 'same', return_sequences=True)(input)
    l1 = layers.BatchNormalization()(l1)

    l2 = layers.ConvLSTM2D(20, 3, 1, 'same', return_sequences=True)(l1)
    l2 = layers.BatchNormalization()(l2)

    l3 = layers.ConvLSTM2D(20, 3, 1, 'same', return_sequences=True)(l2)
    l3 = layers.BatchNormalization()(l3)

    l4 = layers.ConvLSTM2D(10, 3, 1, 'same', return_sequences=False)(l3)
    l4 = layers.BatchNormalization()(l4)

    l5 = layers.Conv2D(1, 3, 1, 'same', activation='sigmoid')(l4)

    model = tf.keras.Model(input, l5)
    return model


def train():
    image_batch = []
    label_batch = []
    k = 0

    Model = build_model()
    Model.build(input_shape=[Batch_size, Time_steps, L_node, W_node, Channel])
    Model.summary()

    # optimizer = tf.keras.optimizers.Adadelta()
    optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate)

    for steps in range(training_steps):
        rand = random.sample(range(1, 6630), Batch_size)

        for i in range(Batch_size):
            image_sequence, label = read_and_load(rand[i])

            if i == 0:
                image_batch = image_sequence
                label_batch = label
            else:
                image_batch = layers.concatenate([image_batch, image_sequence], 0)  # (5, 5, 256, 256, 1)
                label_batch = layers.concatenate([label_batch, label], 0)  # (5, 256, 256, 1)

        with tf.GradientTape() as Tape:
            output = Model(image_batch, training=True)  # (5, 256, 256, 1)
            # loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(output, label_batch))
            loss1 = tf.reduce_mean(1 - tf.image.ssim(output, label_batch, max_val=1))
            loss2 = tf.reduce_mean(tf.abs(output - label_batch))
            loss = loss1 + loss2
        gradients = Tape.gradient(loss, Model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, Model.trainable_variables))

        if k % 10 == 0:
            print("Step:{} SSIM Loss:{:.4f} L1 Loss:{:.4f}".format(k, loss1, loss2))
            output_save = np.reshape(output[0], newshape=[L_node, W_node])
            cv2.imwrite(Output_dir + str(k) + '.jpg', output_save * 255.)

        k = k + 1


def main():
    train()


if __name__ == '__main__':
    main()