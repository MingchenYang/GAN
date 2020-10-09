import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

L_node = 192
W_node = 256
Channel = 1


def Generator():
    # Encoder:
    input = tf.keras.Input(shape=[L_node, W_node, Channel * 2])
    # input: (None, 192, 256, 2)
    e1 = layers.Conv2D(64, 5, 2, 'same')(input)
    # e1: (None, 96, 128, 64)
    e2 = layers.LeakyReLU(0.2)(e1)
    e2 = layers.Conv2D(64, 3, 1, 'same', activation='relu')(e2)
    e2 = layers.Conv2D(128, 5, 2, 'same')(e2)  # Downsampling
    e2 = layers.BatchNormalization()(e2)
    # e2: (None, 48, 64, 128)
    e3 = layers.LeakyReLU(0.2)(e2)
    e3 = layers.Conv2D(128, 3, 1, 'same', activation='relu')(e3)
    e3 = layers.Conv2D(256, 5, 2, 'same')(e3)  # Downsampling
    e3 = layers.BatchNormalization()(e3)
    # e3: (None, 24, 32, 256)
    e4 = layers.LeakyReLU(0.2)(e3)
    e4 = layers.Conv2D(256, 3, 1, 'same', activation='relu')(e4)
    e4 = layers.Conv2D(512, 5, 2, 'same')(e4)  # Downsampling
    e4 = layers.BatchNormalization()(e4)
    # e4: (None, 12, 16, 512)
    e5 = layers.LeakyReLU(0.2)(e4)
    e5 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(e5)
    e5 = layers.Conv2D(512, 5, 2, 'same')(e5)  # Downsampling
    e5 = layers.BatchNormalization()(e5)
    # e5: (None, 6, 8, 512)
    e6 = layers.LeakyReLU(0.2)(e5)
    e6 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(e6)
    e6 = layers.Conv2D(512, 5, 2, 'same')(e6)  # Dowmsampling
    e6 = layers.BatchNormalization()(e6)
    # e6: (None, 3, 4, 512)
    '''
    e7 = layers.LeakyReLU(0.2)(e6)
    e7 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(e7)
    e7 = layers.Conv2D(512, 5, 2, 'valid')(e7)  # Downsampling
    e7 = layers.BatchNormalization()(e7)
    # e7: (None, 1, 2, 512)
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
    d2 = layers.Activation('relu')(d2)
    d2 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(d2)
    d2 = layers.Conv2DTranspose(512, 5, 2, 'same')(d2)  # Upsampling
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.Dropout(0.5)(d2)
    d2 = layers.concatenate([d2, e6], 3)
    # d2: (None, 2, 4, 512*2)
    '''
    d3 = layers.Activation('relu')(e6)
    d3 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(d3)
    d3 = layers.Conv2DTranspose(512, 5, 2, 'same')(d3)  # Upsampling
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.Dropout(0.5)(d3)
    d3 = layers.concatenate([d3, e5], 3)
    # d3: (None, 6, 8, 512*2)
    d4 = layers.Activation('relu')(d3)
    d4 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(d4)
    d4 = layers.Conv2DTranspose(512, 5, 2, 'same')(d4)  # Upsampling
    d4 = layers.BatchNormalization()(d4)
    d4 = layers.Dropout(0.5)(d4)
    d4 = layers.concatenate([d4, e4], 3)
    # d4: (None, 12, 16, 512*2)
    d5 = layers.Activation('relu')(d4)
    d5 = layers.Conv2D(512, 3, 1, 'same', activation='relu')(d5)
    d5 = layers.Conv2DTranspose(256, 5, 2, 'same')(d5)  # Upsampling
    d5 = layers.BatchNormalization()(d5)
    d5 = layers.Dropout(0.5)(d5)
    d5 = layers.concatenate([d5, e3], 3)
    # d5: (None, 24, 32, 256*2)
    d6 = layers.Activation('relu')(d5)
    d6 = layers.Conv2D(256, 3, 1, 'same', activation='relu')(d6)
    d6 = layers.Conv2DTranspose(128, 5, 2, 'same')(d6)  # Upsampling
    d6 = layers.BatchNormalization()(d6)
    d6 = layers.Dropout(0.5)(d6)
    d6 = layers.concatenate([d6, e2], 3)
    # d6: (None, 48, 64, 128*2)
    d7 = layers.Activation('relu')(d6)
    d7 = layers.Conv2D(128, 3, 1, 'same', activation='relu')(d7)
    d7 = layers.Conv2DTranspose(64, 5, 2, 'same')(d7)  # Upsampling
    d7 = layers.BatchNormalization()(d7)
    d7 = layers.Dropout(0.5)(d7)
    d7 = layers.concatenate([d7, e1], 3)
    # d7: (None, 96, 128, 64*2)
    d8 = layers.Activation('relu')(d7)
    d8 = layers.Conv2D(64, 3, 1, 'same', activation='relu')(d8)
    d8 = layers.Conv2DTranspose(1, 5, 2, 'same')(d8)  # Upsampling
    d8 = layers.Activation('tanh')(d8)
    # d8: (None, 192, 256, 1)
    output = d8
    model = tf.keras.Model(input, output)
    return model


def train():
    generator = Generator()
    generator.summary()


def main(argv=None):
    train()


if __name__=='__main__':
    main()