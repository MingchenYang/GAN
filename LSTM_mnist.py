import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

Batch_size = 128
Buffer_size = 1000
Units = 128
Time_steps = 28
Input_dim = 28
Output_dim = 10
Learning_rate = 0.001


def build_model():
    input = tf.keras.Input(shape=[Time_steps, Input_dim])
    l1 = layers.LSTM(Units)(input)
    l2 = layers.Dense(128)(l1)
    l3 = layers.Dense(Output_dim, activation='softmax')(l2)
    model = tf.keras.Model(input, l3)
    return model


def train():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, Output_dim)
    y_test = tf.keras.utils.to_categorical(y_test, Output_dim)

    Adam = tf.keras.optimizers.Adam(learning_rate=Learning_rate)

    model = build_model()
    model.summary()
    model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, batch_size=Batch_size, epochs=10)

    evaluate = model.evaluate(x=x_test, y=y_test)
    print(evaluate)


def main(argv=None):
    train()


if __name__ == '__main__':
    main()