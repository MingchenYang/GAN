import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

L_node = 256
W_node = 256
Channel = 1

Input_dir = 'S:/pix2pix optflow/Test256/label/'
Training_dir = 'S:/pix2pix optflow/Test256/training/'
Save_path = 'C:/Users/yangm90/Documents/pycharm transfer files/3.0/save/20000Gmodel.h5'
Output_dir = 'S:/pix2pix optflow/Test256/UGAN3.0_test/'


def read_and_load(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = tf.reshape(img, shape=[1, L_node, W_node, Channel])
    img = tf.cast(img, tf.float32)
    img = img / 255.
    return img


def train():
    k = 1

    Model = tf.keras.models.load_model(Save_path)
    Model.summary()

    for name in os.listdir(Input_dir):
        Input_path = Input_dir + name  # Label path
        Training_path = Training_dir + name  # Training path
        print(name)

        Label = read_and_load(Input_path)
        Image = read_and_load(Training_path)
        Output = Model(Label, training=False)

        l1_loss = tf.reduce_mean(tf.abs(Output - Image))
        ssim_loss = tf.reduce_mean(1 - tf.image.ssim(Output, Image, max_val=1))
        psnr_loss = tf.reduce_mean(tf.image.psnr(Output, Image, max_val=1))
        Text_path = Output_dir + 'result.txt'
        with open(Text_path, 'a') as File:
            print('Num:{}, L1 Loss:{:.4f}, SSIM Loss:{:.4f}, PSNR loss:{:.4f}'.format(k, l1_loss, ssim_loss, psnr_loss), file=File)

        Output = np.reshape(Output, newshape=[L_node, W_node])
        Output = Output * 255.
        Output_path = Output_dir + name
        cv2.imwrite(Output_path, Output)

        k = k + 1


def main(argv=None):
    train()


if __name__=='__main__':
    main()