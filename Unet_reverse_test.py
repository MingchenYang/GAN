import os
import cv2
import numpy as np
import tensorflow as tf

L_node = 256
W_node = 256
Channel = 1

Image_dir = 'S:/UCSD_ped2/Test256/label_dis_removal/'  # Image dataset in reverse file is the input of generator
Label_dir = 'S:/UCSD_ped2/Test256/training_removal/'  # Label dataset in reverse is the real image
Output_dir = 'S:/UCSD_ped2/Test256/Unet_Reverse_dis_removal_test/'
Save_path = 'S:/UCSD_ped2/Train256/save_Reverse_dis_removal/100000Gmodel.h5'


def read_and_load(path):
    img = cv2.imread(path, 1)
    img = tf.reshape(img, shape=[1, L_node, W_node, Channel * 3])
    img = tf.cast(img, tf.float32)
    img = img / 255.
    return img


def train():
    k = 1

    Model = tf.keras.models.load_model(Save_path)
    Model.summary()

    for name in os.listdir(Image_dir):
        print(name)
        Image_path = Image_dir + name
        Label_path = Label_dir + name
        data = read_and_load(Image_path)
        label = read_and_load(Label_path)

        input = label
        output = Model(input, training=False)

        l1_loss = tf.reduce_mean(tf.abs(output - data))
        ssim_loss = tf.reduce_mean(1 - tf.image.ssim(output, data, max_val=1))
        psnr_loss = tf.reduce_mean(tf.image.psnr(output, data, max_val=1))
        Text_path = Output_dir + 'result.txt'
        with open(Text_path, 'a') as File:
            print('Num:{}, L1 Loss:{:.4f}, SSIM Loss:{:.4f}, PSNR loss:{:.4f}'.format(k, l1_loss, ssim_loss, psnr_loss),
                  file=File)

        output = np.reshape(output, newshape=[L_node, W_node, Channel * 3])
        output = output * 255.
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        Output_path = Output_dir + name
        cv2.imwrite(Output_path, output)

        k = k + 1


def main(argv=None):
    train()


if __name__=='__main__':
    main()