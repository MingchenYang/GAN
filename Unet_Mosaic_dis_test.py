import os
import cv2
import numpy as np
import tensorflow as tf

L_node = 256
W_node = 256
Channel = 1

Image_dir = 'S:/UCSD_ped2/Test256/training/'
Mosaic_dir = 'S:/UCSD_ped2/Test256/training_Mosaic/'
Label_dir = 'S:/UCSD_ped2/Test256/label_dis_removal/'
Output_dir = 'S:/UCSD_ped2/Test256/Unet_Mosaic_dis_test/'
Save_path = 'S:/UCSD_ped2/Train256/save_Mosaic_dis/80000Gmodel.h5'
# Save_path = 'C:/Users/yangm90/Documents/pycharm transfer files/8.0/save_Mosaic/200000Gmodel.h5'


def read_and_load_grey(path):
    img = cv2.imread(path, 0)
    img = tf.reshape(img, shape=[1, L_node, W_node, Channel])
    img = tf.cast(img, tf.float32)
    img = img / 255.
    return img


def read_and_load_color(path):
    img = cv2.imread(path, 1)
    img = tf.reshape(img, shape=[1, L_node, W_node, Channel * 3])
    img = tf.cast(img, tf.float32)
    img = img / 255.
    return img


def train():
    k = 1

    Model = tf.keras.models.load_model(Save_path)
    # Model.summary()

    for name in os.listdir(Image_dir):
        print(name)
        Image_path = Image_dir + name
        Mosaic_path = Mosaic_dir + name
        Label_path = Label_dir + name[0: 4] + '.png'
        data = read_and_load_grey(Image_path)
        mosaic = read_and_load_grey(Mosaic_path)
        label = read_and_load_color(Label_path)

        input = tf.keras.layers.concatenate([mosaic, label], 3)  # (None, 256, 256, 4) mosaic: 1 label: 3
        output = Model(input, training=False)

        l1_loss = tf.reduce_mean(tf.abs(output - data))
        ssim_loss = tf.reduce_mean(1 - tf.image.ssim(output, data, max_val=1))
        psnr_loss = tf.reduce_mean(tf.image.psnr(output, data, max_val=1))
        Text_path = Output_dir + 'result.txt'
        with open(Text_path, 'a') as File:
            print('Num:{}, L1 Loss:{:.4f}, SSIM Loss:{:.4f}, PSNR loss:{:.4f}'.format(k, l1_loss, ssim_loss, psnr_loss),
                  file=File)

        output = np.reshape(output, newshape=[L_node, W_node])
        output = output * 255.
        Output_path = Output_dir + name
        cv2.imwrite(Output_path, output)

        k = k + 1


def main(argv=None):
    train()


if __name__=='__main__':
    main()