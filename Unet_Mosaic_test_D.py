import os
import cv2
import numpy as np
import tensorflow as tf

L_node = 256
W_node = 256
Channel = 1

Image_dir = 'S:/UCSD_ped2/Test256/training/'
Mosaic_dir = 'S:/UCSD_ped2/Test256/training_Mosaic/'
Label_dir = 'S:/UCSD_ped2/Test256/label/'
Test_dir = 'S:/UCSD_ped2/Test256/training/'
# Output_dir = 'S:/UCSD_ped2/Test256/Unet_reverse_test/'
Save_path_G = 'S:/UCSD_ped2/Train256/save_Mosaic/60000Gmodel.h5'
Save_path_D = 'S:/UCSD_ped2/Train256/save_Mosaic/60000Dmodel.h5'
# Save_path_G = 'C:/Users/yangm90/Documents/pycharm transfer files/8.0/save_Mosaic/200000Gmodel.h5'
# Save_path_D = 'C:/Users/yangm90/Documents/pycharm transfer files/8.0/save_Mosaic/200000Dmodel.h5'


def read_and_load(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = tf.reshape(img, shape=[1, L_node, W_node, Channel])
    img = tf.cast(img, tf.float32)
    img = img / 255.
    return img


def train():
    k = 1

    GModel = tf.keras.models.load_model(Save_path_G)
    GModel.summary()

    DModel = tf.keras.models.load_model(Save_path_D)
    DModel.summary()

    for name in os.listdir(Image_dir):
        Image_path = Image_dir + name
        Mosaic_path = Mosaic_dir + name
        Label_path = Label_dir + name
        Test_path = Test_dir + name
        # Result_path = Result_dir + name
        data = read_and_load(Image_path)
        mosaic = read_and_load(Mosaic_path)
        label = read_and_load(Label_path)
        test = read_and_load(Test_path)
        # result = read_and_load(Result_path)

        # gen_input = tf.keras.layers.concatenate([mosaic, label], 3)
        # gen_output = GModel(gen_input, training=False)
        gen_output = test

        disc_input = tf.keras.layers.concatenate([gen_output, mosaic, label], 3)
        disc_output = DModel(disc_input, training=False)
        disc_output = disc_output.numpy()
        # print(k, disc_output[0, 0])
        # print(disc_output[0, 0])

        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # g_loss_entropy = cross_entropy(tf.ones_like(disc_output) * 0.9, disc_output)
        # tf.print(k, g_loss_entropy)
        g_loss_entropy = cross_entropy(tf.ones_like(disc_output) * 0.9, disc_output)
        tf.print(k, g_loss_entropy)

        # l1_loss = tf.reduce_mean(tf.abs(output - data))
        # ssim_loss = tf.reduce_mean(1 - tf.image.ssim(output, data, max_val=1))
        # psnr_loss = tf.reduce_mean(tf.image.psnr(output, data, max_val=1))
        # Text_path = Output_dir + 'result.txt'
        # with open(Text_path, 'a') as File:
        #    print('Num:{}, L1 Loss:{:.4f}, SSIM Loss:{:.4f}, PSNR loss:{:.4f}'.format(k, l1_loss, ssim_loss, psnr_loss),
        #          file=File)

        # output = np.reshape(output, newshape=[L_node, W_node])
        # output = output * 255.
        # Output_path = Output_dir + name
        # cv2.imwrite(Output_path, output)

        k = k + 1


def main(argv=None):
    train()


if __name__=='__main__':
    main()