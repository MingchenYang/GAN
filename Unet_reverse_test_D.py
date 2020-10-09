import os
import cv2
import numpy as np
import tensorflow as tf

L_node = 256
W_node = 256
Channel = 1

Image_dir = 'S:/pix2pix optflow/Train256/label/'  # Image dataset in reverse file is the input of generator
Label_dir = 'S:/pix2pix optflow/Train256/training/'  # Label dataset in reverse is the real image
# Result_dir = 'S:/pix2pix optflow/Train256/Unet_reverse_test/'  # Result dataset of generator
# Output_dir = 'S:/pix2pix optflow/Test256/Unet_reverse_test/'
Save_path_G = 'S:/pix2pix optflow/Train256/save_reverse/10000Gmodel.h5'
Save_path_D = 'S:/pix2pix optflow/Train256/save_reverse/10000Dmodel.h5'


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
        Label_path = Label_dir + name
        # Result_path = Result_dir + name
        data = read_and_load(Image_path)
        label = read_and_load(Label_path)
        # result = read_and_load(Result_path)

        gen_output = GModel(label, training=False)

        disc_input = tf.keras.layers.concatenate([gen_output, label], 3)
        disc_output = DModel(disc_input, training=False)
        disc_output = disc_output.numpy()
        print(k, disc_output[0, 0])

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