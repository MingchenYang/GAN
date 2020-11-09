import os
import cv2
import numpy as np
import tensorflow as tf

Time_steps = Num_sequence = 5
L_node = 256
W_node = 256
Channel = 1

File_dir = 'S:/UCSD_ped2/Test256/training_Multi/'
Output_dir = 'S:/UCSD_ped2/Test256/Unet_Multi_test/'
Save_path = 'S:/UCSD_ped2/Train256/save_Multi/96000Gmodel.h5'
# Save_path = 'C:/Users/yangm90/Documents/pycharm transfer files/8.0/save_Mosaic/200000Gmodel.h5'


def read_and_load(path):
    Sequence_image = np.load(path)['image']  # (5, 256, 256)
    Sequence_image = np.reshape(Sequence_image,
                                newshape=[Num_sequence, L_node, W_node, Channel])  # (5, 256, 256, 1)
    for num in range(Num_sequence):
        if num == 0:
            image = Sequence_image[num, ]
            image_reshape = image
        else:
            image_reshape = np.concatenate([image_reshape, image], 2)  # (256, 256, 5)
    Sequence_image = np.reshape(image_reshape, newshape=[1, L_node, W_node, Time_steps])  # (1, 256, 256, 5)
    Sequence_image = tf.cast(Sequence_image, tf.float32)
    Sequence_image = Sequence_image / 255.

    Sequence_label = np.load(path)['label']
    Sequence_label = np.reshape(Sequence_label, newshape=[1, L_node, W_node, Channel])  # (1, 256, 256, 1)
    Sequence_label = tf.cast(Sequence_label, tf.float32)
    Sequence_label = Sequence_label / 255.

    return Sequence_image, Sequence_label


def train():
    k = 1

    Model = tf.keras.models.load_model(Save_path)
    # Model.summary()

    for name in os.listdir(File_dir):
        print(name)

        File_path = File_dir + name
        image, data = read_and_load(File_path)

        input = image
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
        Output_path = Output_dir + name[0: 4] + '.jpg'
        cv2.imwrite(Output_path, output)

        k = k + 1


def main(argv=None):
    train()


if __name__=='__main__':
    main()