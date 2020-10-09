import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

Buffer_size = 1000
Batch_size = 10
L_node = 158
W_node = 238
Channel = 1
Learning_rate_gen = 0.01
Training_steps = 1000

Image_dir = 'S:/pix2pix optflow/Train/training/'
Label_dir = 'S:/pix2pix optflow/Train/label/'
Output_dir = 'S:/pix2pix optflow/Train/result/'
Test_dir = 'S:/pix2pix optflow/Test/label/7060.jpg'
Output_test_dir = 'S:/pix2pix optflow/Test/result/7060.jpg'
Save_path = 'S:/pix2pix optflow/Train/save/'
Image_path = [os.path.join(Image_dir, i) for i in os.listdir(Image_dir)]
Label_path = [os.path.join(Label_dir, i) for i in os.listdir(Label_dir)]


def recover(img):
    max_value = np.max(img)
    img = (255. / max_value) * img
    return img


def mse(img1, img2):
    mse = np.mean((img1 - img2)**2)
    return mse


def read_and_load(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.cast(image, tf.float32)
    image = image / 255.
    return image


def Generator():
    input = tf.keras.Input(shape=[L_node, W_node, Channel])
    layer1_1 = tf.keras.layers.Conv2D(64, 1, 1, 'same', activation='relu')(input)
    layer1_2 = tf.keras.layers.Conv2D(128, 3, 1, 'same', activation='relu')(input)
    layer1_3 = tf.keras.layers.Conv2D(32, 5, 1, 'same', activation='relu')(input)
    #layer1_1 = tf.keras.layers.Activation('relu')(layer1_1)
    #layer1_2 = tf.keras.layers.Activation('relu')(layer1_2)
    #layer1_3 = tf.keras.layers.Activation('relu')(layer1_3)
    layer1 = tf.keras.layers.concatenate([layer1_1, layer1_2, layer1_3], axis=-1)
    layer2 = tf.keras.layers.Activation('relu')(layer1)
    layer2 = tf.keras.layers.Conv2D(64, 3, 1, 'same', activation='relu', dilation_rate=2)(layer2)
    #layer2 = tf.keras.layers.Activation('relu')(layer2)
    layer3 = tf.keras.layers.Conv2D(1, 3, 1, 'same', activation='sigmoid')(layer2)
    #layer4 = tf.keras.layers.Activation('sigmoid')(layer4)
    output = layer3
    model = tf.keras.Model(input, output)
    return model


def train():
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # Slice the input image path string, get a dataset about string
    path_image_ds = tf.data.Dataset.from_tensor_slices(Image_path)
    # Load all images into dataset
    image_ds = path_image_ds.map(read_and_load, num_parallel_calls=AUTOTUNE)
    # Slice the input label path string, get a dataset about string
    path_label_ds = tf.data.Dataset.from_tensor_slices(Label_path)
    # Load all labels into dataset
    label_ds = path_label_ds.map(read_and_load, num_parallel_calls=AUTOTUNE)
    # Pack images and labels together
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    # Shuffle, Repeat, Batch operations
    ds = image_label_ds.shuffle(buffer_size=Buffer_size)
    ds = ds.repeat()
    ds = ds.batch(Batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    '''
    k = 0
    for img, label in ds:
        for i in range(10):
            plt.subplot(5, 4, 2*i + 1)
            plt.imshow(img[i])
            plt.subplot(5, 4, 2*i + 2)
            plt.imshow(label[i])
        plt.savefig(Output_dir + str(k) + '.jpg')
        plt.close()
        k += 1
    '''

    generator = Generator()
    generator.build(input_shape=(Batch_size, L_node, W_node, Channel))
    generator.summary()

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_rate_gen, beta_1=0.9, beta_2=0.999)

    k = 0
    # cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    for epoch in range(Training_steps):
        for data, label in ds:
            with tf.GradientTape() as Tape:
                gen_output = generator(data, training=True)
                # g_loss = cross_entropy(gen_output, label)
                loss = 1 - tf.image.ssim(gen_output, label, max_val=1)
                g_loss = tf.reduce_mean(loss)
                # for index in range(Batch_size):
                #    print(loss[index])
                # g_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(gen_output, label))
            g_gradients = Tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
            if k % 10 == 0:
                print("Step:{} Loss:{:.4f}".format(k, g_loss))
            if k % 100 == 0:
                gen_result = np.reshape(gen_output, newshape=[Batch_size, L_node, W_node])
                label_result = np.reshape(label, newshape=[Batch_size, L_node, W_node])
                print(mse(gen_result[1], label_result[1]))
                # recover_gen = recover(gen_result[1])
                # cv2.imshow('gen_result', recover_gen)
                # cv2.waitKey(0)
                for i in range(10):
                    plt.subplot(5, 4, 2*i + 1)
                    plt.imshow(gen_result[i], cmap=plt.gray())
                    plt.subplot(5, 4, 2*i + 2)
                    plt.imshow(label_result[i], cmap=plt.gray())
                plt.savefig(Output_dir + str(k) + '.jpg')
                plt.close()
                cv2.imwrite('S:/pix2pix optflow/Test/result/test.jpg', gen_result[1] * 255.)
                cv2.imwrite('S:/pix2pix optflow/Test/result/label.jpg', label_result[1] * 255.)
            if k % 1000 == 0:
                # Save the model with structure and weights initializer
                generator.save(Save_path + str(k) + 'model' + '.h5')

            k = k + 1

'''
            if k == 100:
                test_image = cv2.imread('S:/pix2xpix optflow/Test/training')
                plt.imshow(test_image)
                test_image = tf.reshape(test_image, shape=[1, L_node, W_node, Channel])
                output_test_result = generator(test_image, training=False)
                output_test_result = np.reshape(output_test_result, newshape=[L_node, W_node])
                cv2.imwrite(Output_test_dir, output_test_result)
'''


def main(argv=None):
    train()


if __name__ == '__main__':
    main()