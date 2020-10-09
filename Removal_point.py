import cv2
import glob
import numpy as np

L_node = 240
W_node = 360
Channel = 3
Kernel_size = 3
Sigma = 3
Radius = 15

Input_dir = 'S:/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/'
Output_dir = 'S:/UCSD_ped2/Train/training_removal/'

Input_path = glob.glob(Input_dir + 'Train*')


def optflow(prev, next):
    diff = np.abs(next - prev)
    # diff = cv2.medianBlur(diff, ksize=Kernel_size)
    # cv2.imwrite(Output_dir + 'diff.png', diff)
    image = np.zeros_like(prev)

    for l in range(L_node):
        for w in range(W_node):
            l_left = max([l - Radius, 0])
            l_right = max([l + Radius + 1, 0])
            w_left = max([w - Radius, 0])
            w_right = max([w + Radius + 1, 0])

            if np.mean(diff[l_left: l_right, w_left: w_right]) > 3:
                image[l, w] = prev[l, w]
                # image[l, w] = 255
    return image


def train():
    k = 1

    for i, path in enumerate(Input_path):
        Input_name = glob.glob(path + '/*.tif')
        for j, name in enumerate(Input_name):
            if j == (len(Input_name) - 1):
                break

            previous = cv2.imread(name, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            print(name)
            name_next = Input_name[j + 1]
            next_frame = cv2.imread(name_next, cv2.IMREAD_GRAYSCALE).astype(np.float32)

            image = optflow(previous, next_frame)

            k_4d = '%04d' % k
            image = image.astype(np.uint8)
            cv2.imwrite(Output_dir + str(k_4d) + '.png', image)
            k += 1

            # image_median = cv2.medianBlur(image, ksize=Kernel_size)
            # cv2.imwrite(Output_dir + 'median.png', image_median)

            # image_avg = cv2.blur(image, ksize=(Kernel_size, Kernel_size))
            # cv2.imwrite(Output_dir + 'avg.png', image_avg)

            # cv2.imshow('image_diff', image)
            # cv2.waitKey(0)


def main(argv=None):
    train()


if __name__=='__main__':
    main()