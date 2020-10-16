# Remove the background before processing optical flow images
import cv2
import glob
import numpy as np

L_node = 240
W_node = 360
Channel = 3

# Input_dir = 'S:/UCSD_ped2/Train/training_removal_row_full_pack/'
# Output_dir = 'S:/UCSD_ped2/Train/label_dis_removal_full/'

Input_dir = 'S:/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/'
Output_dir = 'S:/UCSD_ped2/Train/label_dis/'

Input_path = glob.glob(Input_dir + 'Train*')
dis = cv2.DISOpticalFlow_create(2)


def train():
    k = 1
    for i, path in enumerate(Input_path):
        Input_name = glob.glob(path + '/*.tif')
        mag_max = 0

        # -----------------------------------------------
        # Get the maximum value of magnitude in per video
        # -----------------------------------------------
        for j, name in enumerate(Input_name):
            if j == (len(Input_name) - 1):
                break

            frame1 = cv2.imread(name)
            previous = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('previous', previous)
            print(name)
            name_next = Input_name[j + 1]
            frame2 = cv2.imread(name_next)
            next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('next', next_fame)

            # Get magnitude and angle
            flow = dis.calc(previous, next_frame, None)
            # flow = cv2.calcOpticalFlowFarneback(previous, next_fame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            if mag_max < np.max(mag):
                mag_max = np.max(mag)
        print(mag_max)

        # ----------------------
        # Get optical flow image
        # ----------------------
        for j, name in enumerate(Input_name):
            if j == (len(Input_name) - 1):
                break

            # Get previous image and current image
            previous = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
            name_next = Input_name[j + 1]
            next_frame = cv2.imread(name_next, cv2.IMREAD_GRAYSCALE)

            # Get magnitude and angle
            flow = dis.calc(previous, next_frame, None)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Get HSV matrix
            hsv = np.zeros(shape=(flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            # hsv[..., 1] = 255 / mag_max * mag
            # for l in range(L_node):
            #    print(hsv[l, :, 1])

            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # cv2.imwrite(Output_dir + 'hsv_mag.png', hsv[:, :, 1])
            # cv2.imwrite(Output_dir + 'hsv_ang.png', hsv[:, :, 0])
            # cv2.imwrite(Output_dir + 'bgr_mag.png', bgr[:, :, 1])
            # cv2.imwrite(Output_dir + 'bgr_ang.png', bgr[:, :, 0])
            # cv2.imshow('bgr', bgr)
            # cv2.waitKey(0)

            k_4d = '%04d' % k
            cv2.imwrite(Output_dir + str(k_4d) + '.png', bgr)
            k += 1


def main(argv=None):
    train()


if __name__ == '__main__':
    main()