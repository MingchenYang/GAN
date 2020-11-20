# CUHK dataset, test number of frames in one video
import cv2
import glob

num = 1

Video_dir = 'S:/CUHK/Train/training_videos/'
Image_dir = 'S:/CUHK/Train/training/'

Video_path = glob.glob(Video_dir + '/*.avi')

for path in Video_path:
    cap = cv2.VideoCapture(path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(fps, width, height, fps_num)