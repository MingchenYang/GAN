# CUHK dataset, transform videos to images, full image - 1
import cv2
import glob

num = 1

Video_dir = 'S:/CUHK/Train/training_videos/'
Image_dir = 'S:/CUHK/Train/training/'

Video_path = glob.glob(Video_dir + '/*.avi')

for path in Video_path:
    num_record = 1
    cap = cv2.VideoCapture(path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(fps, width, height, fps_num)

    success, image = cap.read()

    while success:
        Image_path = Image_dir + str(num).zfill(5) + '.png'
        cv2.imwrite(Image_path, image)
        print(Image_path)
        num += 1
        num_record += 1
        success, image = cap.read()
        if num_record == fps_num:
            break

    cap.release()