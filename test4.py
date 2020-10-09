import cv2

image1_dir = 'S:/UCSD_ped2/Train/training_removal_background/0001.png'
image2_dir = 'S:/UCSD_ped2/Train/training_removal_background/0002.png'
image3_dir = 'S:/UCSD_ped2/Train/training_removal_background/0003.png'

image1 = cv2.imread(image1_dir, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(image2_dir, cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread(image3_dir, cv2.IMREAD_GRAYSCALE)

image = image1 + image2 + image3

cv2.imshow('image', image)
cv2.waitKey(0)