import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

File_path = 'S:/UCSD_ped1/Train256/training_Multi/0195.npz'

num_sequence = 5
L_node = 256
W_node = 256

Sequence_image = np.load(File_path)['image']
print(Sequence_image.shape)
Sequence_image = np.reshape(Sequence_image, newshape=[num_sequence, L_node, W_node])
print(Sequence_image.shape)
for i in range(num_sequence):
    plt.subplot(1, 6, i + 1)
    plt.imshow(Sequence_image[i])

Sequence_label = np.load(File_path)['label']
print(Sequence_label.shape)
Sequence_label = np.reshape(Sequence_label, newshape=[L_node, W_node])
plt.subplot(1, 6, num_sequence + 1)
plt.imshow(Sequence_label)

plt.show()
cv2.waitKey(0)