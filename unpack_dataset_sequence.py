import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

File_path = 'S:/pix2pix optflow/Generator256/sequence_edge_label/3000.npz'
File_path = 'S:/pix2pix optflow/Train256/training_Multi/3000.npz'

num_sequence = 5
L_node = 256
W_node = 256

Sequence_image = np.load(File_path)['image']
print(Sequence_image.shape)
Sequence_image = np.reshape(Sequence_image, newshape=[num_sequence, L_node, W_node])
print(Sequence_image.shape)
for i in range(num_sequence):
    plt.subplot(1, 7, i + 1)
    plt.imshow(Sequence_image[i])

Sequence_label = np.load(File_path)['label']
print(Sequence_label.shape)
Sequence_label = np.reshape(Sequence_label, newshape=[L_node, W_node])
plt.subplot(1, 7, num_sequence + 1)
plt.imshow(Sequence_label)

Sequence_edge_label = np.load(File_path)['image_label']
print(Sequence_edge_label.shape)
Sequence_edge_label = np.reshape(Sequence_edge_label, newshape=[L_node, W_node])
plt.subplot(1, 7, num_sequence + 2)
plt.imshow(Sequence_edge_label)

plt.show()
cv2.waitKey(0)