import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

File_path = 'S:/UCSD_ped2/Train256/training_Sequence1/0177.npz'

num_sequence = 2
L_node = 256
W_node = 256
Channel = 1

Sequence_image = np.load(File_path)['image']
print(Sequence_image.shape)
Sequence_image = np.reshape(Sequence_image, newshape=[num_sequence, L_node, W_node])
print(Sequence_image.shape)
for i in range(num_sequence):
    plt.subplot(1, 6, i + 1)
    plt.imshow(Sequence_image[i])

Sequence_label = np.load(File_path)['label']
print(Sequence_label.shape)
# Sequence_label = np.reshape(Sequence_label, newshape=[Channel * 3, L_node, W_node])
label_l1 = Sequence_label[0: 65536, ]
label_l2 = Sequence_label[65536: 131072, ]
label_l3 = Sequence_label[131072: 196608, ]
label_l1 = np.reshape(label_l1, newshape=[L_node, W_node, Channel])
label_l2 = np.reshape(label_l2, newshape=[L_node, W_node, Channel])
label_l3 = np.reshape(label_l3, newshape=[L_node, W_node, Channel])
label = np.concatenate([label_l1, label_l2, label_l3], 2)
cv2.imshow('label', label)
cv2.waitKey(0)
plt.subplot(1, 6, num_sequence + 1)
plt.imshow(Sequence_label)

plt.show()
cv2.waitKey(0)