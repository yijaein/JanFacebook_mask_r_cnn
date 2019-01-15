
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = '/home/bong6/data/mrcnn_cer/classificationdataset_224 (copy)/train/Type_1_re/5976.png'

img = cv2.imread(path, cv2.IMREAD_COLOR)

b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]

print(np.max(r), np.max(g), np.max(b))

img[img[:, :, 0] > 0] = 255
img[img[:, :, 2] > 0] = 255

# img[:, :, 1] = 0

plt.imshow(img)
plt.show()
