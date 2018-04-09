import b418
import numpy as np
import cv2 as cv

# Load images and compute average image
imgs = b418.load_data()
average_image = np.mean(imgs, axis = 0)

# Save average image
cv.imwrite('average_image.png', average_image)
