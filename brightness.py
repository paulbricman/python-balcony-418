import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import b418

# Load images and compute mean RGB values for each image, smooth the signal
imgs = b418.load_data()
average_value_list = np.mean(imgs, axis=(1, 2, 3))
smooth_value_list = gaussian_filter(average_value_list, 20)

# Known sunrise and sunset times
known_sunrise = b418.time_to_sample(6, 43)
known_sunset = b418.time_to_sample(19, 52)
X = [known_sunrise, known_sunset]
Y = smooth_value_list[X]

# Plot series
plt.plot(average_value_list, 'b')
plt.plot(smooth_value_list, 'c')
plt.axvline(x=known_sunrise, color = 'y')
plt.axvline(x=known_sunset, color = 'r')

plt.title('Light Intensity')
plt.legend(('original', 'smooth', 'sunrise', 'sunset'))
plt.show()
