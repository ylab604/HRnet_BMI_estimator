from PIL import Image
import numpy as np


image = Image.open("/home/ylab3/HRnet_BMI_estimator/4_05_6_depth.jpeg")
np_array = np.array(image)
pil_image = Image.fromarray(np_array)

np_array[0][0].shape

#pil_image.show()

