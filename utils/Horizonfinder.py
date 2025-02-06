import cv2
import numpy as np
import matplotlib.pyplot as plt
import tools

image_folder = "Data\RFC Pose Estimation Images"
label_folder = "Data\Bounding_box_labels"
# Load the image
image, _, _, _, _, _, _ = tools.load_image_pair_and_labels(image_folder, label_folder, 0)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Show the image
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
plt.show()

# Perform Canny edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Show edges
plt.imshow(edges, cmap='gray')
plt.show()