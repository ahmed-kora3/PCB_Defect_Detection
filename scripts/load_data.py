import os
import cv2
import matplotlib.pyplot as plt

# Dataset path
dataset_path = "dataset/pcb-defect-dataset/train/images"
# Read image names
images = os.listdir(dataset_path)

# Print total images
print("Total Images:", len(images))

# Read first image
img_path = os.path.join(dataset_path, images[0])

image = cv2.imread(img_path)

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display image
plt.imshow(image)
plt.title("PCB Image")
plt.axis("off")

plt.show()