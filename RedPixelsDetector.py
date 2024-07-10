import cv2
import numpy as np

# Load the image
image_path = "red_bag.jpg"
image = cv2.imread(image_path)

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for red color in HSV
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

# Create masks for the red color
mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)

# Count the number of red pixels
red_pixel_count = cv2.countNonZero(red_mask)

print(f"Number of red pixels: {red_pixel_count}")

# Optional: Display the mask
cv2.imshow("Red Mask", red_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
