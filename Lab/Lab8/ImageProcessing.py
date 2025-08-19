import cv2
import matplotlib.pyplot as plt
try:
    # Use a real image path here
    image = cv2.imread('images (2).jpeg', cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image not found. Check the file path.")
except FileNotFoundError as e:
    print(e)
    exit()

histogram_original = cv2.calcHist([image], [0], None, [256], [0, 256])

plt.figure(figsize=(8, 6))
plt.title('Histogram of Original Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.plot(histogram_original)
plt.xlim([0, 256])
plt.grid(True)
plt.show()

equalized_image = cv2.equalizeHist(image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Equalized Image')
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

histogram_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

plt.figure(figsize=(8, 6))
plt.title('Histogram of Equalized Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.plot(histogram_equalized)
plt.xlim([0, 256])
plt.grid(True)
plt.show()