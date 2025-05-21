import cv2
import numpy as np
image = cv2.imread("moon.jpg")
background = cv2.imread("background.jpg")
background = cv2.resize(background, (image.shape[1], image.shape[0]))
mask = np.zeros(image.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (50, 50, image.shape[1]-50, image.shape[0]-50)
cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
foreground = image * mask2[:, :, np.newaxis]
output_image = np.where(foreground == 0, background, foreground)
cv2.imwrite("output_image.jpg", output_image)
cv2.imshow('Original Frame', image)
cv2.imshow("Virtual Background", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()