import cv2
import numpy as np
image = cv2.imread('AR.png')  
image = cv2.resize(image, (640, 480))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        area = cv2.contourArea(approx)
        if area > 1000:
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)
            cv2.putText(image, "Marker Detected", tuple(approx[0][0]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
cv2.imshow("Marker Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()