import numpy as np
import cv2 as cv

cascade_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

image = cv.imread('img.jpg')
img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

face = cascade_classifier.detectMultiScale(img_gray, 1.3, 5)

for (x, y, w, h) in face:
    cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv.imshow('image', image)
cv.waitKey(0)
cv.destroyAllWindows()