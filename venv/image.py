import numpy as np
import cv2
from matplotlib import pyplot as plt
#lähde: opencv-python-tutroals.readthedocs.io
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

img = cv2.imread('meidomegis.png')
megis = cv2.imread('megislowres.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)

    roi = img[y:y + h, x:x + w]

    rows, cols, channels = roi.shape
    img2 = cv2.resize(megis, (cols, rows))

    # I want to put logo on top-left corner, So I create a ROI
    rows, cols, channels = img2.shape

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    # Put logo in ROI and modify the main image
    print (roi.shape)
    print(img2.shape)

    dst = cv2.add(img1_bg, img2_fg)
    img[y:y+h, x:x+w] = dst

cv2.imshow('img',img)
cv2.imwrite('taidetta.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()