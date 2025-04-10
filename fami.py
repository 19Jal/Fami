import numpy as np
import cv2
from playsound import playsound

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

while True:
    success, img = cap.read()

    # check if we get the frame
    if success == True:
        img = img[240:480, 240:480]
        fgmask = fgbg.apply(img)
        _, thresh = cv2.threshold(fgmask.copy(), 180, 255, cv2.THRESH_BINARY)
        # creating a kernel of 4*4
        kernel = np.ones((7, 7), np.uint8)
        # applying errosion to avoid any small motion in video
        thresh = cv2.erode(thresh, kernel, iterations = 2)
        # dilating our image
        thresh = cv2.dilate(thresh, None, iterations = 5)

        # finding the contours
        contours, heirarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # finding area of contour
            area = cv2.contourArea(contour)
            if area > 2000:
                print(area)
                playsound('jingle.mp3')
                break

        cv2.imshow('frame',img)
        cv2.imshow('frame2', thresh)
        if cv2.waitKey(1)==27: #ESC = exit
            break

cap.release()
cv2.destroyAllWindows()