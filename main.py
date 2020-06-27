#Author: Kanit Mann

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
kernel = np.ones((5,5),np.uint8)

while True:
    success, img = cap.read()
    #getting dimensions and channel for the screen
    x1, y1, z = img.shape
    #img = np.flip(img, axis=1)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 100, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask1 + mask2
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # loop over the contours

    if len(contours) != 0:
        # draw in blue the contours that were found
        cv2.drawContours(img, contours, -1, 255, 3)

        # find the biggest area
        c = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(c)
        # draw the book contour (in green)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #Center:  cv2.rectangle(img,(325,225),(325,225),(0,0,255),2)

        # draw the contour and center of the shape on the image
        cv2.circle(img, (x+h//2, y+w//2), 7, (255, 255, 255), -1)
        cv2.putText(img, "center", (x+h//2-20, y+w//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        #center of contoured area
        cy = x+h//2
        #center of img
        Cy = x1//2
        error = y1/100*2
        if Cy>cy+error:
            print("left")
        elif Cy<cy-error:
            print("right")
        else:
            print("move")
    #cv2.drawContours(img ,contours,-1,(0,0,255),3)

    cv2.imshow("output",img)
    cv2.imshow("final",mask)

    #cv2.imshow("Contour",contours)
    if(cv2.waitKey(1) & 0xff==ord('q')):
        break
cv2.destroyAllWindows()
cap.release()