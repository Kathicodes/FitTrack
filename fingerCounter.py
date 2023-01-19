import mediapipe as mp
import cv2
import numpy as np
import os
import time
import HandTrackingModule as htm
import FitTrac as ft


def getNumber(ar):
    s = ""
    for i in ar:
        s += str(ar[i])

    if (s == "00000"):
        #  return ("STOP")
        return ("")
    elif (s == "01000"):
        ft.bicep_curl_count()
        return ("")
    elif (s == "01100"):
        ft.lateral_raise_count()
        return ("")
    elif (s == "01110"):
        ft.arm_circle_count()
        return ("")
    elif (s == "01111"):
        ft.dumb_punch_count()
        return ("")
    elif (s == "11111"):
        return ("")
    elif (s == "01001"):
        return ("")
    elif (s == "01011"):
        print(s)


wcam, hcam = 1080, 720
cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)
pTime = 0
detector = htm.handDetector(detectionCon=0.75)

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=True)
    list = detector.findPosition(img, draw=False)
    # print(lmList)

    cv2.putText(img, 'SHOW 5 TO START', (850, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, 'Select exercise:', (850, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 0), 3,
                cv2.LINE_AA)
    cv2.putText(img, '1) Bicep Curls', (850, 175), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, '2) Lateral Raise', (850, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, '3) Arm Circles', (850, 325), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, '4) Dumbell Punch', (850, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)

    tipId = [4, 8, 12, 16, 20]
    if (len(list) != 0):
        fingers = []
        # thumb
        if (list[tipId[0]][1] > list[tipId[0] - 1][1]):
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 fingers
        for id in range(1, len(tipId)):
            if (list[tipId[id]][2] < list[tipId[id] - 2][2]):
                fingers.append(1)
            else:
                fingers.append(0)

        ss = str(getNumber(fingers))
        cv2.putText(img, ss, (5, 150), cv2.FONT_ITALIC,
                    2.5, (245, 117, 66), 10)


        print("In Finger Counter ......")

    cv2.imshow("image", img)
    if (cv2.waitKey(3) & 0xFF == ord('q')):
        break