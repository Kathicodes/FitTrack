import cv2
import os
import time
import HandTrackingModule as htm
import FitTrac as ft
def getNumber(ar):
    s = ""
    for i in ar:
        s += str(ar[i])

    if (s == "00000"):
        return ("STOP")
    elif (s == "01000"):
        #cv2.putText(img, str("Bicep curl"), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), cv2.LINE_AA)
        ft.video_input(cap)
        return ("Bicep curl")
    elif (s == "01100"):
        # ft.video_input(cap)
        return ("Lunges")
    elif (s == "01110"):
        # ft.video_input(cap)
        return ("Lateral Raise")
    elif (s == "01111"):
        # ft.video_input(cap)
        return ("Push ups")
    elif (s == "11111"):
        # ft.video_input(cap)
        return ("Squats")
    elif (s == "01001"):
        return ("START")
    elif (s == "01011"):
        return (7)


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

        ss =str(getNumber(fingers))
        cv2.putText(img, str(getNumber(fingers)), (5, 150), cv2.FONT_ITALIC,
                    2.5, (245, 117, 66), 10)
        
        #if(ss == "Bicep curl"):
        #    ft.video_input(cap)
            
        
    cv2.imshow("image", img)
    if (cv2.waitKey(3) & 0xFF == ord('q')):
        break