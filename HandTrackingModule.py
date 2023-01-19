import cv2
from mediapipe import mediapipe as mp

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        print("In findHands")
        #print(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        self.result = self.hands.process(img_rgb)

        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        if self.result.multi_hand_landmarks:
            hand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                x, y = int(lm.x * w), int(lm.y * h)
                lmList.append([id, x, y])
                if draw:
                    cv2.circle(img, (x, y), 15, (255, 0, 255), cv2.FILLED)

        return lmList


def main():
    cap = cv2.VideoCapture(1)
    detector = handDetector()
    while True:
        success, image = cap.read()
        #if image is not None:
        image = detector.findHands(image)
        list = detector.findPosition(image)
        if len(list) != 0:
            print(list[4])

        cv2.imshow("Image", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()