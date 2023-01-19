import cv2
import mediapipe as mp
import numpy as np
# import HandTrackingModule as htm
# import fingerCounter as fc
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
#Calculating Angles
def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle >180:
        angle = 360-angle
    return angle

#New function definition for different poses
def bicep_curl(image,results,counter):
    # Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark
        # #print(landmarks)
        #Get coordinates
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        # Calculate angle
        left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        #return(left_angle)
        cv2.putText(image, str(left_angle), 
                    tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    )
        # Repetition counter
        if left_angle > 160:
            stage = "down"
        if left_angle < 30 and stage =="down":
            stage = "up"
            counter = counter + 1
            print(counter)
    except:
        pass    

def video_input(cap):
    wcam, hcam = 1080, 720
    cap = cv2.VideoCapture(0)
    cap.set(3, wcam)
    cap.set(4, hcam)
    # Curl counter variables
    counter = 0 
    stage = None
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            #Recolour image to RGB
            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            #Make detection
            results = pose.process(image)
            #Recolour back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            print("AAA AAA AAA")
            print(results)
            print("AAA AAA AAA")
            #print(s)
            #if (s == "01000"):
            bicep_curl(image,results,0)
            '''  elif (s == "01100"):
                lateral_raise(image,results,counter)
            elif (s == "01110"):
                arm_circle(image,results,counter)
            elif (s == "01111"):
                dumb_punch(image,results,counter)
            '''
            #Rener curl counter
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            cv2.putText(image, 'Bicep Curls', (750, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(image, 'REPS', (15,12), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10,60),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (65,12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (85,60),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                    
            #Render detections
            mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )
            cv2.imshow('Mediapipe Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            cap.release()
            cv2.destroyAllWindows()

            #results.pose_landmarks
            #print(mp_pose.POSE_CONNECTIONS)
            #print(len(landmarks))
            for lndmrk in mp_pose.PoseLandmark:
                print(lndmrk)
            #print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
            #print(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
            #print(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])