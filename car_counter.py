from ultralytics import YOLO
import cv2
import cvzone
from sort import *
import math
import numpy as np

cap = cv2.VideoCapture("footage.mp4")
cap.set(3, 1280)
cap.set(4, 720)

classNames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bs', 'train', 'trck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'mbrella', 'handbag', 'tie', 'sitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'srfboard', 'tennis racket', 'bottle', 'wine glass', 'cp', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'dont', 'cake', 'chair', 'coch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mose', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrsh'
]

# MASK
mask = cv2.imread("mask.png")

# TRACKER ID
tracker = Sort(max_age=20 , min_hits = 3 , iou_threshold=0.3)

# YOLO MODEL
model = YOLO('yolov8n.pt')

#TEXT FORMATTING 

#text_position = (int(x1), max(0, int(y1) - 20))
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.4      
font_color = (255, 255, 0)
font_thickness = 1

# CORDINATES/LIMITS OF THE LINE
limits = [524,346,1496,298]

# COUNTER
tcount = []

# MAIN CODE
while True:
    success, img = cap.read()

    # Resize the mask and fits it
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    imgRegion = cv2.bitwise_and(img,mask)

    # Detecting
    results = model(imgRegion , stream=True)

    # Detections for tracker
    detections = np.empty((0,5))

    # Looping to get coordinates of the bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            #x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)    #This can also be used
            w , h = x2-x1 , y2-y1
            bbox = int(x1),int(y1),int(w),int(h)
            #cv2.rectangle(img , (x1,y1) , (x2,y2) , (255,255,0) , 3 )  #This can also be used

            #Confidence
            conf = math.ceil((box.conf[0]*100))/100
            #CLASS
            cls = int(box.cls[0])
            currentclass = classNames[cls]
            if currentclass == "car" or currentclass=="truck" or currentclass == "bus" or currentclass == "motorbike" and conf>0.3:
                #cvzone.cornerRect(img,(bbox) , l=10 , t=3)
                #cv2.putText(img, f"{conf}", text_position, font, font_scale, font_color, font_thickness)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentArray))


    # TRACKING DETECTIONS AND GIVING UNIQUE ID TO EACH CAR
    resultsTracker = tracker.update(detections)
    cv2.line(img , (limits[0] , limits[1]) , (limits[2],limits[3]) , (0,0,255) , 5)
    for results in resultsTracker:
        x1,y1,x2,y2,id = results
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        w , h = x2-x1 , y2-y1

        cvzone.cornerRect(img,(x1,y1,w,h) , l=10 , t=3)
        cv2.putText(img, f"{id}  {conf}", (int(x1), max(0, int(y1) - 20)), font, font_scale, font_color, font_thickness)
        cx , cy = x1+w//2 , y1+h//2
        cv2.circle(img , (cx,cy) , 5 , (255,0,0) , cv2.FILLED)

        if limits[0]< cx < limits[2] and limits[1]-10 < cy < limits[1]+10:
            if tcount.count(id)==0:
                tcount.append(id)
    #cv2.putText(img, f"COUNT : {tcount}", (50,50), font, 3, font_color, font_thickness)
    cvzone.putTextRect(img, f"COUNT : {len(tcount)}", (50,50))

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()