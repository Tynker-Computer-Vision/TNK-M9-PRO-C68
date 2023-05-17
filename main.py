import numpy as np
import cv2
import math

''' '''
# 1. Set the position of goal
goalX = 530
goalY = 130

# 2. Trajectory arrayCords
xCords = []
yCords = []

# 3. Load the tracker
tracker = cv2.legacy.TrackerCSRT_create()

''' '''
confidenceThreshold = 0.3
NMSThreshold = 0.1

modelConfiguration = 'cfg/yolov3.cfg'
modelWeights = 'yolov3.weights'

labelsPath = 'coco.names'

labels = open(labelsPath).read().strip().split('\n')

yoloNetwork = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

video = cv2.VideoCapture("bb2.mp4")

# 4. Flag variable o show whether basketball is detected or not
detected = False

# 8 Define the function to draw the bounding box


def drawBox(img, bbox):
    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2])
    h = int(bbox[3])

    cv2.rectangle(img, (x, y), ((x+w), (y+h)), (255, 0, 255), 3, 1)
    cv2.putText(img, "Tracking", (75, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# 9 Define the function to track the goal
def goalTrack(img, bbox):
    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2])
    h = int(bbox[3])

    # Get the CENTER Points of the Bounding Box
    c1 = x + int(w/2)
    c2 = y + int(h/2)

    # # Draw a small circle using CENTER POINTS
    # cv2.circle(img, (c1, c2), 2, (0, 0, 255), 5)

    cv2.circle(img, (int(goalX), int(goalY)), 2, (0, 255, 0), 3)

    # Calculate Distance
    dist = math.sqrt(((c1-goalX)**2) + (c2-goalY)**2)

    # Goal is reached if distance is less than 20 pixel points
    if (dist <= 20):
        cv2.putText(img, "Goal", (300, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    xCords.append(c1)
    yCords.append(c2)

    # Draw the circles for the previous center points
    for i in range(len(xCords)-1):
        cv2.circle(img, (xCords[i], yCords[i]), 2, (0, 0, 255), 5)


while True:
    check, image = video.read()
    image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)

    dimensions = image.shape[:2]
    H, W = dimensions

    # 5. Only detect the ball if not detected
    if detected == False:
        blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416))
        yoloNetwork.setInput(blob)

        layerName = yoloNetwork.getUnconnectedOutLayersNames()
        layerOutputs = yoloNetwork.forward(layerName)

        boxes = []
        confidences = []
        classIds = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]

                if confidence > confidenceThreshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY,  width, height) = box.astype('int')
                    x = int(centerX - (width/2))
                    y = int(centerY - (height/2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIds.append(classId)

        detectionNMS = cv2.dnn.NMSBoxes(
            boxes, confidences, confidenceThreshold, NMSThreshold)

        if (len(detectionNMS) > 0):
            for i in detectionNMS.flatten():

                if labels[classIds[i]] == "sports ball":
                    x, y, w, h = boxes[i]

                    color = (255, 0, 0)

                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    # 6. Initialise the tracker on the img and the bounding box
                    tracker.init(image, boxes[i])
                    # 7. Changing flag variable value to true
                    detected = True
    else:
        # Update the tracker on the img and the bounding box
        trackerInfo = tracker.update(image)
        success = trackerInfo[0]
        bbox = trackerInfo[1]

        # Call drawBox() if the tracking is successful
        if success:
            drawBox(image, bbox)
        else:
            cv2.putText(image, "Lost", (75, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Call goalTrack() function
        goalTrack(image, bbox)

    cv2.imshow('Image', image)
    cv2.waitKey(1)

    key = cv2.waitKey(25)
    if key == 32:
        print("Stopped")
        break
