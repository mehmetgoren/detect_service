import numpy as np
from typing import List
import jetson.inference as inf
import jetson.utils as utils

#    -- ClassID: 72
#    -- Confidence: 0.722135
#    -- Left:    91.0593
#    -- Top:     247.185
#    -- Right:   513.588
#    -- Bottom:  512.167
#    -- Width:   422.528
#    -- Height:  264.983
#    -- Area:    111963

import cv2

net = inf.detectNet("ssd-mobilenet-v2", threshold=0.5)

# user = 'admin'
# pwd = 'admin123456'
# ip = '192.168.0.100'
# port = 8554
# route = 'profile0'
# url = f'rtsp://{user}:{pwd}@{ip}:{port}/{route}'

camera_no = 4
user = 'admin'
pwd = 'a12345678'
ip = '192.168.0.108'
port = '554'
subtype = 0  # 0 is main stream, 1 is extra stream
url = f'rtsp://{user}:{pwd}@{ip}:{port}/cam/realmonitor?channel={camera_no}&subtype={subtype}'

cap = cv2.VideoCapture(url)

while True:
    success, img = cap.read()
    img_rgb = img #cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_cuda = utils.cudaFromNumpy(img_rgb)
    detections = net.Detect(img_cuda, overlay='OVERLAY_NONE')

    for d in detections:
        x1, y1, x2, y2 = int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)
        class_name = net.GetClassDesc(d.ClassID)
        cv2.rectangle(img, (x1, y1), (x2,y2), (255,0,255),2)
        cv2.putText(img, class_name, (x1+5, y1+15), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255,0,255), 2)
        cv2.putText(img, f'FPS: {int(net.GetNetworkFPS())}', (30,30), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255,0,255), 2)


    #img = utils.cudaToNumpy(img_cuda)
    cv2.imshow("image", img)
    cv2.waitKey(1)
