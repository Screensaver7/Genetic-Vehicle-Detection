## Simply outputs a video of bounding boxes

import cv2
import scipy as sp
import numpy as np

cap = cv2.VideoCapture()
#http://www.chart.state.md.us/video/video.php?feed=13015dbd01210075004d823633235daa
#Use this until we find a better traffic camera
cap.open('./highway/input/in%06d.jpg')
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('X','V','I','D'), 60, (w,h), True)
#bg subtractor
fgbg = cv2.createBackgroundSubtractorKNN()
#parameters for blob detector
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 0
params.maxThreshold = 123
params.filterByArea = True
params.minArea = 5
params.maxArea = 3384
params.filterByConvexity = True
params.minConvexity = 0.5622
detector = cv2.SimpleBlobDetector_create(params)

cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Stream", w, h)

confmat = np.zeros((2,2))
score = float(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        img1 = sp.zeros(frame.size, sp.uint8)
        img1 = frame
        #bg subtract
        img2 = fgbg.apply(frame)
        #invert image
        cv2.bitwise_not(img2, img2)
        #gaussian
        img2 = cv2.GaussianBlur(img2,(19, 19),6.5)
        #blob detect
        points = detector.detect(img2)
        #draw bounding boxes
        for p in points:
            x1 = int(p.pt[0]-p.size/2)
            y1 = int(p.pt[1]-p.size/2)
            x2 = int(p.pt[0]+p.size/2)
            y2 = int(p.pt[1]+p.size/2)
            cv2.rectangle(img1,(x1,y1),(x2,y2),(0,0,255),1)

        cv2.imshow("Stream", img1)
        out.write(img1)
        #cv2.waitKey(67) waits for 0.067 seconds making this ~15fps
        #Stop loop with "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else: break

cap.release()
out.release()
cv2.destroyAllWindows()
