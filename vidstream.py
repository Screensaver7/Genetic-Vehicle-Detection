import cv2
import numpy as np
#Put input folder on same directory

cap = cv2.VideoCapture()
#http://www.chart.state.md.us/video/video.php?feed=13015dbd01210075004d823633235daa
#Use this until we find a better traffic camera
cap.open('./highway/input/in%06d.jpg')
#cap.open('./highway/groundtruth/gt%06d.png')

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*2)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*2)

fgbg = cv2.createBackgroundSubtractorKNN()
detector = cv2.SimpleBlobDetector_create()

cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Stream", w, h)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        img1 = fgbg.apply(frame)
        cv2.bitwise_not(img1, img1)
        img1 = cv2.GaussianBlur(img1, (11, 11), 30)
        points = detector.detect(img1)
        img1 = cv2.drawKeypoints(img1, points, np.array([]),
                                 (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #Canny edge detector
        cv2.imshow("Stream", img1)
        #cv2.waitKey(67) waits for 0.067 seconds making this ~15fps
        #Stop loop with "q"
        if cv2.waitKey(17) & 0xFF == ord('q'):
            break
    else: break
    
cap.release()
cv2.destroyAllWindows()
