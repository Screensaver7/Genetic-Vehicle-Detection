import cv2
import scipy as sp

cap = cv2.VideoCapture()
cap.open('rtmp://itsvideo.arlingtonva.us:8001/live/cam36.stream')

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        #img1 = sp.zeros(frame.shape, sp.uint8)
        cv2.imshow("Stream", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else: break
 
