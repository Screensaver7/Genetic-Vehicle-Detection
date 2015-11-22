import cv2
import scipy as sp

cap = cv2.VideoCapture()
#http://www.chart.state.md.us/video/video.php?feed=13015dbd01210075004d823633235daa
#Use this until we find a better traffic camera
cap.open('rtmp://170.93.143.139:1935/rtplive/13015dbd01210075004d823633235daa')

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*1.5)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*1.5)

cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Stream", w, h)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        #Copying frame
        img1 = sp.zeros(frame.shape, sp.uint8)
        img1 = cv2.GaussianBlur(frame, (3, 3), 7)
        #Canny edge detector
        img2 = cv2.Canny(img1, 50, 90)
        cv2.imshow("Stream", img2)
        #cv2.waitKey(67) waits for 0.067 seconds making this ~15fps
        #Stop loop with "q"
        if cv2.waitKey(67) & 0xFF == ord('q'):
            break
    else: break
    
cap.release()
cv2.destroyAllWindows()
