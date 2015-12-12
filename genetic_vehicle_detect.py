##
## genetic_vehicle_detect.py
## CS482
## Arron Bao Van, Zhaozhuo Li
##
## in:  # of generations, # of children each generation
## out: best values for blob detector to maximize vehicle detection
##
## Usage: python genetic_vehicle_detect.py 10 5
##

import cv2
import random
import scipy as sp
import numpy as np
from sys import argv
#Put input folder on same directory

def main(gen, child, attrib):
    cap = cv2.VideoCapture()
    #http://www.chart.state.md.us/video/video.php?feed=13015dbd01210075004d823633235daa
    #Use this until we find a better traffic camera
    cap.open('./highway/input/in%06d.jpg')

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #bg subtractor
    fgbg = cv2.createBackgroundSubtractorKNN()
    #parameters for blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = attrib['minT']
    params.maxThreshold = attrib['maxT']
    params.filterByArea = True
    params.minArea = attrib['minA']
    params.maxArea = attrib['maxA']
    params.filterByConvexity = True
    params.minConvexity = attrib['minCov']
    detector = cv2.SimpleBlobDetector_create(params)
    
    cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Stream", w, h)

    confmat = np.zeros((2,2))
    out = 1
    score = float(0)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if(confmat[0][1] != 0 and confmat[1][0] != 0):
                score = confmat[0][0]/float(confmat[0][0]+confmat[1][0])+\
                        confmat[0][0]/float(confmat[0][0]+confmat[0][1])
            #bg subtract
            img1 = fgbg.apply(frame)
            #white bg for bbox
            imgbbox = sp.ones((240,320), sp.uint8) * 255
            #invert image
            cv2.bitwise_not(img1, img1)
            #gaussian
            img1 = cv2.GaussianBlur(img1,(attrib['filt'], attrib['filt']),
                                    attrib['sigma'])
            #blob detect
            points = detector.detect(img1)
            #draw bounding boxes
            for p in points:
                x1 = int(p.pt[0]-p.size/2)
                y1 = int(p.pt[1]-p.size/2)
                x2 = int(p.pt[0]+p.size/2)
                y2 = int(p.pt[1]+p.size/2)
                cv2.rectangle(img1,(x1,y1),(x2,y2),1)
                cv2.rectangle(imgbbox,(x1,y1),(x2,y2),(0,0,0),cv2.FILLED)

            #load gt in grascale
            img_gt = cv2.imread("./highway/groundtruth/gt"+str(out).zfill(6)+".png", 0)
            cv2.bitwise_not(img_gt, img_gt)

            #everything in gt before 470 is blank
            if (out > 470):
                #comparing every 5x, 5y
                for x in range(0,len(img_gt),5):
                    for y in range(0,len(img_gt[x]),5):
                        if (img_gt[x][y] == 0 and imgbbox[x][y] == 0):
                            confmat[0][0] = confmat[0][0] + 1 #TP
                        elif (img_gt[x][y] != 0 and imgbbox[x][y] == 0):
                            confmat[0][1] = confmat[0][1] + 1 #FP
                        elif (img_gt[x][y] == 0 and imgbbox[x][y] != 0):
                            confmat[1][0] = confmat[1][0] + 1 #FN

            strinfo = 'Gen: %(gen)d, Child: %(ch)d' % {"gen":gen, "ch":child}
            cv2.putText(img1, strinfo, (5,235),0,0.4, 0);
            strscore = 'Score: %(sc).5f' % {"sc": score}
            cv2.putText(img1, strscore, (5,12),0,0.4, 0);
            cv2.imshow("Stream", img1)
            #cv2.waitKey(67) waits for 0.067 seconds making this ~15fps
            #Stop loop with "q"
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            out = out + 1
        else: break

    #returns precision + recall
    return score, attrib

    cap.release()

if __name__ == "__main__":
    gen = argv[1]
    numchld = argv[2]
    for i in range(1):
        cv2.startWindowThread()
        params = {'minT': 100, 'maxT': 200, 'minA': 200, 'maxA': 1800,
                  'minCov': 0.78, 'filt': 9, 'sigma': 20}
        print main(1, 10, params)
    cv2.destroyAllWindows()
