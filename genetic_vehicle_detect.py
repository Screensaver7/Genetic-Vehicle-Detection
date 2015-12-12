##
## genetic_vehicle_detect.py
## CS482
## Arron Bao Van, Zhaozhuo Li
##
## in:  # of iterations, # of population
## out: best values for blob detector to maximize vehicle detection
##
## Usage: python genetic_vehicle_detect.py 10 5
##

import cv2
import random
import threading
import scipy as sp
import numpy as np
from sys import argv
from datetime import datetime
#Put input folder on same directory

##
## Threading to calculate score
##
## in:  # of iterations, # of population, data of attribs
## out: thread object
##
class myThread (threading.Thread):
    def __init__(self, iters, numchldrn, attrib):
        threading.Thread.__init__(self)
        self.iters = iters
        self.numchldrn = numchldrn
        self.attrib = attrib
        self.score = 0
    def run(self):
        self.score = genetic(self.iters, self.numchldrn, self.attrib)
##
## Loads images and blob detects with different parameters
##
## in:  # of iterations, # of population, data of attribs
## out: score of blob detection
##
def genetic(gen, child, attrib):
    name = "gen" + str(gen) + "," + str(child)
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

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)

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

            #load gt in grayscale
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
            cv2.imshow(name, img1)
            #cv2.waitKey(67) waits for 0.067 seconds making this ~15fps
            #Stop loop with "q"
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            out = out + 1
        else: break

    #returns precision + recall
    return score

    cap.release()
##
## Mutates parameters randomly
##
## in: data of parameters
## out: data of parameters
##
def mutate(orig):
    keys = orig.keys()
    mutator = generateC()
    mutations = random.randint(0,len(keys)/2)
    swap = random.sample(range(0, len(keys)), mutations)
    for i in swap:
        orig[keys[i]] = mutator[keys[i]]
    return orig
##
## Generates children randomly or with two parents
##
## in: parent1 and parent2, or nothing
## out: 1 child
##
def generateC(parent1={}, parent2={}):
    #new child from parents
    if (len(parent1) != 0 and len(parent2) != 0):
        keys = parent1.keys()
        values = parent1.values()
        rtn = dict(zip(keys, values))
        swap = random.sample(range(0, len(parent1)), len(parent1)/2)
        for i in swap:
            rtn[keys[i]] = parent2[keys[i]]
        mutate(rtn)
        return rtn
    #child w\ random params
    else:
        rtn = {'minT': 0, 'maxT': 0, 'minA': 0, 'maxA': 0,
               'minCov': 0.0, 'filt': 1, 'sigma': 0.0}
        #blob detector threshold constraints
        tcutoff = random.randint(10,245)
        rtn['minT'] = random.randint(0,tcutoff)
        rtn['maxT'] = random.randint(tcutoff,255)
        #blob detector area constraints
        acutoff = random.randint(50,1000)
        rtn['minA'] = random.randint(0,acutoff)
        rtn['maxA'] = random.randint(acutoff,4000)
        #blob detector convexity constraints
        rtn['minCov'] = random.uniform(.3,1.0)
        #guassian filter constraints
        rtn['filt'] = random.choice(range(3, 25, 2))
        #guassian sigma constraints
        rtn['sigma'] = random.uniform(5.0,50.0)
        return rtn
##
## Generates a whole population, keeps parents in next iteration
##
## in: first 2 parents, # of population needed
## out: entire population
##
def generateP(poparr, pop):
    rtn = []
    #find the top instance
    scores = [row[1] for row in poparr]
    ind = scores.index(max([row[1] for row in poparr]))
    rtn.append(poparr[ind])
    del poparr[ind]
    #find second top instance
    while (len(poparr) != 0):
        scores = [row[1] for row in poparr]
        ind = scores.index(max([row[1] for row in poparr]))
        #delete if duplicate
        if (abs(poparr[ind][1] - rtn[0][1]) < 0.0001):
            del poparr[ind]
        else:
            rtn.append(poparr[ind])
            break
    if (len(poparr) == 0 and len(rtn) == 1):
        rtn.append([generateC(), -1.0])
    #generate population w\ parents
    while(len(rtn) < pop):
        rtn.append([generateC(rtn[0][0],rtn[1][0]),-1.0])
    return rtn

if __name__ == "__main__":
    if (len(argv) < 3):
        print "Not enough parameters entered."
        exit(1)
    gen = int(argv[1])
    pop = int(argv[2])
    random.seed(datetime.now())
    #first parents
    poparr = [[generateC(), -1.0], [generateC(), -1.0]]
    cv2.startWindowThread()
    for i in range(gen):
        #fill population
        poparr = generateP(poparr, pop)
        print "\nGeneration %d:" % i
        for p in poparr:
            print p
        #since parents carry over, skip them after first iteration
        start = 0
        if (i != 0):
            start = 2
        for x in range(start, pop, 3):
            threads = []
            #only 3 threads
            for c in range(3):
                if (x + c >= pop): break
                thread = myThread(i, x+c, poparr[x+c][0])
                thread.start()
                threads.append(thread)
            for t in range(len(threads)):
                threads[t].join()
                poparr[x+t][1] = threads[t].score
            threads[:] = []
            cv2.destroyAllWindows()
        print "\nGeneration %d Results:" % i
        for p in poparr:
            print p
    cv2.destroyAllWindows()
    print "\n Final iteration result:\n", generateP(poparr, 2)

