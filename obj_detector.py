import numpy as np
import cv2
import time

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")


class ObjDetector(object):
    
    def __init__(self, show_images, write_video, min_area, verbose):
        
        self.mogSubtractor = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=20, detectShadows=False)
        self.initedAcc = False
        self.accumulater = np.float32()
        self.showImages = show_images
        self.writeVideo = write_video
        self.minArea = min_area
        self.verbose = verbose

    def expand_old(self, x, y, w, h, size):
        cenX = x + (w / 2)
        cenY = y + (h / 2)
        x = cenX - (size / 2)
        w = size
        y = cenY - (size / 2)
        h = size
        return x, y, w, h

    def expand(self, x, y, w, h, size):
        x = x - (size / 2)
        w = w + size
        y = y - (size / 2)
        h = h + size
        return x, y, w, h

    def getLargeAreas(self, contours, height, width):
        contourRects = []
        for cnt in contours:
            a = cv2.contourArea(cnt, False)
            if (a > 50 and a < ((height * width) / 4)):
                r = cv2.boundingRect(cnt)
                sizeincrease = r.area() / 200
            r -= cv2.Point(sizeincrease, sizeincrease)
            r += cv2.Size(sizeincrease * 2, sizeincrease * 2)
            contourRects.append(r)
        return contourRects

    def sortFunc(self, r):
        #[x, y, x + w, y + h]
        return (r[4])

    def findObjects(self, frame, tracker):
        expandAmount = 20
        minArea1 = 150
        
        blurValue = 19
        learningRate = 0.15
        dilateValue = 10


        height = frame.shape[0]
        width = frame.shape[1]
        
#        if (self.initedAcc == False):   
#            self.accumulater = np.float32(frame)
#            self.initedAcc = True
        
        # converting to gray here makes it faster, but we lose a lot of
        # detections with the mog subtractor
        gray = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #gray1 = np.float32(gray)
        #thresh = np.float32(gray)
#        frameDelta = np.float32(gray)

        #cv2.accumulateWeighted(frame, self.accumulater, 0.9)
        #res1 = cv2.convertScaleAbs(self.accumulater);
        #gray1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY);
        #gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
        #frameDelta = cv2.absdiff(frame, res1);
        #gray2 = cv2.cvtColor(frameDelta, cv2.COLOR_BGR2GRAY);

        centers = []  
        
        if self.verbose:
            t1 = time.perf_counter()
        # not sure why, but adding in this first blur before the GaussianBlur helps
        # this is the 2nd slowest
        src_gray = cv2.blur(gray, (3,3))
        if self.verbose:
            print("Blur1 time: " + str(time.perf_counter() - t1))
        
        if self.verbose:
            t1 = time.perf_counter()
        # this is the 3rd slowest
        src_gray = cv2.GaussianBlur(src_gray, (blurValue, blurValue), 0)
        if self.verbose:
            print("Blur2 time: " + str(time.perf_counter() - t1))

        if self.verbose:
            t1 = time.perf_counter()
        # this is the slowest operation
        bgsub = self.mogSubtractor.apply(src_gray, learningRate = learningRate)#gray1)
        if self.verbose:
            print("MOG time: " + str(time.perf_counter() - t1))

        if self.showImages == True:
            cv2.imshow('bgsub', bgsub)

        if self.verbose:
            t1 = time.perf_counter()
        diffPercent = 100.0 * cv2.countNonZero(bgsub) / (height * width)
        # too much difference, just skip this frame
        if diffPercent > 40:
            if self.verbose:
                print("SKIPPED FRAME")
            return centers
        if self.verbose:
            print("Diff Check time: " + str(time.perf_counter() - t1))

        if self.verbose:
            t1 = time.perf_counter()
        kernel = np.ones((dilateValue,dilateValue), np.uint8)
        dilated = cv2.dilate(bgsub, kernel, iterations=2)
        # this erode step could be removed if speed becomes an issue, it helps, but not that much
        dilated = cv2.erode(dilated, kernel, iterations=2)
        if self.verbose:
            print("dilate & erode time: " + str(time.perf_counter() - t1))
        
#        se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
#        bg=cv2.morphologyEx(dilated, cv2.MORPH_DILATE, se)
#        out_gray=cv2.divide(dilated, bg, scale=255)
#        out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1] 

#        if self.showImages == True:
#            cv2.imshow('thresh', dilated)

        if self.verbose:
            t1 = time.perf_counter()
        contours, hierarchy = cv2.findContours(dilated, 
                                                  cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)[-2:]
        if self.verbose:
            print("contours time: " + str(time.perf_counter() - t1))

        if self.showImages == True:
            img = cv2.drawContours(frame, contours, -1, (0, 255, 75), 2)
            cv2.imshow('contours', img)



        if self.verbose:
            t1 = time.perf_counter()
        #areas = self.getLargeAreas(contours, 720, 1280)
        contourRects = []
        for cnt in contours:
            a = cv2.contourArea(cnt, False)
            if (a > minArea1 and a < ((height * width) / 4)):
                x, y, w, h = cv2.boundingRect(cnt)
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                x, y, w, h = self.expand(int(cX - (w/2)), int(cY - (h/2)), w, h, expandAmount)
                contourRects.append([x, y, x + w, y + h, a])
                if (self.showImages == True) | (self.writeVideo == True):
                    #cv2.rectangle(frame,(int(x),int(y)),(int(x)+int(w),int(y)+int(h)),(0,255,0),2)
                    cv2.circle(frame,(cX, cY),7, (0, 255, 0),-1)

        contourRects.sort(reverse=True, key=self.sortFunc)
        # only proceed with the largest 40 areas
        if len(contourRects) > 40:
            contourRects = contourRects[:40]
            if self.verbose:
                print("reduced number of rects")
            
        val = 0.1
        test = non_max_suppression_fast(np.array(contourRects), val)
        #test = cv2.groupRectangles(contourRects, 1, 5)
        if len(test) > 0:
#            for rect in test[0]:
            for rect in test:
                x1 = rect[0]
                y1 = rect[1]
                x2 = rect[2]
                y2 = rect[3]
                area = abs(x1 - x2) * abs(y1 - y2)
                if area > self.minArea:
                    if (self.showImages == True) | (self.writeVideo == True):
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                    #x-center, y-center, x-ul, y-ul, width, height, area
                    xcenter = int((x1 + x2) / 2)
                    ycenter = int((y1 + y2) / 2)
                    xul = x1
                    yul = y1
                    width = x2 - x1
                    height = y2 - y1
                    area2 = rect[4]
                    b = np.array([xcenter, ycenter, xul, yul, width, height, area2])
                    centers.append(b)

        if self.verbose:
            print("find rects time: " + str(time.perf_counter() - t1))
        
        if (self.showImages == True) | (self.writeVideo == True):
            for i in range(len(tracker.tracks)):
                if tracker.tracks[i].firm_track_count > 10:
                    for j in range(len(tracker.tracks[i].trace)-1):
                        if j < 1:
                            x1, y1 = tracker.tracks[i].trace[j]
                            continue
                        x2, y2 = tracker.tracks[i].trace[j + 1]
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 5)
                        x1 = x2
                        y1 = y2
            if self.showImages == True:
                cv2.imshow('image', frame)
                cv2.waitKey(1)
        
        return centers
