import numpy as np
import cv2

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
    
    def __init__(self, show_images, min_area):
        
        self.mogSubtractor = cv2.createBackgroundSubtractorMOG2(10, 16, False)
        self.initedAcc = False
        self.accumulater = np.float32()
        self.showImages = show_images
        self.minArea = min_area

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

    def findObjects(self, frame, tracker):
        expandAmount = 100
        minArea1 = 75
        
        if (self.initedAcc == False):   
            self.accumulater = np.float32(frame)
            self.initedAcc = True

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray1 = np.float32(gray)
        thresh = np.float32(gray)
#        frameDelta = np.float32(gray)


        cv2.accumulateWeighted(frame, self.accumulater, 0.3)
        res1 = cv2.convertScaleAbs(self.accumulater);
        gray1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY);
        #frameDelta = cv2.absdiff(gray1, gray);

        bgsub = self.mogSubtractor.apply(gray1)

        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(bgsub, kernel, iterations=1)

        ret, thresh = cv2.threshold(dilated, 60, 255, 0) 

            
#        if self.showImages == True:
#            cv2.imshow('thresh', thresh)

        _, contours, hierarchy = cv2.findContours(thresh,
                                                  cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)


        centers = []  
        

        height = frame.shape[0]
        width = frame.shape[1]
        #height = 2160
        #width = 3840
        #areas = self.getLargeAreas(contours, 720, 1280)
        contourRects = []
        for cnt in contours:
            a = cv2.contourArea(cnt, False)
            if (a > minArea1 and a < ((height * width) / 4)):
                x, y, w, h = cv2.boundingRect(cnt)
#                sizeincrease = r.area() / 200
#                r -= cv2.Point(sizeincrease, sizeincrease)
#                r += cv2.Size(sizeincrease * 2, sizeincrease * 2)
#            contourRects.append(expand(x, y, w, h, 20))
                x, y, w, h = self.expand(x, y, w, h, expandAmount)
                contourRects.append([x, y, x + w, y + h])
                if self.showImages == True:
                    cv2.rectangle(frame,(int(x),int(y)),(int(x)+int(w),int(y)+int(h)),(0,255,0),2)


        val = 0.1
        test = non_max_suppression_fast(np.array(contourRects), val)
        for rect in test:
            x1 = rect[0]
            y1 = rect[1]
            x2 = rect[2]
            y2 = rect[3]
            area = abs(x1 - x2) * abs(y1 - y2)
            if area > self.minArea:
                if self.showImages == True:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                b = np.array([[(x1 + x2) / 2], [(y1 + y2) / 2], [area]])
                centers.append(np.round(b))

        if self.showImages == True:
            for i in range(len(tracker.tracks)):
                if tracker.tracks[i].firm_track_count > 7:
                    for j in range(len(tracker.tracks[i].trace)-1):
                        if j < 1:
                            x1, y1 = tracker.tracks[i].trace[j]
                            continue
                        x2, y2 = tracker.tracks[i].trace[j + 1]
                        cv2.line(frame, (x1, y1), (x2, y2), (255,0,0), 5)
                        x1 = x2
                        y1 = y2

            cv2.imshow('image', frame)
            cv2.waitKey(1)
        
        return centers
