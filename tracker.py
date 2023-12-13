import cv2
import os
import math
import numpy as np
import statistics
from collections import Counter
import copy
from scipy.optimize import linear_sum_assignment


class Track(object):
    def __init__(self, initLoc, trackIdCount):
        self.track_id = trackIdCount
        self.KF = cv2.KalmanFilter(4, 2)
        self.KF.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]],np.float32)
        self.KF.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]],np.float32)
        self.KF.processNoiseCov = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]],np.float32) * 0.3
        initX = initLoc[0]
        initY = initLoc[1]
        initArea = initLoc[6]
        self.boxes= []
        self.boxes.append([initLoc[2], initLoc[3], initLoc[4], initLoc[5]])
        self.prediction = [initX, initY]
        self.vel_prediction = [0, 0]
        self.skipped_frames = 0 
        self.trace = [] 
        self.trace.append([initX, initY])
        self.areas = []
        self.areas.append(initArea)
        self.init = True
        self.images = []
        self.croppedImages = []
        self.obj_type_predictions = []
        self.firm_track_count = 0
        self.write_initial_frames = True
        self.prediction_history = []
        self.prediction_history.append([initX, initY])
        self.vel_prediction_history = []
        self.vel_prediction_history.append([0, 0])
        self.trajectories = []
        self.trajectories.append(0)
        self.pred_trajectories = []
        self.pred_trajectories.append(0)
        self.distances = []
        self.distances.append(0)
        self.trackerstarted = False
        self.trajectory_diffs = []
        self.trajectory_diffs.append(0)
        self.empty_image_count = 0


class Tracker(object):
    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length, trackIdCountStart, store_images, write_images, base_filename, output_dir):
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCountStart
        self.storeImages = store_images
        self.writeImages = write_images
        self.baseFilename = base_filename
        self.outDir = output_dir
        self.DEBUG = False
        self.trackers = []

    # this method is a hack - I really want to initialize the KalmanFilter with the
    # location of the first point, but I can't figure out how to do that, this seems
    # to accomplish it
    def initKF(self, i):
        x, y = self.tracks[i].prediction;
        for j in range(10):
            self.tracks[i].KF.correct(np.array([np.float32(x), np.float32(y)], np.float32))
            self.tracks[i].KF.predict()
        self.tracks[i].init = False

    def writeImage(self, cropped_frame, x, y, frame_num, track_id, speed, area, ux, uy, width, height):
        if len(cropped_frame) > 1 and int(area) > 0:
            baseDir = os.path.join(self.outDir, self.baseFilename)
            if not os.path.exists(baseDir):
                os.makedirs(baseDir)
            trackDir = os.path.join(baseDir, str(track_id))
            if not os.path.exists(trackDir):
                os.makedirs(trackDir)
            file = os.path.join(trackDir, self.baseFilename + '_v2_' + str(frame_num) + '_' + str(int(x)) + '_' + str(int(y)) + '_' + str("%.5f" % speed) + '_' + str(int(area)) + '_' + str(track_id) + '_' + str(int(ux)) + '_' + str(int(uy)) + '_' + str(int(width)) + "_" + str(int(height)) + '.png')
            #print('writing file: ' + file)
            cv2.imwrite(file, cropped_frame)
        else:
            print("skipped write image " + str(len(cropped_frame)) + ", " + str(area) + ", x=" + str(x) + ", y=" + str(y))
            if (len(cropped_frame) < 2):
                print(cropped_frame)
        

    def get100SizeCrop(self, frame, cenX, cenY):
        height = frame.shape[0]
        width = frame.shape[1]
        expandNegX = 0
        expandNegY = 0
        expandPosX = 0
        expandPosY = 0

        uxpt = int(cenX - 100)
        lxpt = int(uxpt + 200)
        uypt = int(cenY - 100)
        lypt = int(uypt + 200)

        if uxpt < 0:
            expandNegX = 0 - uxpt
            uxpt = 0
        if uxpt > width:
            return np.zeros([200, 200, 3], dtype=np.uint8)

        if lxpt < 0:
            return np.zeros([200, 200, 3], dtype=np.uint8)
        if lxpt > width:
            expandPosX = lxpt - width
            lxpt = width

        if uypt < 0:
            expandNegY = 0 - uypt
            uypt = 0
        if uypt > height: 
            return np.zeros([200, 200, 3], dtype=np.uint8)

        if lypt < 0: 
            return np.zeros([200, 200, 3], dtype=np.uint8)
        if lypt > height:
            expandPosY = lypt - height
            lypt = height

        cropped_frame = frame[uypt:lypt, uxpt:lxpt]
        if (expandNegX > 0) | (expandNegY > 0) | (expandPosX > 0) | (expandPosY > 0):
            cropped_frame = cv2.copyMakeBorder(cropped_frame, top=expandNegY, bottom=expandPosY, left=expandNegX, right=expandPosX, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if (cropped_frame.shape[0] < 200) | (cropped_frame.shape[1] < 200):
            raise ValueError("Crop error")
        return cropped_frame

    def getCroppedImage(self, frame, ulx, uly, width, height):
        if (width == 0) & (height == 0):
            return np.zeros([1, 1, 3], dtype=np.uint8)


        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        expandNegX = 0
        expandNegY = 0
        expandPosX = 0
        expandPosY = 0

        uxpt = ulx
        lxpt = ulx + width
        uypt = uly
        lypt = uly + height

        if uxpt < 0:
            expandNegX = 0 - uxpt
            uxpt = 0
        if uxpt > frameWidth:
            return np.zeros([1, 1, 3], dtype=np.uint8)
        
        if lxpt < 0:
            return np.zeros([1, 1, 3], dtype=np.uint8)
        if lxpt > frameWidth:
            expandPosX = lxpt - frameWidth
            lxpt = frameWidth
            
        if uypt < 0:
            expandNegY = 0 - uypt
            uypt = 0
        if uypt > frameHeight: 
            return np.zeros([1, 1, 3], dtype=np.uint8)
        
        if lypt < 0: 
            return np.zeros([1, 1, 3], dtype=np.uint8)
        if lypt > frameHeight:
            expandPosY = lypt - frameHeight
            lypt = frameHeight

        cropped_frame = frame[uypt:lypt, uxpt:lxpt]
        if (expandNegX > 0) | (expandNegY > 0) | (expandPosX > 0) | (expandPosY > 0):
            cropped_frame = cv2.copyMakeBorder(cropped_frame, top=expandNegY, bottom=expandPosY, left=expandNegX, right=expandPosX, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if (cropped_frame.shape[0] < 1) | (cropped_frame.shape[1] < 1):
            raise ValueError("Crop error")
        return cropped_frame
        

    def get_ave_dir_change(self, track):
        count = 0
        total = 0
        numbig = 0            
        for i in range (len(track.trajectory_diffs)):
            index = -(i + 1)
            tj = track.trajectory_diffs[index]
            total += tj
            count += 1
            if tj > 90:
                numbig += 1
            if count > 10:
                break

        if count > 0:
            return (numbig, (total / count))
        else:
            return (numbig, 0)
        
    def angle_between(self, p1, p2):
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        return np.rad2deg((ang1 - ang2) % (2 * np.pi))

    def calc_angle(self, dx, dy):
        rads = math.atan2(dy, dx)
        degs = math.degrees(rads)
        adj = 360 - (degs - 90)
        if adj >= 360:
            return adj - 360
        return adj        

    def calc_ave_dist(self, track, x, y, num):
        if (len(track.trace) <= 2):
            lastX, lastY = track.trace[-1]
            xDiff = x - lastX
            yDiff = y - lastY
            dist = np.sqrt(xDiff * xDiff + yDiff * yDiff)
            return dist
        
        total = 0
        count = 0
        for i in range(len(track.distances) - 1):
            index = -(i + 1)
            total += track.distances[index]
            count += 1
            # just get the average of the previous 5 points
            if count > num:
                break

        if count > 0:
            return total / count
        else:
            return 0

    def calc_ave_area(self, areas, num):
        total = 0
        count = 0
        for i in range(len(areas)):
            index = -(i + 1)
            total += areas[index]
            count += 1
            # just get the average of the previous 5 points
            if count > num:
                break

        if count > 0:
            return total / count
        else:
            return 0

    def get_rect(self, x, y, area):
        l = math.sqrt(area)
        ux = x - (l/2)
        uy = y - (l/2)
        return (int(ux), int(uy), int(l), int(l))
        
    def get_center(self, ux, uy, width, height):
        cx = ux + (width / 2)
        cy = uy + (height / 2)
        return (int(cx), int(cy))

    def isclose(self, x1, y1, x2, y2):
        xDiff = x1 - x2
        yDiff = y1 - y2
        distance = np.sqrt(xDiff * xDiff + yDiff * yDiff)
        return (distance < 50)

    def L2Norm(self, H1, H2):
        distance = 0
        for i in range(len(H1)):
            distance += np.square(H1[i]-H2[i])
        return np.sqrt(distance)

    def removeLastImages(self, track, num):
        baseDir = os.path.join(self.outDir, self.baseFilename)
        if os.path.exists(baseDir):
            trackDir = os.path.join(baseDir, str(track.track_id))
            if os.path.exists(trackDir):
                dirlist = os.listdir(trackDir)
                if (len(dirlist) > num):
                    for filename in sorted(dirlist)[-num:]:
                        filename_relPath = os.path.join(trackDir,filename)
                        os.remove(filename_relPath)

    def updateTracks(self, detections, frame, frame_num):
        height = frame.shape[0]
        width = frame.shape[1]

        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount)
                detX, detY, ulX, ulY, dWidth, dHeight, area = detections[i]
                cropped_frame = self.get100SizeCrop(frame, detX, detY)
                if (len(cropped_frame) < 1):
                    # I don't think this should happen anymore
                    self.tracks[i].empty_image_count += 1
                track.images.append(cropped_frame)
                cropped_frame_actual = self.getCroppedImage(frame, ulX, ulY, dWidth, dHeight)
                track.croppedImages.append(cropped_frame_actual)
                self.trackIdCount += 1
                self.tracks.append(track)
            return

        # first see if any of the detections match firm trackers
        # if so don't process them further
        # NOTE: the built in opencv image trackers don't work very well for anything but large birds
        # so this is all commented out, maybe we can figure out a way to include it later
#        show_frame = copy.copy(frame)        
#        detects_to_delete = []
#        for i in range(len(self.trackers)):
#            (success, box) = self.trackers[i].update(frame)
#            if success:
#                print("tracker success" + str(i))
#                x = box[0][0]
#                y = box[0][1]
#                w = box[0][2]
#                h = box[0][3]
#                cx, cy = self.get_center(x, y, w, h)
#                for i in range(len(detections)):
#                    if self.isclose(detections[i][0], detections[i][1], cx, cy):
#                        detects_to_delete.append(i)
#                        print("found close")
#                cv2.rectangle(show_frame,(int(x),int(y)),(int(x)+int(w),int(y)+int(h)),(255,0,0),2)
#                cv2.imshow('thresh', show_frame)
#            else:
#                print("tracker not success" + str(i))
#
#        if len(detects_to_delete) > 0:
#            for ele in sorted(detects_to_delete, reverse = True):
#                del detections[ele]

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
#        print("starting with M(detections)=" + str(M) + ", N(tracks)=" + str(N))
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            predX, predY = self.tracks[i].prediction
            prevArea = self.tracks[i].areas[-1]
            lastX, lastY = self.tracks[i].trace[-1]
            ci = self.tracks[i].croppedImages[-1]
            for j in range(len(detections)):
#                try:
                detX, detY, ulX, ulY, dWidth, dHeight, area = detections[j]
                xDiff = detX - predX
                yDiff = detY - predY
                distanceFromPred = np.sqrt(xDiff * xDiff + yDiff * yDiff)
                
                xDiff = detX - lastX
                yDiff = detY - lastY
                # distance from previous point to new detected point (not between new point and predicted point that we computed earlier)
                distanceFromLast = np.sqrt(xDiff * xDiff + yDiff * yDiff)

                # make the cost function take into account similar area
                areaRatio = 0
                if (area > 0) & (prevArea > 0):
                    if area < prevArea:
                        areaRatio = area / prevArea
                    else:
                        areaRatio = prevArea / area
                
                # another thing to try is make the cost function take into
                # account a similarity metric for the cropped image
#                dist_test_ref_1 = 0
                ci2 = self.getCroppedImage(frame, ulX, ulY, dWidth, dHeight)
                score = 0
                if (len(ci) > 0) & (len(ci2) > 0):
                    hist = cv2.calcHist([ci], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#                    hist = cv2.calcHist([ci], [0], None, [256], [0, 256])
#                        flat_array_1 = ci.flatten()
#                        RH1 = Counter(flat_array_1)
#                        H1 = []
#                        for k in range(256):
#                            if k in RH1.keys():
#                                H1.append(RH1[k])
#                            else:
#                                H1.append(0)
#                        
                    hist2 = cv2.calcHist([ci2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#                    hist2 = cv2.calcHist([ci2], [0], None, [256], [0, 256])
#                        flat_array_1 = ci2.flatten()
#                        RH1 = Counter(flat_array_1)
#                        test_H = []
#                        for k in range(256):
#                            if k in RH1.keys():
#                                test_H.append(RH1[k])
#                            else:
#                                test_H.append(0)
#                                
#                        dist_test_ref_1 = self.L2Norm(test_H, H1)
                    score = cv2.compareHist(hist, hist2, cv2.HISTCMP_CORREL)
                        
                # the +10 is to penalize newly established tracks in case an older one could match this detection
                # this is small but it helps a bit. I need to figure out a better way to handle this
#                print("t" + str(self.tracks[i].track_id) + "d" + str(j) + ", distanceFromPred: " + str(distanceFromPred) + ", areaComp: " + str((distanceFromPred / 4) * areaRatio) + ", histComp: " + str(score) + ", lastDist: " + str(distanceFromLast / 8))
                if len(self.tracks[i].trace) < 3:
#                    cost[i][j] = distanceFromPred + 20 - ((distanceFromPred / 4) * areaRatio) - ((distanceFromPred / 2) * score) + (distanceFromLast / 8)
                    cost[i][j] = distanceFromPred + 20 - ((distanceFromPred / 1.2) * score) - ((distanceFromPred / 16) * areaRatio) + (distanceFromLast / 6)
                else:                        
#                    cost[i][j] = distanceFromPred - ((distanceFromPred / 4) * areaRatio) - ((distanceFromPred / 2) * score) + (distanceFromLast / 8)
                    cost[i][j] = distanceFromPred - ((distanceFromPred / 1.2) * score) - ((distanceFromPred / 16) * areaRatio) + (distanceFromLast / 6)
#                print("total: " + str(cost[i][j]))
#                except:
#                    print("Exception!!")
#                    pass

        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # each row of the assignment array will be for a track object (the index = the index of the tracks array)
        # each row will be set to a value, the value is the index of the detections array for the
        # closest detection point

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                dt = self.dist_thresh
                # dist_thresh should vary based on track confidence (length of track)
                trackLen = len(self.tracks[i].trace)
                # and num skipped
                numSkipped = self.tracks[i].skipped_frames
                if numSkipped < 1:
                    if trackLen > 32:
                        dt = dt / 8
                    elif trackLen > 16:
                        dt = dt / 4
                    elif trackLen > 8:
                        dt = dt / 2
                elif numSkipped < 4:
                    if trackLen > 16:
                        dt = dt / 4
                    elif trackLen > 8:
                        dt = dt / 2
                else:
                    if trackLen > 8:
                        dt = dt / 2                    

                x, y, ulX, ulY, dWidth, dHeight, area = detections[assignment[i]]

                predX, predY = self.tracks[i].prediction
                xDiff = x - predX
                yDiff = y - predY
                distanceFromPred = np.sqrt(xDiff * xDiff + yDiff * yDiff)
                    
                lastX, lastY = self.tracks[i].trace[-1]
                xDiff = x - lastX
                yDiff = y - lastY
                # distance from previous point to new detected point (not between new point and predicted point that we computed earlier)
                distanceFromLast = np.sqrt(xDiff * xDiff + yDiff * yDiff)

                priorAveDistance = self.calc_ave_dist(self.tracks[i], x, y, 4)
                distanceDiff = np.abs(priorAveDistance - distanceFromLast)
                distanceRatio = 1
                if (distanceFromLast > 0) & (priorAveDistance > 0):
                    if distanceFromLast < priorAveDistance:
                        distanceRatio = distanceFromLast / priorAveDistance
                    else:
                        distanceRatio = priorAveDistance / distanceFromLast

                prevArea = self.tracks[i].areas[-1]
                priorAveArea = self.calc_ave_area(self.tracks[i].areas, 5)
                areaRatio = 1
                if (area > 0) & (priorAveArea > 0):
                    if area < priorAveArea:
                        areaRatio = area / priorAveArea
                    else:
                        areaRatio = priorAveArea / area

#                print("track " + str(self.tracks[i].track_id) + " assigned " + str(assignment[i]) + "[" + str(len(self.tracks[i].trace)) + ", " + str(distanceFromPred) + "," + str(distanceFromLast) + "," + str(priorAveDistance) + "," + str(area) + ", " + str(priorAveArea) + "]")
                if ((distanceFromPred > dt) | (distanceFromLast > 750)):
                    # so if it gets rejected because predicted is too far away
                    # we should next check to see if actual distance is really close
                    if (len(self.tracks[i].trace) > 5) & (distanceRatio > 0.9):
#                        print("TOO FAR FROM PREDICTED BUT KEPT BECAUSE CLOSE TO ACTUAL")
                        continue
                    else:
#                        print("Drop 1: " + str(distanceFromPred) + str(cost[i][assignment[i]]) + ", " + str(distanceRatio))
                        assignment[i] = -1
                        un_assigned_tracks.append(i)
                else:
                    if len(self.tracks[i].trace) > 1:
                        # prior trajectory from track object
                        priorTrejectory = self.tracks[i].trajectories[-1]
                        # trajectory to new detected point
                        trajectoryToDetected = self.calc_angle((x - lastX), (y - lastY))
                        trajDiff = np.abs(priorTrejectory - trajectoryToDetected)
                        if trajDiff > 180:
                            trajDiff = np.abs(trajDiff - 360)

                        if len(self.tracks[i].trace) == 3:
                            priorDistance = self.tracks[i].distances[-1]
                            r = 1
                            if priorDistance > distanceFromLast:
                                r = distanceFromLast / priorDistance
                            else:
                                r = priorDistance / distanceFromLast

                            if r < 0.4:
#                                print("Drop distanceRatio (len = 3): " + str(r) + ", " + str(distanceFromLast) + ", " + str(priorDistance) + ", " + str(self.tracks[i].distances[-1]))
                                assignment[i] = -1
                                un_assigned_tracks.append(i)
                            elif trajDiff > 100:
#                                print("Drop trajDiff (len = 3): " + str(trajDiff) + ", " + str(trajectoryToDetected) + ", " + str(priorTrejectory))
                                assignment[i] = -1
                                un_assigned_tracks.append(i)

                        if len(self.tracks[i].trace) > 3:
                            dr2 = 0
                            if (priorAveDistance > 0):
                                if (distanceFromLast > priorAveDistance):
                                    dr2 = (distanceFromLast / priorAveDistance)
                            # these are pretty big ratios, but it gets rid of a lot suprisingly
                            if trajDiff > 170:
#                                print("Drop trajDiff: " + str(priorTrejectory) + ", " + str(trajectoryToDetected) + ", " + str(trajDiff))
                                assignment[i] = -1
                                un_assigned_tracks.append(i)
                            elif areaRatio < 0.05:
#                                print("Drop areaRatio: " + str(areaRatio) + ", " + str(priorAveArea) + ", " + str(prevArea) + ", " + str(area))
                                assignment[i] = -1
                                un_assigned_tracks.append(i)
                            elif distanceRatio < 0.1:
#                                print("Drop distanceRatio: " + str(distanceRatio) + ", " + str(distanceFromLast) + ", " + str(priorAveDistance))
                                assignment[i] = -1
                                un_assigned_tracks.append(i)
                            # if both distance and traj are way differen this is probably not the same track
                            elif (distanceRatio < 0.1) & (trajDiff > 120):
#                                print("Drop distanceRatio & trajDiff")
                                assignment[i] = -1
                                un_assigned_tracks.append(i)
                            elif (dr2 > 3):
                                # if prior ave distance is small and suddenly this distance is large
                                # that is bad, other way around could be collision, but birds don't suddenly shoot
                                assignment[i] = -1
                                un_assigned_tracks.append(i)
                                
                                
                        
                        if (len(self.tracks[i].trace) > 15):
#                            print("trajDiff: " + str(trajDiff) + ", distRatio: " + str(distanceRatio) + ", areaRatio: " + str(areaRatio))
                            if (trajDiff > 70) & (distanceRatio < 0.2):
                                print("POSSIBLE COLLISION")
                            elif (trajDiff > 70) & (areaRatio < 0.2):
                                print("POSSIBLE COLLISION")
                            elif (distanceRatio < 0.2) & (areaRatio < 0.2):
                                print("POSSIBLE COLLISION")
                                
                        


        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
                if i not in assignment:
                    un_assigned_detects.append(i)

        # if we blindly start new tracks for all un-assigned detections then the new track could steal future detections
        # that should belong to an established track, need to figure this out...

        for i in range(len(assignment)):
            prevX, prevY = self.tracks[i].trace[-1]
            prevArea = self.tracks[i].areas[-1]
            x = 0
            y = 0
            objArea = 0
            # this means we found a detection to add to a track
            if(assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                x, y, ulX, ulY, dWidth, dHeight, area = detections[assignment[i]]
                self.tracks[i].trace.append([x, y])
                objArea = area
                self.tracks[i].areas.append(area)
                self.tracks[i].boxes.append([ulX, ulY, dWidth, dHeight])
                self.tracks[i].KF.correct(np.array([np.float32(x), np.float32(y)], np.float32))
                self.tracks[i].firm_track_count += 1
#                if self.tracks[i].skipped_frames > 0:
#                    self.tracks[i].skipped_frames -= 1
                self.tracks[i].skipped_frames = 0
                cropped_frame_actual = self.getCroppedImage(frame, ulX, ulY, dWidth, dHeight)
                self.tracks[i].croppedImages.append(cropped_frame_actual)
            else:
                # carry forward the prediction for a little while
                x, y = self.tracks[i].prediction
                self.tracks[i].trace.append([x, y])
                objArea = prevArea
                self.tracks[i].areas.append(prevArea)
                self.tracks[i].boxes.append(self.tracks[i].boxes[-1])
                self.tracks[i].KF.correct(np.array([np.float32(x), np.float32(y)], np.float32))
                self.tracks[i].skipped_frames += 1
                self.tracks[i].croppedImages.append(self.tracks[i].croppedImages[-1])

            priorTraj = self.tracks[i].trajectories[-1]
            # compute here the trajectory from prior point to this point
            traj = self.calc_angle((x - prevX), (y - prevY))
            self.tracks[i].trajectories.append(traj)
            
            trajDiff = np.abs(priorTraj - traj)
            if trajDiff > 180:
                trajDiff = np.abs(trajDiff - 360)
            self.tracks[i].trajectory_diffs.append(trajDiff)
            

            # compute the distance from the prior point to this point
            xDiff = x - prevX
            yDiff = y - prevY
            distance = np.sqrt(xDiff * xDiff + yDiff * yDiff)
            if len(self.tracks[i].distances) == 1:
                self.tracks[i].distances[0] = distance
            self.tracks[i].distances.append(distance)

            cropped_frame = self.get100SizeCrop(frame, x, y)
            if (len(cropped_frame) == 0):
                # I don't think this should happen anymore
                self.tracks[i].empty_image_count += 1
            self.tracks[i].images.append(cropped_frame)

            if self.tracks[i].init is True:
                self.initKF(i)

            prediction = self.tracks[i].KF.predict()

            if len(self.tracks[i].trace) > 3:
                predX = prediction[0,0]
                predY = prediction[1,0]
                velX = prediction[2,0]
                velY = prediction[3,0]
                if (predX >= 0) & (predY >= 0) & (predX <= width) & (predY <= height):
                    self.tracks[i].prediction = [predX, predY]
                    self.tracks[i].vel_prediction = [velX, velY]
                else:
                    if predX < 0:
                        predX = 0
                    if predX > width:
                        predX = width
                    if predY < 0:
                        predY = 0
                    if predY > height:
                        predY = 0
                    self.tracks[i].prediction = [predX, predY]
                    self.tracks[i].vel_prediction = [velX, velY]
#                    self.tracks[i].prediction = self.tracks[i].trace[-1]
#                    self.tracks[i].vel_prediction = [0, 0]
            else:
                self.tracks[i].prediction = self.tracks[i].trace[-1]
                self.tracks[i].vel_prediction = [0, 0]

            self.tracks[i].prediction_history.append(self.tracks[i].prediction)
            self.tracks[i].vel_prediction_history.append(self.tracks[i].vel_prediction)

            # compute here the predicted trajectory
            predX, predY = self.tracks[i].prediction
            predTraj = self.calc_angle((predX - x), (predY - y))
            self.tracks[i].pred_trajectories.append(predTraj)

            if self.writeImages:
                if self.tracks[i].firm_track_count > 10 and len(cropped_frame) > 1:
                    if self.tracks[i].write_initial_frames == True:
                        # then write out all previous images, but do only once
                        self.tracks[i].write_initial_frames = False
                        prev_frame_num = frame_num - len(self.tracks[i].images) - 1
                        for k in range(len(self.tracks[i].images)):
                            px, py = self.tracks[i].trace[k]
                            parea = self.tracks[i].areas[k]
                            pux, puy, pbWidth, pbHeight = self.tracks[i].boxes[k]
                            pspeed = self.tracks[i].vel_prediction_history[k]
                            pvel_mag = np.sqrt(np.power(pspeed[0], 2) + np.power(pspeed[1], 2))
                            pvel_mag = self.tracks[i].distances[k]
                            # todo, figure out how to get frame_num of previous better
                            self.writeImage(self.tracks[i].images[k], px, py, prev_frame_num, self.tracks[i].track_id, pvel_mag, parea, pux, puy, pbWidth, pbHeight)
                            prev_frame_num += 1
                            
                    ux, uy, bWidth, bHeight = self.tracks[i].boxes[-1]
                    speed = self.tracks[i].vel_prediction
                    vel_mag = np.sqrt(np.power(speed[0], 2) + np.power(speed[1], 2))
                    vel_mag = self.tracks[i].distances[-1]
                    self.writeImage(cropped_frame, x, y, frame_num, self.tracks[i].track_id, vel_mag, objArea, ux, uy, bWidth, bHeight)

            # only maintain history of a certain length
            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) - self.max_trace_length):
                    del self.tracks[i].trace[j]
                    del self.tracks[i].images[j]
                    del self.tracks[i].prediction_history[j]
                    del self.tracks[i].vel_prediction_history[j]
                    del self.tracks[i].pred_trajectories[j]
                    del self.tracks[i].trajectories[j]
                    del self.tracks[i].distances[j]
                    del self.tracks[i].trajectory_diffs[j]
                    del self.tracks[i].croppedImages[j]
                    del self.tracks[i].boxes[j]
                    del self.tracks[i].areas[j]


        del_tracks = []
        for i in range(len(self.tracks)):
            # If tracks are not detected for long time, remove them
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)
            # check to see if there are lots of large direction changes (> 120) in the last 10 steps
            # if there are get rid of the track
            if (len(self.tracks[i].trace) > 5):
                numBig, aveDirChange = self.get_ave_dir_change(self.tracks[i])
#                dir_mean = np.mean(self.tracks[i].trajectory_diffs[-10:])
#                dir_stdev = np.std(self.tracks[i].trajectory_diffs[-10:])
                dist_mean = np.mean(self.tracks[i].distances[-10:])
                dist_stdev = np.std(self.tracks[i].distances[-10:])
#                dist_var = np.var(self.tracks[i].distances[-10:])
                if (numBig > 6):
                    del_tracks.append(i)
                if (numBig > 3) & (aveDirChange > 100):
#                    print("Removing track due to large dir changes: " + str(aveDirChange) + ", " + str(aveDirChange))
                    del_tracks.append(i)
                elif (dist_stdev / dist_mean) > 3:
#                    print("Removing track due to large variation in distances: " + str(dist_mean) + ", " + str(dist_stdev) + ", " + str(dist_var))
                    del_tracks.append(i)
                elif (dist_mean > 750):
                    del_tracks.append(i)
                elif (max(self.tracks[i].distances[-10:]) > 750):
                    del_tracks.append(i)
#                dist_stdev = statistics.stdev(self.tracks[i].distances)
#                dist_var = statistics.variance(self.tracks[i].distances)
#                print("distance mean: " + str(dist_mean) + ", stdev: " + str(dist_stdev) + ", var: " + str(dist_var))
                
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in sorted(del_tracks, reverse=True):
                if id < len(self.tracks):
                    if self.writeImages:
                        self.removeLastImages(self.tracks[id], self.tracks[id].skipped_frames)
                    del self.tracks[id]
                else:
                    print("ERROR: id is greater than length of tracks")

        # Start new tracks
        if(len(un_assigned_detects) != 0):
            for i in range(len(un_assigned_detects)):
                # keep tracks from exploding
                if len(self.tracks) < 30:
                    track = Track(detections[un_assigned_detects[i]], self.trackIdCount)
                    x, y, ulX, ulY, dWidth, dHeight, area = detections[un_assigned_detects[i]]
                    cropped_frame = self.get100SizeCrop(frame, x, y)
                    if (len(cropped_frame) < 1):
                        # I don't think this should happen anymore
                        self.tracks[i].empty_image_count += 1
                    track.images.append(cropped_frame)
                    cropped_frame_actual = self.getCroppedImage(frame, ulX, ulY, dWidth, dHeight)
                    track.croppedImages.append(cropped_frame_actual)
                    self.trackIdCount += 1
                    self.tracks.append(track)


        # NOTE: the built in opencv image trackers don't work very well for anything but large birds
        # so this is all commented out, maybe we can figure out a way to include it later
#        for i in range(len(self.tracks)):
#            if (len(self.tracks[i].trace) > 10):
#                if not self.tracks[i].trackerstarted:
#                    x = self.tracks[i].trace[-1][0]
#                    y = self.tracks[i].trace[-1][1]
#                    a = self.tracks[i].areas[-1]
#                    if a > 0:
#                        r = self.get_rect(x, y, a)
#                        print("init tracker for track " + str(self.tracks[i].track_id))
#                        mt = cv2.MultiTracker_create()
#                        mt.add(cv2.TrackerMOSSE_create(), frame, r)
#                        cv2.rectangle(show_frame,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(0,0,255),2)
#                        cv2.imshow('thresh', show_frame)
#                        self.trackers.append(mt)
#                        self.tracks[i].trackerstarted = True

