import cv2
import os
import math
import numpy as np
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
        initArea = initLoc[2]
        self.prediction = [initX, initY]
        self.vel_prediction = [0, 0]
        self.skipped_frames = 0 
        self.trace = [] 
        self.trace.append([initX, initY])
        self.areas = []
        self.areas.append(initArea)
        self.init = True
        self.images = []
        self.obj_type_predictions = []
        self.firm_track_count = 0
        self.write_initial_frames = True
        self.prediction_history = []
        self.prediction_history.append([initX, initY])
        self.trajectories = []
        self.trajectories.append(0)
        self.pred_trajectories = []
        self.pred_trajectories.append(0)
        self.distances = []
        self.distances.append(0)


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

    # this method is a hack - I really want to initialize the KalmanFilter with the
    # location of the first point, but I can't figure out how to do that, this seems
    # to accomplish it
    def initKF(self, i):
        x, y = self.tracks[i].prediction;
        for j in range(10):
            self.tracks[i].KF.correct(np.array([np.float32(x), np.float32(y)], np.float32))
            self.tracks[i].KF.predict()
        self.tracks[i].init = False

    def writeImage(self, cropped_frame, x, y, frame_num, track_id, speed, area, angle_to_pred, angle_to_det, distance):
        if len(cropped_frame) > 1 and int(area) > 0:
            baseDir = os.path.join(self.outDir, self.baseFilename)
            if not os.path.exists(baseDir):
                os.makedirs(baseDir)
            trackDir = os.path.join(baseDir, str(track_id))
            if not os.path.exists(trackDir):
                os.makedirs(trackDir)
            file = os.path.join(trackDir, self.baseFilename + '_' + str(frame_num) + '_' + str(round(int(x))) + '_' + str(round(int(y))) + '_' + str("%.5f" % speed) + '_' + str(int(area)) + '_' + str(track_id) + '_' + str(angle_to_pred) + '_' + str(angle_to_det) + '_' + str(distance) + '.png')
            #print('writing file: ' + file)
            cv2.imwrite(file, cropped_frame)
        

    def get100SizeCrop(self, frame, cenX, cenY):
        uxpt = int(cenX - 100)
        lxpt = int(uxpt + 200)
        uypt = int(cenY - 100)
        lypt = int(uypt + 200)

        height = frame.shape[0]
        width = frame.shape[1]

        if uxpt < 0 or lxpt < 0 or uypt < 0 or lypt < 0:
            return []
        
        if uxpt > width or lxpt > width:
            return []
        
        if uypt > height or lypt > height:
            return []
        
        cropped_frame = frame[uypt:lypt, uxpt:lxpt]
        return cropped_frame

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

    def calc_ave_dist(self, distances):
        total = 0
        count = 0
        for i in range(len(distances)):
            total += distances[-i]
            count += 1
            # just get the average of the previous 5 points
            if count > 5:
                break

        if count > 0:
            return total / count
        else:
            return 0

    def updateTracks(self, detections, frame, frame_num):
        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            predX, predY = self.tracks[i].prediction
            for j in range(len(detections)):
                try:
                    detX, detY, area = detections[j]
                    xDiff = detX - predX
                    yDiff = detY - predY
                    distance = np.sqrt(xDiff * xDiff + yDiff * yDiff)
                    # penalize newly established tracks in case an older one could match this detection
                    # this is small but it helps a bit. I need to figure out a better way to handle this
                    if len(self.tracks[i].trace) < 3:
                        cost[i][j] = distance + 5
                    else:                        
                        cost[i][j] = distance
                except:
                    pass

        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                dt = self.dist_thresh
                # dist_thresh should vary based on track confidence (length of track)
                trackLen = len(self.tracks[i].trace)
                # TODO: this should also be based on the speed of the object and num skipped
                numSkipped = self.tracks[i].skipped_frames
                relVel = self.tracks[i].vel_prediction
                if trackLen > 16:
                    dt = dt / 12
                elif trackLen > 8:
                    dt = dt / 8
                elif trackLen > 4:
                    dt = dt / 2
                if (cost[i][assignment[i]] > dt):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                else:
                    if len(self.tracks[i].trace) > 1:
                        # prior trajectory from track object
                        priorTrejectory = self.tracks[i].trajectories[-1]
                        
                        # trajectory to predicted next point
                        predTrajectory = self.tracks[i].pred_trajectories[-1]
                        
                        # trajectory to new detected point
                        lastX, lastY = self.tracks[i].trace[-1]
                        predX, predY = self.tracks[i].prediction
                        x, y, area = detections[assignment[i]]
                        trajectoryToDetected = self.calc_angle((x - lastX), (y - lastY))

                        # distance from previous point to new detected point (not between new point and predicted point that we computed earlier)
                        xDiff = x - lastX
                        yDiff = y - lastY
                        distance = np.sqrt(xDiff * xDiff + yDiff * yDiff)

                        priorAveDistance = self.calc_ave_dist(self.tracks[i].distances) #self.tracks[i].distances[-1]
                        distanceDiff = np.abs(priorAveDistance - distance)
                        distDifRatio = 0
                        if (priorAveDistance > 0) & (distanceDiff > 0):
                            distDifRatio = distanceDiff / priorAveDistance
                        
                        trajDiff = np.abs(priorTrejectory - trajectoryToDetected)
                        if trajDiff > 180:
                            trajDiff = np.abs(trajDiff - 360)
                        if self.DEBUG:
                            print(str(frame_num) + "-" + str(self.tracks[i].track_id) + ", " +
                                  str(lastX) + ", " + str(lastY) + ", " + 
                                  str(predX) + ", " + str(predY) + ", " + 
                                  str(x) + ", " + str(y) + ", " + 
                                  str(priorTrejectory) + ", " + str(predTrajectory) + ", " + str(trajectoryToDetected) + ", " + str(trajDiff) + ", " + str(distanceDiff) + ", " + str(distDifRatio) + ", " + str(self.tracks[i].skipped_frames))
                        if len(self.tracks[i].trace) > 3:
                            if trajDiff > 100:
                                assignment[i] = -1
                                un_assigned_tracks.append(i)
                        if len(self.tracks[i].trace) > 4:
                            # this is a pretty big ratio, but it gets rid of a lot suprisingly
                            if distDifRatio > 1.5:
                                assignment[i] = -1
                                un_assigned_tracks.append(i)
                pass
            else:
                self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in sorted(del_tracks, reverse=True):
                if id < len(self.tracks):
                    del self.tracks[id]
                    del assignment[id]
                else:
                    print("ERROR: id is greater than length of tracks")

        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
                if i not in assignment:
                    un_assigned_detects.append(i)

        # Start new tracks
        if(len(un_assigned_detects) != 0):
            for i in range(len(un_assigned_detects)):
                # keep tracks from exploding
                if len(self.tracks) < 30:
                    track = Track(detections[un_assigned_detects[i]], self.trackIdCount)
                    self.trackIdCount += 1
                    self.tracks.append(track)

        # if we blindly start new tracks for all un-assigned detections then the new track could steal future detections
        # that should belong to an established track, need to figure this out...

        for i in range(len(assignment)):
            prevX, prevY = self.tracks[i].trace[-1]
            x = 0
            y = 0
            area = 0
            # this means we found a detection to add to a track
            if(assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                x, y, area = detections[assignment[i]]
                self.tracks[i].trace.append([x, y])
                self.tracks[i].areas.append(area)
                self.tracks[i].KF.correct(np.array([np.float32(x), np.float32(y)], np.float32))
                self.tracks[i].firm_track_count += 1
            else:
                # carry forward the prediction for a little while
                x, y = self.tracks[i].prediction
                self.tracks[i].trace.append([x, y])
                self.tracks[i].areas.append(0)
                self.tracks[i].KF.correct(np.array([np.float32(x), np.float32(y)], np.float32))

            # compute here the prior trajectory
            priorTraj = self.calc_angle((x - prevX), (y - prevY))
            self.tracks[i].trajectories.append(priorTraj)

            # compute the distance from the prior point to this point
            xDiff = x - prevX
            yDiff = y - prevY
            distance = np.sqrt(xDiff * xDiff + yDiff * yDiff)
            self.tracks[i].distances.append(distance)

            cropped_frame = self.get100SizeCrop(frame, x, y)
            self.tracks[i].images.append(cropped_frame)

            if self.tracks[i].init is True:
                self.initKF(i)

            prediction = self.tracks[i].KF.predict()

            if len(self.tracks[i].trace) > 3:
                predX = prediction[0,0]
                predY = prediction[1,0]
                velX = prediction[2,0]
                velY = prediction[3,0]
                if predX != 0 or predY != 0:
                    self.tracks[i].prediction = [predX, predY]
                    self.tracks[i].vel_prediction = [velX, velY]
                else:
                    self.tracks[i].prediction = self.tracks[i].trace[-1]
                    self.tracks[i].vel_prediction = [0, 0]
            else:
                self.tracks[i].prediction = self.tracks[i].trace[-1]
                self.tracks[i].vel_prediction = [0, 0]

            self.tracks[i].prediction_history.append(self.tracks[i].prediction)

            # compute here the predicted trajectory
            predX, predY = self.tracks[i].prediction
            predTraj = self.calc_angle((predX - x), (predY - y))
            self.tracks[i].pred_trajectories.append(predTraj)

            if self.writeImages:
                if self.tracks[i].firm_track_count > 8 and len(cropped_frame) > 1:
                    speed = self.tracks[i].vel_prediction
                    vel_mag = np.sqrt(np.power(speed[0], 2) + np.power(speed[1], 2))
                    if self.tracks[i].write_initial_frames == True:
                        # then write out all previous images, but do only once
                        self.tracks[i].write_initial_frames = False
                        prev_frame_num = frame_num - len(self.tracks[i].images)
                        for k in range(len(self.tracks[i].images)):
                            px, py = self.tracks[i].trace[k]
                            area = self.tracks[i].areas[k]
                            # todo, figure out how to get frame_num of previous better
                            self.writeImage(self.tracks[i].images[k], px, py, prev_frame_num, self.tracks[i].track_id, vel_mag, area, self.tracks[i].trajectories[k], self.tracks[i].pred_trajectories[k], self.tracks[i].distances[k])
                            prev_frame_num += 1
                    self.writeImage(cropped_frame, x, y, frame_num, self.tracks[i].track_id, vel_mag, area, self.tracks[i].trajectories[-1], self.tracks[i].pred_trajectories[-1], self.tracks[i].distances[-1])

            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) - self.max_trace_length):
                    del self.tracks[i].trace[j]
                    del self.tracks[i].images[j]
                    del self.tracks[i].prediction_history[j]
                    del self.tracks[i].pred_trajectories[j]
                    del self.tracks[i].trajectories[j]
                    del self.tracks[i].distances[j]
