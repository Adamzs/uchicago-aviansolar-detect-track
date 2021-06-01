import math
import cv2
import copy
import os
import sys
import time
import argparse
import pandas as pd
from obj_detector import ObjDetector
from tracker import Tracker

def calc_trajectory(track):
    if len(track.trace) > 5:
        x1 = track.trace[-5][0]
        y1 = -track.trace[-5][1]
        x2 = track.trace[-1][0]
        y2 = -track.trace[-1][1]
        dx = x2 - x1
        dy = y2 - y1
        return calc_angle(dx, dy)
    return 0

def calc_angle(dx, dy):
    rads = math.atan2(dy, dx)
    degs = math.degrees(rads)
    adj = 360 - (degs - 90)
    if adj >= 360:
        return adj - 360
    return adj        


def find_objects(file, filename, outpath, showImages, writeImages, verbose):
    storeImages = True

    print("Calling opencv VideoCapture with file: " + file)
    try:
        cap = cv2.VideoCapture(file)
        
        detector = ObjDetector(showImages, 200)
        tracker = Tracker(900, 10, 150, 100, storeImages, writeImages, filename, outpath)
        
        if showImages == True:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 1920, 1080)
    
        count = 1
        # set this value if you want to start processing a video somewhere in the middle
        start_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            
            if ret == False:
                break
    
            if count > start_frame:
                orig_frame = copy.copy(frame)
        
                centers = detector.findObjects(frame, tracker)
                tracker.updateTracks(centers, orig_frame, count)
                
                if verbose:
                    print('processed frame: ' + str(count) + ", num tracks = " + str(len(tracker.tracks)))
            count += 1

    except cv2.error as error:
        print("Error with video file: " + file + " - " + error)

    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_file", help="The video file to process")
    parser.add_argument("output_dir", help="The path to the output folder to use if writing out moving object images")
    parser.add_argument("-show_video", action="store_true", help="Should the video be displayed while processing?")
    parser.add_argument("-write_images", action="store_true", help="Should images of the moving objects be written to disk?")
    parser.add_argument("-v", action="store_true", help="Verbose - should more detail be written about progress?")

    args = parser.parse_args()

    full_path_video_file = args.video_file
    
    if not os.path.exists(full_path_video_file):
        print("File does not exist: " + full_path_video_file)
        exit(1)

    output_dir_base = args.output_dir

    fileName = os.path.basename(full_path_video_file)
    fileNameBase = os.path.splitext(fileName)[0]
    fileExt = os.path.splitext(fileName)[1]
    outputDir = os.path.join(output_dir_base, fileNameBase)

    if not os.path.exists(outputDir):
        try:
            os.makedirs(outputDir)
        except FileExistsError:
            pass
            
    out_filename = os.path.join(outputDir, "processing_log.txt")
    with open(out_filename, 'w') as out_file:
        start = time.time()
        out_file.write("Attempting to process file: " + fileName + "\n")
        out_file.flush()
        if fileExt in [".mp4", ".MP4", ".MOD", "mod", ".MTS", ".mts", ".mkv"]:
            out_file.write("File valid, starting processing\n")
            out_file.flush()
            find_objects(full_path_video_file, fileNameBase, outputDir, args.show_video, args.write_images, args.v)
            delta = time.time() - start
            out_file.write("Took: " + str(delta) + " seconds to process video\n")
            out_file.flush()
        else:
            out_file.write("Not valid file type, not processing\n")
            out_file.flush()
