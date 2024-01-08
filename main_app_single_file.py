import cv2
import copy
import os
import time
import argparse
from obj_detector import ObjDetector
from tracker import Tracker

def find_objects(file, filename, outpath, showImages, writeImages, writeVideoFile, verbose):
    storeImages = True
    
    print("Calling opencv VideoCapture with file: " + file)
    try:
        cap = cv2.VideoCapture(file)
        
        width  = cap.get(3)   # float `width`
        height = cap.get(4)  # float `height`
    
        print("width=" + str(width) + ", height=" + str(height))
        #
        if writeVideoFile:
            writer = cv2.VideoWriter(os.path.join(outpath,"output.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (int(width),int(height)))
        
        detector = ObjDetector(showImages, writeVideoFile, 200, verbose)
        tracker = Tracker(700, 5, 100, 100, storeImages, writeImages, filename, outpath)
        
        if showImages == True:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 1920, 1080)
            cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('thresh', 1920, 1080)
            cv2.namedWindow('bgsub', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('bgsub', 1920, 1080)
            cv2.namedWindow('contours', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('contours', 1920, 1080)

    
        count = 4022
        # set this value if you want to start processing a video somewhere in the middle
        start_frame_number = 4022
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
        while cap.isOpened():
            t1 = time.perf_counter()
            ret, frame = cap.read()
            t2 = time.perf_counter()
            if ret == False:
                break
    
            # print("frame: " + str(count))
            if count in [12, 1031, 1032, 1040, 2340, 2414, 4022, 5036, 5429, 6265]:
                print('frame: ' + str(count))

            if showImages | writeVideoFile:
                orig_frame = copy.copy(frame)
            else:
                orig_frame = frame
    
            centers = detector.findObjects(frame, tracker)
            tracker.updateTracks(centers, orig_frame, count)
        
            if verbose:
                print("Time with frame read: " + str(time.perf_counter() - t1))
                print("Time without read: " + str(time.perf_counter() - t2))
                
            if writeVideoFile:
                writer.write(frame)
            
            if verbose:
                print('processed frame: ' + str(count) + ", num tracks = " + str(len(tracker.tracks)))
                
            count += 1

    except cv2.error as error:
        print("Error with video file: " + file + " - " + str(error))

    cap.release()
    if writeVideoFile:
        writer.release()
    if showImages:
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_file", help="The video file to process")
    parser.add_argument("output_dir", help="The path to the output folder to use if writing out moving object images")
    parser.add_argument("-show_video", action="store_true", help="Should the video be displayed while processing?")
    parser.add_argument("-write_video", action="store_true", help="Should video output with detection boxes be written to disk?")
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
            find_objects(full_path_video_file, fileNameBase, outputDir, args.show_video, args.write_images, args.write_video, args.v)
            delta = time.time() - start
            out_file.write("Took: " + str(delta) + " seconds to process video\n")
            out_file.flush()
        else:
            out_file.write("Not valid file type, not processing\n")
            out_file.flush()
