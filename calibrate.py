#!/usr/bin/env python

'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images

usage:
    calibrate.py [--debug <output path>] [--square_size] [<image mask>] [--camera true/false]

default values:
    --debug:    ./output/
    --square_size: 1.0
    <image mask> defaults to ../data/left*.jpg
    --camera 1 (true)
'''

# Python 2/3 compatibility
from __future__ import print_function

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

import numpy as np
import cv2 as cv

# local modules
from common import splitfn

# built-in modules
import os

if __name__ == '__main__':
    import sys
    import getopt
    from glob import glob

    args, img_mask = getopt.getopt(sys.argv[1:], '', ['debug=', 'square_size=', 'threads=', 'camera=', 'raspberry='])
    args = dict(args)
    args.setdefault('--debug', './output/')
    args.setdefault('--square_size', 1.0)
    args.setdefault('--threads', 4)
    args.setdefault('--camera', 1)
    args.setdefault('--raspberry', 0)
    if not img_mask:
        img_mask = './chessboard/*.png'  # default
    else:
        img_mask = img_mask[0]

    img_names = glob(img_mask)
    debug_dir = args.get('--debug')
    if debug_dir and not os.path.isdir(debug_dir):
        os.mkdir(debug_dir)
    square_size = float(args.get('--square_size'))

    pattern_size = (11, 7)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    name = 0
    found = 0
    exitFlag = 0
    if args.get('--camera') == '1':
	print("Camera flag is 1")
        if args.get('--raspberry') == 0:
	   print("Raspberry flag is 0")
           cap = cv.VideoCapture(0)
	   if cap.isOpened():
		while(exitFlag <= 0):
          		ret,img = cap.read()
           		gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            
          		if ret:
              		   found, corners = cv.findChessboardCorners(gray, pattern_size)
              		   [h, w] = gray.shape[:2]

           		if found:
               		   term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
               		   cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
               		   name = name + 1
               		   outfile = os.path.join(debug_dir, str(name) + '_chess.png')
               		   cv.imwrite(outfile, img)
               		   cv.drawChessboardCorners(img, pattern_size, corners, found)
	  	           img_points.append(corners)
		           obj_points.append(pattern_points)

		        cv.imshow('image',img)
           		exitFlag = cv.waitKey(1)
	        cv.destroyAllWindows()
       		cap.release()
	else:
	   print("Raspberry flag is ", args.get('--raspberry'))
	   #initialize the camera and grab a reference to the raw camera capture
           cap = PiCamera()
	   cap.resolution = (640, 480)
	   cap.framerate = 32
           rawCapture = PiRGBArray(cap, size=(640, 480)) 
	   # allow the camera to warmup
	   time.sleep(0.1)
	   # grab an image from the camera
	   for frame in cap.capture_continuous(rawCapture, format="bgr", use_video_port=True):
		# grab the raw NumPy array representing the image, then initialize the timestamp
		# and occupied/unoccupied text
		img = frame.array
		gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
		found, corners = cv.findChessboardCorners(gray, pattern_size)
		[h, w] = gray.shape[:2]
		if found:
			term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
                  	cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
                	name = name + 1
                 	outfile = os.path.join(debug_dir, str(name) + '_chess.png')
                	cv.imwrite(outfile, img)
                 	cv.drawChessboardCorners(img, pattern_size, corners, found)
                 	img_points.append(corners)
                 	obj_points.append(pattern_points)

		# show the frame
		cv.imshow("Frame", img)
		key = cv.waitKey(1) & 0xFF
 
		# clear the stream in preparation for the next frame
		rawCapture.truncate(0)
 
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			cv.destroyAllWindows()	
			break

    else:
       for fn in img_names if debug_dir else []:
         print(fn)
         path, name, ext = splitfn(fn)
         img_found = os.path.join(debug_dir, name + '_chess.png')
         outfile = os.path.join(debug_dir, name + '_undistorted.png')

         img = cv.imread(fn)
         if img is None:
            print("skipped image ",name)
            continue
         else:
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            found, corners = cv.findChessboardCorners(gray,pattern_size)
            [h, w] = gray.shape[:2]
            if found:
                term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
                cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
                img_points.append(corners)
                obj_points.append(pattern_points)

    # calculate camera distortion
    if obj_points:
        print("Processing image points\n")
        rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)

        print("\nRMS:", rms)
        print("camera matrix:\n", camera_matrix)
        print("distortion coefficients: ", dist_coefs.ravel())
        
   # cap = cv.VideoCapture(0)
   # exitFlag = 0
   # while(exitFlag <= 0):
   #     ret,img = cap.read()
            
   #     if ret:
   #         dst = cv.undistort(img, camera_matrix, dist_coefs)
   #         cv.imshow('original',img)
   #         cv.imshow('undistorted',dst)
   #         exitFlag = cv.waitKey(1)
       
   # cv.destroyAllWindows()
   # cap.release()
