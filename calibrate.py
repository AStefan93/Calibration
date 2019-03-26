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

    args, img_mask = getopt.getopt(sys.argv[1:], '', ['debug=', 'square_size=', 'threads=', 'camera='])
    args = dict(args)
    args.setdefault('--debug', './output/')
    args.setdefault('--square_size', 1.0)
    args.setdefault('--threads', 4)
    args.setdefault('--camera', 1)
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
    exitFlag = 0
    if args.get('--camera') == 1:
        cap = cv.VideoCapture(0)
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

    print("Exited loop with ",exitFlag)
    # calculate camera distortion
    if obj_points:
        print("Processing image points\n")
        rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)

        print("\nRMS:", rms)
        print("camera matrix:\n", camera_matrix)
        print("distortion coefficients: ", dist_coefs.ravel())
        
    cap = cv.VideoCapture(0)
    exitFlag = 0
    while(exitFlag <= 0):
        ret,img = cap.read()
            
        if ret:
            dst = cv.undistort(img, camera_matrix, dist_coefs)
            cv.imshow('original',img)
            cv.imshow('undistorted',dst)
            exitFlag = cv.waitKey(1)
       
    cv.destroyAllWindows()
    cap.release()
