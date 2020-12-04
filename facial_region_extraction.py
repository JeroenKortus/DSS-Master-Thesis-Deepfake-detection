'''
Author: Jeroen Kortus

This script is used generate the different configurations of facial regions.
'''

import cv2
import csv  
import numpy as np
from skimage.draw import polygon, ellipse
import math
from pathlib import Path
import os

def eyes_region(landmarks_x, landmarks_y, img_width, img_height):
	brow_offset_x = ((landmarks_x[36] - landmarks_x[17]) + (landmarks_x[45] - landmarks_x[26]))/6
	brow_offset_y = ((landmarks_y[36] - landmarks_y[17]) + (landmarks_y[45] - landmarks_y[26]))/6

	outline_eyesbrows = [*range(26,16,-1)]
	eyes_poly_x = np.array(landmarks_x)[outline_eyesbrows] - brow_offset_x
	eyes_poly_y = np.array(landmarks_y)[outline_eyesbrows] - brow_offset_y

	eyes_poly_x = np.append(eyes_poly_x, np.sum(np.array(landmarks_x)[[0,1]])/2)
	eyes_poly_y = np.append(eyes_poly_y, np.sum(np.array(landmarks_y)[[0,1]])/2)

	eyes_poly_x = np.append(eyes_poly_x, np.sum(np.array(landmarks_x)[[15,16]])/2)
	eyes_poly_y = np.append(eyes_poly_y, np.sum(np.array(landmarks_y)[[15,16]])/2)

	eyes_Y, eyes_X = polygon(eyes_poly_x, eyes_poly_y)

	rm_index = []
	for i in range(len(eyes_Y)):
		if eyes_Y[i] >= img_height or eyes_X[i] >= img_width:
			rm_index.append(i)
	eyes_X = np.delete(eyes_X, rm_index)
	eyes_Y = np.delete(eyes_Y, rm_index)

	return eyes_X, eyes_Y

def nose_region(landmarks_x, landmarks_y, img_width, img_height):
	eye_dist_x 			= (landmarks_x[42] - landmarks_x[39])
	eye_dist_y 			= (landmarks_y[42] - landmarks_y[39])
	side_node_right_x 	= (landmarks_x[35] - landmarks_x[33])
	side_node_right_y 	= (landmarks_y[35] - landmarks_y[33])
	side_node_left_x 	= (landmarks_x[31] - landmarks_x[33])
	side_node_left_y 	= (landmarks_y[31] - landmarks_y[33])
	nose_bottom_dist_x 	= (landmarks_x[33] - landmarks_x[30])
	nose_bottom_dist_y 	= (landmarks_y[33] - landmarks_y[30])

	nose_poly_x = [landmarks_x[21], landmarks_x[22]]
	nose_poly_y = [landmarks_y[21], landmarks_y[22]]

	nose_poly_x.append(landmarks_x[27] + 0.25*eye_dist_x)
	nose_poly_y.append(landmarks_y[27] + 0.25*eye_dist_y)
	nose_poly_x.append(landmarks_x[35] + side_node_right_x)
	nose_poly_y.append(landmarks_y[35] + side_node_right_y)

	for x in np.array(landmarks_x)[[*range(35,31,-1)]]+0.5*nose_bottom_dist_x:
		nose_poly_x.append(x)
	for y in np.array(landmarks_y)[[*range(35,31,-1)]]+0.5*nose_bottom_dist_y:
		nose_poly_y.append(y)

	nose_poly_x.append(landmarks_x[31] + side_node_left_x)
	nose_poly_y.append(landmarks_y[31] + side_node_left_y)
	nose_poly_x.append(landmarks_x[27] - 0.25*eye_dist_x)
	nose_poly_y.append(landmarks_y[27] - 0.25*eye_dist_y)

	nose_Y, nose_X = polygon(nose_poly_x, nose_poly_y)

	rm_index = []
	for i in range(len(nose_Y)):
		if nose_Y[i] >= img_height or nose_X[i] >= img_width:
			rm_index.append(i)
	nose_X = np.delete(nose_X, rm_index)
	nose_Y = np.delete(nose_Y, rm_index)

	return nose_X, nose_Y

def mouth_region(landmarks_x, landmarks_y, img_width, img_height):
	mouth_width  = 1.3* ((landmarks_y[54] - landmarks_y[48])**2 + (landmarks_x[54] - landmarks_x[48])**2)**0.5
	mouth_height = 1.7* ((landmarks_y[51] - landmarks_y[57])**2 + (landmarks_x[51] - landmarks_x[57])**2)**0.5
	center_x = int(np.sum(np.array(landmarks_x)[[48, 51, 54, 57]])/4)
	center_y = int(np.sum(np.array(landmarks_y)[[48, 51, 54, 57]])/4)
	angle = math.atan((landmarks_y[54] - landmarks_y[48])/(landmarks_x[54] - landmarks_x[48]))

	mouth_Y, mouth_X = ellipse(center_x, center_y, mouth_width/2, mouth_height/2, rotation=angle)

	rm_index = []
	for i in range(len(mouth_Y)):
		if mouth_Y[i] >= img_height or mouth_X[i] >= img_width:
			rm_index.append(i)
	mouth_X = np.delete(mouth_X, rm_index)
	mouth_Y = np.delete(mouth_Y, rm_index)

	return mouth_X, mouth_Y

def full_face_region(landmarks_x, landmarks_y, img_width, img_height):
	left_offset_x = landmarks_x[0] - landmarks_x[1]
	left_offset_y = landmarks_y[0] - landmarks_y[1]
	right_offset_x = landmarks_x[16] - landmarks_x[15]
	right_offset_y = landmarks_y[16] - landmarks_y[15]

	face_outline = [*range(0,17)]
	face_poly_x = np.array(landmarks_x)[face_outline]
	face_poly_y = np.array(landmarks_y)[face_outline]
	face_poly_x = np.append(face_poly_x, landmarks_x[16] + right_offset_x*3)
	face_poly_y = np.append(face_poly_y, landmarks_y[16] + right_offset_y*3)
	face_poly_x = np.append(face_poly_x, landmarks_x[0] + left_offset_x*3)
	face_poly_y = np.append(face_poly_y, landmarks_y[0] + left_offset_y*3)

	face_Y, face_X = polygon(face_poly_x, face_poly_y)

	rm_index = []
	for i in range(len(face_Y)):
		if face_Y[i] >= img_height or face_X[i] >= img_width:
			rm_index.append(i)
	face_X = np.delete(face_X, rm_index)
	face_Y = np.delete(face_Y, rm_index)

	return face_X, face_Y



image_path = "/folder/with/images/"
output_path = "/output/folder/"

# Get all csv files paths from the folder with the square bitmaps and the csv files.
images = Path(image_path).glob("*.csv")

for idx, image in enumerate(images):
	filename = os.path.basename(os.path.normpath(str(image)))[0:-4]
	img = cv2.imread(image_path + filename + ".bmp")
	landmarks_x = []
	landmarks_y = []
	# Load the csv file and read the landmarks.
	with open(image_path + filename + ".csv") as csvfile:  
	    data = csv.DictReader(csvfile)
	    for row in data:
	        for i in range(68):
	        	x = " x_"+str(i)
	        	y = " y_"+str(i)
	        	landmarks_x.append(float(row[x]))
	        	landmarks_y.append(float(row[y]))

	img_width, img_height, _ = img.shape

	# Functions can be toggled commented out in case they are not used. This will dave time in case of large datasets.
	eyes_X, eyes_Y = eyes_region(landmarks_x, landmarks_y, img_width, img_height)
	nose_X, nose_Y = nose_region(landmarks_x, landmarks_y, img_width, img_height)
	mouth_X, mouth_Y = mouth_region(landmarks_x, landmarks_y, img_width, img_height)
	face_X, face_Y = full_face_region(landmarks_x, landmarks_y, img_width, img_height)

	# The basic regions can be used to create different configurations.
	# The code bellow created the rest-of-the-face region.
	# First the full-face mask is used to set all pixels included to the pixel from the image
	# Then the eyes, nose, and mouth regions are set to 0 to remove them from the mask.
	cropped_face = np.zeros(img.shape, dtype=np.uint8)
	cropped_face[face_X, face_Y] = img[face_X, face_Y]
	cropped_face[eyes_X, eyes_Y] = 0
	cropped_face[nose_X, nose_Y] = 0
	cropped_face[mouth_X, mouth_Y] = 0

	cv2.imwrite(output_path + filename + ".bmp", cropped_face)

