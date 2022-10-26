# USAGE
# python encode_faces.py --dataset dataset --encodings encodings.pickle

# argument parser
import argparse
# operating system
import os
# pickle to save the encodings
import pickle
import time
from datetime import datetime

# openCV
import cv2
# face_recognition library by @ageitgey
import face_recognition
# import the necessary packages
from imutils import paths

from constants import ENCODINGS_PATH

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
                help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized database of facial encodings")
ap.add_argument("-d", "--detection_method", type=str, default="cnn",
                help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset, then initialize
# out data list (which we'll soon populate)
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []

start_time = time.time()

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # load the input image and convert it from RGB (OpenCV ordering)
    # to dlib ordering (RGB)
    print("[INFO] processing image {}/{} - {} at {}".format(i +
          1, len(imagePaths), imagePath, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

    # loading image to BGR
    image = cv2.imread(imagePath)

    # converting image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    boxes = face_recognition.face_locations(image,
                                            model=args["detection_method"])

    print("       Detected {} faces".format(len(boxes)))
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(image, boxes)

    # build a dictionary of the image path, bounding box location,
    # and facial encodings for the current image
    d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
         for (box, enc) in zip(boxes, encodings)]
    data.extend(d)

end_time = time.time()

print("---Encoding faces took {} minutes ---".format((end_time - start_time) / 60))

# dump the facial encodings data to disk
print("[INFO] serializing encodings...")
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
print("Encodings of images saved in {}".format(ENCODINGS_PATH))
