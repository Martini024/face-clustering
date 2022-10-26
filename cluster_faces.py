# USAGE
# python cluster_faces.py --encodings encodings.pickle

import argparse
import os
import pickle
import shutil
import time
from datetime import datetime

import cv2
import numpy as np
from imutils import build_montages
# import the necessary packages
# DBSCAN model for clustering similar encodings
from sklearn.cluster import DBSCAN

from constants import CLUSTERING_RESULT_PATH, ENCODINGS_PATH, FACE_DATA_PATH

# add constants file in the code (clustering_result)


def promptForClearOutputFolder():
    prompt = f'Following process will clear the contents in result folder, please ensure the folder is unused before proceeding, enter \"y\" to continue (y/n): '
    answer = input(prompt).strip().lower()

    if answer not in ['y', 'n']:
        print(f'{answer} is an invalid choice, please select y/n')
        return promptForClearOutputFolder()

    if answer == 'y':
        return True

    return False


def move_image(image, id, labelID):

    path = CLUSTERING_RESULT_PATH+'/label'+str(labelID)
    # os.path.exists() method in Python is used to check whether the specified path exists or not.
    # os.mkdir() method in Python is used to create a directory named path with the specified numeric mode.
    if os.path.exists(path) == False:
        os.mkdir(path)

    filename = str(id) + '.jpg'
    # Using cv2.imwrite() method
    # Saving the image

    cv2.imwrite(os.path.join(path, filename), image)

    return


promptForClearOutputFolder()
if os.path.exists(CLUSTERING_RESULT_PATH):
    shutil.rmtree(CLUSTERING_RESULT_PATH)
os.makedirs(CLUSTERING_RESULT_PATH)
os.makedirs(os.path.join(CLUSTERING_RESULT_PATH, "summary"))

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-j", "--jobs", type=int, default=-1,
                help="# of parallel jobs to run (-1 will use all CPUs)")
args = vars(ap.parse_args())

# load the serialized face encodings + bounding box locations from
# disk/encodings pickle file, then extract the set of encodings to so we can cluster on them

print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
data = np.array(data)
encodings = [d["encoding"] for d in data]

# cluster the embeddings
print("[INFO] clustering...")

start_time = time.time()

# creating DBSCAN object for clustering the encodings with the metric "euclidean"
clt = DBSCAN(metric="euclidean", n_jobs=args["jobs"])
clt.fit(encodings)

# determine the total number of unique faces found in the dataset
# clt.labels_ contains the label ID for all faces in our dataset (i.e., which cluster each face belongs to).
# To find the unique faces/unique label IDs, used NumPy’s unique function.
# The result is a list of unique labelIDs
labelIDs = np.unique(clt.labels_)

# we count the numUniqueFaces . There could potentially be a value of -1 in labelIDs — this value corresponds
# to the “outlier” class where a 128-d embedding was too far away from any other clusters to be added to it.
# “outliers” could either be worth examining or simply discarding based on the application of face clustering.
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))

# loop over the unique face integers
for labelID in labelIDs:
    # find all indexes into the `data` array that belong to the
    # current label ID, then randomly sample a maximum of 25 indexes
    # from the set
    print("[INFO] faces for face ID: {}".format(labelID))
    idxs = np.where(clt.labels_ == labelID)[0]
    idxs = np.random.choice(idxs, size=min(25, len(idxs)),
                            replace=False)

    # initialize the list of faces to include in the montage
    faces = []

    # loop over the sampled indexes
    for i in idxs:
        # load the input image and extract the face ROI
        image = cv2.imread(data[i]["imagePath"])
        (top, right, bottom, left) = data[i]["loc"]
        face = image[top:bottom, left:right]

        # puting the image in the clustered folder
        move_image(image, i, labelID)

        # force resize the face ROI to 96mx96 and then add it to the
        # faces montage list
        face = cv2.resize(face, (96, 96))
        faces.append(face)

    # create a montage using 96x96 "tiles" with 5 rows and 5 columns
    montage = build_montages(faces, (96, 96), (5, 5))[0]

    # show the output montage
    title = "Face ID #{}".format(labelID)
    title = "Unknown Faces" if labelID == -1 else title
    """
	cv2.imshow(title, montage)
	cv2.waitKey(0)
	"""
    montageFileName = os.path.join(
        CLUSTERING_RESULT_PATH, 'summary', title + '.jpg')
    cv2.imwrite(montageFileName, montage)

end_time = time.time()

print("---Clustering faces took {} minutes ---".format((end_time - start_time) / 60))
