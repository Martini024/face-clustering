# Face-Clustering (done under Consulting & Analytics Club, IIT Guwahati)

Clustering set of images based on the faces recognized using the DBSCAN clustering algorithm.

Face recognition and face clustering are different. When performing face recognition we are applying *supervised learning* where we have both

- example images of faces we want to recognize along with
- the names that correspond to each face (i.e., the "class labels").

But in face clustering we need to perform *unsupervised learning* — we have only the faces themselves with no names/labels.
From there we need to identify and count the number of unique people in a dataset.

- extract and quantify the faces in a dataset
- another to cluster the faces, where each resulting cluster (ideally) represents a unique individual

The model is developed in the python language.

## Dependencies

- face_recognition
- imutils
- sklearn.cluster
- argparse
- pickle
- openCV (cv2)
- os

## Quickstart

### MacOS

#### Option 1: Run using Terminal

1. Restore dependencies using following commands

    ```
    <!-- Install Anaconda using Homebrew -->
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    brew install --cask anaconda
    ```

    ```
    <!-- Create face-clustering using Anaconda, and resolve dependencies -->
    conda create --name face-clustering
    conda install pip
    pip install cmake face_recognition imutils scikit-learn argparse opencv-python 
    ```

2. Check the following sections about `encode_images.py` & `cluster_faces.py` to run the clustering algorithm

#### Option 2: No-brainier Command Tool

To make `face-clustering.command` executable by double-clicking, run following command in terminal

```bash
chmod +x {path to face-clustering.command}
```

If `"face-clustering.command" cannot be opened because it is from an unidentified developer.` comes out, go to System Preference -> Security & Privacy -> General -> Open Anyway

\>_> ok lor, double click, job done !!!

### Windows

1. Follow the guide to install Anaconda on Windows [:link:](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html)
2. Follow the guide to install Git on Windows [:link:](https://git-scm.com/download/win)

3. Paste following command to `~/.bash_profile` and adjust the path to `conda.exe` according to your directories

    `eval "$('{Absolute path to anaconda3/Scripts/conda.exe}' 'shell.bash' 'hook')"`

4. Paste following command to `~/.bashrc` and adjust the path according to your directories

    `. /c/Anaconda3/etc/profile.d/conda.sh`

5. Run `face-clustering.sh` with Git Bash

## encode_images.py

encode_faces.py script will contains code used to extract a 128-d feature vector representation for each face.

**Arguments:**

- -i --dataset : The path to the input directory of faces and images.
- -e --encodings : The path to our output serialized pickle file containing the facial encodings.
- -d --detection_method : Face detection method to be used. Can be "hog" or "cnn" (Default: cnn)

**What it does**

- create a list of all imagePaths in our dataset using the dataset path provided in our command line argument.
- we compute the 128-d face encodings for each detected face in the rgb image
- For each of the detected faces + encodings, we build a dictionary that includes:
  - The path to the input image
  - The location of the face in the image (i.e., the bounding box)
  - The 128-d encoding itself
- Can be reused. write the data list to disk as a serialized encodings.pickle file

**Usage - To run**
$python encode_faces.py --dataset dataset --encodings encodings.pickle --detection_method "cnn"

## cluster_faces.py

we have quantified and encoded all faces in our dataset as 128-d vectors, the next step is to cluster them into groups.
*Our hope is that each unique individual person will have their own separate cluster*

For this task we need a clustering algorithm, many clustering algorithms such as k-means and Hierarchical
Agglomerative Clustering, require us to specify the number of clusters we seek ahead of time.
Therefore, we need to use a density-based or graph-based clustering algorithm
*Density-based spatial clustering of applications with noise (DBSCAN)*

**Arguments:**

- -i --encodings : The path to the encodings pickle file that we generated in our previous script.
- -d --jobs : DBSCAN is multithreaded and a parameter can be passed to the constructor containing the number of parallel jobs to run.
              A value of -1 will use all CPUs available (default).

**What it does**

- Loaded the facial encodings data from disk, Organized the data as a NumPy array, Extracted the 128-d encodings from the data , placing them in a list
- create a DBSCAN object and then fit the model on the encodings
- loop to populate all the images in the database, and check the cluster and create a directory for the cluster.
- We employ the build_montages function of imutils to generate a single image montage containing a 5×5 grid of faces

**To run**
$python cluster_faces.py --encodings encodings.pickle --jobs -1

## Application

This can be used to cluster out the gallery in mobile applications or any other application with large number of images which makes the operation inefficient for humans.

## Acknowledgement

- @ageitgey for the face_recognition library
  