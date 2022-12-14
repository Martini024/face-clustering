echo "\033[0;36m
  __           _       _            _           _            _             
 / _|         (_)     | |          | |         | |          (_)            
| |_ __ _  ___ _  __ _| |______ ___| |_   _ ___| |_ ___ _ __ _ _ __   __ _ 
|  _/ _\` |/ __| |/ _\` | |______/ __| | | | / __| __/ _ \ '__| | '_ \ / _\` |
| || (_| | (__| | (_| | |     | (__| | |_| \__ \ ||  __/ |  | | | | | (_| |
|_| \__,_|\___|_|\__,_|_|      \___|_|\__,_|___/\__\___|_|  |_|_| |_|\__, |
                                                                      __/ |
                                                                     |___/ \033[0m
"

echo "Note: image with many faces will prolong the recognition process, \n      remove them if you don't wanna torture yourself\n"

read -p "1. Where is the dataset folder (default in dataset): " dataset
if [[ -z "$dataset" ]]; then
   dataset="dataset"
fi
echo "   Target dataset folder: $dataset"

find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

read -p "2. Install dependencies: y/n " yn
case $yn in
    [Yy]* ) if find_in_conda_env ".*face-clustering.*" ; then
                conda activate face-clustering
            else 
                conda create --name face-clustering 
            fi
            conda install pip
            pip install cmake
            pip install dlib face_recognition imutils scikit-learn argparse 
            pip install opencv-python
            ;;
    [Nn]* ) if find_in_conda_env ".*face-clustering.*" ; then
                conda activate face-clustering
            else 
                conda create --name face-clustering 
            fi
            ;;
    * ) echo "Please answer yes or no.";;
esac

echo ""

cd "$(dirname "$0")"
echo "\033[0;32mStart extracting faces, please be patient...\033[0m\n"

python encode_faces.py --dataset "$dataset" --encodings encodings.pickle

echo ""

echo "\033[0;32mClustering...\033[0m\n"
python cluster_faces.py --encodings encodings.pickle --jobs -1