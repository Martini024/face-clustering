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
if [[ -z "$var" ]]; then
   dataset="dataset"
fi
echo "   Target dataset folder: $dataset"

read -p "2. Install Anaconda (using Homebrew): y/n " yn
case $yn in
    [Yy]* ) brew install --cask anaconda; break;;
    [Nn]* ) break;;
    * ) echo "Please answer yes or no.";;
esac

read -p "3. Install dependencies: y/n " yn
case $yn in
    [Yy]* ) conda create --name facial-clustering; conda install pip; pip install cmake face_recognition imutils scikit-learn argparse opencv-python; break;;
    [Nn]* ) break;;
    * ) echo "Please answer yes or no.";;
esac

echo ""

cd "$(dirname "$0")"
echo "\033[0;32mStart extracting faces, please be patient...\033[0m\n"

python encode_faces.py --dataset $dataset --encodings encodings.pickle

echo ""

echo "\033[0;32mClustering...\033[0m\n"
python cluster_faces.py --encodings encodings.pickle --jobs -1