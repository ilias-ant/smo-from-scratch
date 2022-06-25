sudo apt-get update
sudo apt-get install wget
sudo apt-get install bzip2

cd data/
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2
bzip2 -d gisette_scale.bz2
