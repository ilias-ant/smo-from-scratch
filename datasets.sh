cd data/
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2
bzip2 -d gisette_scale.bz2
ln -s gisette_scale gisette_train.txt

wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.t.bz2
bzip2 -d gisette_scale.t.bz2
ln -s gisette_scale.t gisette_test.txt
