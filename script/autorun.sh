#!/bin/bash
unset PYTHONPATH

source ~/github/mlsl/_install/intel64/bin/mlslvars.sh thread
source /opt/intel/impi/2018.1.163/intel64/bin/mpivars.sh release_mt

source activate mxnet_v1.5

echo "PYTHONPATH=$PYTHONPATH"
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

python elastic.py --model resnet50_v1 --command "python resnet50_imagenet.py"
