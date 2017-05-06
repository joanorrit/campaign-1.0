export INST=$HOME/campaign-1.0/
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export DYLD_LIBRARY_PATH=$LD_LIBRARY_PATH
./configure --with-cuda-lib=/usr/local/cuda/lib64 --prefix=$INST/install --exec-prefix=$INST/install
