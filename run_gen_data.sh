#!/bin/bash

ntrain=32768
nval=4096
ntest=4096
ng=144
datapath='/path/to/data/poisson'

e1=1  # poissons diffusion eigenvalue range
e2=5

adr1=0.2 # advection to diffusion ratio range
adr2=1
# for AD ratio we saved a set of velocity scales that correspond to AD ration in utils/*.npy. See python script for details

o1=1 # helmholtz wave number range
o2=10

# create poissons examples
python utils/gen_data_poisson.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
                    --ng=$ng --sparse --n 128 --datapath $datapath --e1 $e1 --e2 $e2
# create AD examples
#python utils/gen_data_advdiff.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
#                    --ng=$ng --sparse --n 128 --datapath $datapath --adr1 $adr1 --adr2 $adr2
# create Helm examples
#python utils/gen_data_helmholtz.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
#                    --ng=$ng --sparse --n 128 --datapath $datapath --o1 $o1 --o2 $o2

