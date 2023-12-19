#!/bin/bash

packageName=scidx-1
minor=0
revision=0

tarName=${packageName}.${minor}.${revision}
make dist

tar -xzvf ${tarName}.tar.gz
cp LICENSE ${tarName}/
cp autogen.sh ${tarName}/
cp distribution.sh ${tarName}/
cp examples/minmax_values.txt ${tarName}/examples


tar -czvf ${tarName}.tar.gz ${tarName}


