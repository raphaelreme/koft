#!/bin/bash

mkdir -p dataset/dupre

# Dupre tracking dataset
wget https://github.com/raphaelreme/koft/releases/download/dupre_data/20160412_dupreannotation_stk0001.csv
wget https://github.com/raphaelreme/koft/releases/download/dupre_data/20160412_stk_0001.tif

mv 20160412_dupreannotation_stk0001.csv dataset/dupre
mv 20160412_stk_0001.tif dataset/dupre

# Dupre's video for SINETRA motion
wget https://github.com/raphaelreme/SINETRA/releases/download/dupre_data/dupre_20140829_1_contracting.tiff
mv dupre_20140829_1_contracting.tiff dataset/dupre
