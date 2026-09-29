#!/bin/bash

mkdir -p dataset/dupre
mkdir -p dataset/trase-in

# Dupre tracking dataset
wget https://github.com/raphaelreme/koft/releases/download/dupre_data/20160412_dupreannotation_stk0001.csv
wget https://github.com/raphaelreme/koft/releases/download/dupre_data/20160412_stk_0001.tif

mv 20160412_dupreannotation_stk0001.csv dataset/dupre
mv 20160412_stk_0001.tif dataset/dupre

# Dupre's video for SINETRA motion
wget https://github.com/raphaelreme/SINETRA/releases/download/dupre_data/dupre_20140829_1_contracting.tiff
mv dupre_20140829_1_contracting.tiff dataset/dupre


# TRASE-IN first video
wget https://github.com/raphaelreme/koft/releases/download/hanson_data/tdt_contrxn-1.avi
mv tdt_contrxn-1.avi dataset/trase-in
