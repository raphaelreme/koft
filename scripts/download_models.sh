#!/bin/bash

mkdir -p models/
cd models/

# Download StarDist for Hydra's neurons (TRASE-IN dataset)
wget https://github.com/raphaelreme/trase-in/releases/download/stardist_model/stardist.zip
unzip stardist.zip
/bin/rm stardist.zip

# Download TrackAstra trained on Sinetra
wget https://github.com/raphaelreme/koft/releases/download/trackastra_model/trackastra_sinetra.zip
unzip trackastra_sinetra.zip
/bin/rm trackastra_sinetra.zip
