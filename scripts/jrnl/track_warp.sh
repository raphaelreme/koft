# Warp Then Track vs KOFT

set -e

SCRIPTPATH=$(dirname "$0")

# Different detections method ?
# bash $SCRIPTPATH/_track_warp.sh hydra_flow 0.2 50.0 fake 0.0 $@
# bash $SCRIPTPATH/_track_warp.sh hydra_flow 0.2 50.0 fake 0.1 $@
# bash $SCRIPTPATH/_track_warp.sh hydra_flow 0.2 50.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh hydra_flow 0.2 50.0 fake 0.3 $@
# bash $SCRIPTPATH/_track_warp.sh hydra_flow 0.2 50.0 fake 0.4 $@

# Wavelet on springs/flow
bash $SCRIPTPATH/_track_warp.sh hydra_flow 0.2 50.0 wavelet 0.2 $@
bash $SCRIPTPATH/_track_warp.sh springs_2d 0.2 50.0 wavelet 0.2 $@


# Noise impact on Hydra Flow (Fake@80%)
# bash $SCRIPTPATH/_track_warp.sh hydra_flow 0.02 5.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh hydra_flow 0.07 5.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh hydra_flow 0.20 5.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh hydra_flow 0.35 5.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh hydra_flow 0.85 5.0 fake 0.2 $@

# bash $SCRIPTPATH/_track_warp.sh hydra_flow 0.02 50.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh hydra_flow 0.07 50.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh hydra_flow 0.20 50.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh hydra_flow 0.35 50.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh hydra_flow 0.85 50.0 fake 0.2 $@

# bash $SCRIPTPATH/_track_warp.sh hydra_flow 0.02 500.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh hydra_flow 0.07 500.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh hydra_flow 0.20 500.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh hydra_flow 0.35 500.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh hydra_flow 0.85 500.0 fake 0.2 $@


# Noise impact on Springs 2D (Fake@80%)
# bash $SCRIPTPATH/_track_warp.sh springs_2d 0.02 5.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh springs_2d 0.07 5.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh springs_2d 0.20 5.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh springs_2d 0.35 5.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh springs_2d 0.85 5.0 fake 0.2 $@

# bash $SCRIPTPATH/_track_warp.sh springs_2d 0.02 50.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh springs_2d 0.07 50.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh springs_2d 0.20 50.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh springs_2d 0.35 50.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh springs_2d 0.85 50.0 fake 0.2 $@

# bash $SCRIPTPATH/_track_warp.sh springs_2d 0.02 500.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh springs_2d 0.07 500.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh springs_2d 0.20 500.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh springs_2d 0.35 500.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh springs_2d 0.85 500.0 fake 0.2 $@

