# Warp Then Track vs KOFT

set -e

SCRIPTPATH=$(dirname "$0")

# Fixed detla = 50, PSNR = 1.5. Different detections method ?
# bash $SCRIPTPATH/_track_warp.sh flow_20140829_1 1.5 50.0 fake 0.0 $@
# bash $SCRIPTPATH/_track_warp.sh flow_20140829_1 1.5 50.0 fake 0.1 $@
# bash $SCRIPTPATH/_track_warp.sh flow_20140829_1 1.5 50.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh flow_20140829_1 1.5 50.0 fake 0.3 $@
# bash $SCRIPTPATH/_track_warp.sh flow_20140829_1 1.5 50.0 fake 0.4 $@

# Wavelet on springs/flows
bash $SCRIPTPATH/_track_warp.sh flow_20140829_1 1.5 50.0 wavelet 0.2 $@
bash $SCRIPTPATH/_track_warp.sh springs 1.5 50.0 wavelet 0.2 $@


# HOTA vs PSNR on flows (Fake@80%)
# bash $SCRIPTPATH/_track_warp.sh flow_20140829_1 0.1 5.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh flow_20140829_1 0.5 5.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh flow_20140829_1 1.5 5.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh flow_20140829_1 3.0 5.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh flow_20140829_1 10.0 5.0 fake 0.2 $@

bash $SCRIPTPATH/_track_warp.sh flow_20140829_1 0.1 50.0 fake 0.2 $@
bash $SCRIPTPATH/_track_warp.sh flow_20140829_1 0.5 50.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh flow_20140829_1 1.5 50.0 fake 0.2 $@
bash $SCRIPTPATH/_track_warp.sh flow_20140829_1 3.0 50.0 fake 0.2 $@
bash $SCRIPTPATH/_track_warp.sh flow_20140829_1 10.0 50.0 fake 0.2 $@

# bash $SCRIPTPATH/_track_warp.sh flow_20140829_1 0.1 50.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh flow_20140829_1 0.5 50.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh flow_20140829_1 1.5 50.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh flow_20140829_1 3.0 50.0 fake 0.2 $@
# bash $SCRIPTPATH/_track_warp.sh flow_20140829_1 10.0 50.0 fake 0.2 $@


