set -e

# Run for each flow on one springs simulation, one flow simulaiton and hydra real data

$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --flow none
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --flow farneback
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --flow tvl1
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --flow raft --scale 2
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --flow vxm
