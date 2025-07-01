set -e

# For figures, we run either on springs_2d with seed 0 or hydra_flow without randomizing the sequence and seed 0
# Required for Fig2
bash scripts/tip/generate_dataset.sh 0 --simulator.base_video.randomise False

# Run OF
# Required for fig3.a and 3.b
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --flow none --seed 0
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --flow farneback --seed 0
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --flow tvl1 --seed 0
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --flow raft --scale 2 --seed 0
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --flow vxm --seed 0

# Run Tracking
