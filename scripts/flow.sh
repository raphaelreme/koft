set -e

# Check different of for the default video

$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --flow none --run_dupre True --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --flow farneback --run_dupre True --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --flow tvl1 --run_dupre True --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --flow raft --scale 2 --run_dupre True --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --flow vxm --run_dupre True --seed $@


# Run Farneback for all kinds of snr videos

$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --alpha 0.02 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --alpha 0.02 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --alpha 0.02 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --alpha 0.07 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --alpha 0.07 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --alpha 0.07 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --alpha 0.2 --delta 5.0 --seed $@
# $RUN_KOFT_ENV expyrun config/optical_flow/default.yml --alpha 0.2 --delta 50.0 --seed $@  # Already done above
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --alpha 0.2 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --alpha 0.35 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --alpha 0.35 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --alpha 0.35 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --alpha 0.85 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --alpha 0.85 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/default.yml --alpha 0.85 --delta 500.0 --seed $@
