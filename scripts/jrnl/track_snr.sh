set -e

# Run for springs and flow with Fake detection with F1 at 80% and alpha from 0.02 to 0.85 and Delta 5.0 to 500.0

# Springs
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name springs_2d --alpha 0.02 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name springs_2d --alpha 0.02 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name springs_2d --alpha 0.02 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name springs_2d --alpha 0.07 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name springs_2d --alpha 0.07 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name springs_2d --alpha 0.07 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name springs_2d --alpha 0.2 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name springs_2d --alpha 0.2 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name springs_2d --alpha 0.2 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name springs_2d --alpha 0.35 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name springs_2d --alpha 0.35 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name springs_2d --alpha 0.35 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name springs_2d --alpha 0.85 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name springs_2d --alpha 0.85 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name springs_2d --alpha 0.85 --delta 500.0 --seed $@


# Flow
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name hydra_flow --alpha 0.02 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name hydra_flow --alpha 0.02 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name hydra_flow --alpha 0.02 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name hydra_flow --alpha 0.07 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name hydra_flow --alpha 0.07 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name hydra_flow --alpha 0.07 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name hydra_flow --alpha 0.2 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name hydra_flow --alpha 0.2 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name hydra_flow --alpha 0.2 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name hydra_flow --alpha 0.35 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name hydra_flow --alpha 0.35 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name hydra_flow --alpha 0.35 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name hydra_flow --alpha 0.85 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name hydra_flow --alpha 0.85 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name hydra_flow --alpha 0.85 --delta 500.0 --seed $@
