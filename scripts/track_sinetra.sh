set -e

# Run for springs and flow with Fake detection with F1 from 100% to 60% and Wavelet detection

# Springs
$RUN_KOFT_ENV expyrun config/track/sinetra.yml --simulation_name springs_2d --seed $@ --detection.detector wavelet
$RUN_KOFT_ENV expyrun config/track/sinetra.yml --simulation_name springs_2d --seed $@ --detection.detector fake --detection.fake.fpr 0.0 --detection.fake.fnr 0.0
$RUN_KOFT_ENV expyrun config/track/sinetra.yml --simulation_name springs_2d --seed $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
$RUN_KOFT_ENV expyrun config/track/sinetra.yml --simulation_name springs_2d --seed $@ --detection.detector fake --detection.fake.fpr 0.2 --detection.fake.fnr 0.2
$RUN_KOFT_ENV expyrun config/track/sinetra.yml --simulation_name springs_2d --seed $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
$RUN_KOFT_ENV expyrun config/track/sinetra.yml --simulation_name springs_2d --seed $@ --detection.detector fake --detection.fake.fpr 0.4 --detection.fake.fnr 0.4


# Flow
$RUN_KOFT_ENV expyrun config/track/sinetra.yml --simulation_name hydra_flow --seed $@ --detection.detector wavelet
$RUN_KOFT_ENV expyrun config/track/sinetra.yml --simulation_name hydra_flow --seed $@ --detection.detector fake --detection.fake.fpr 0.0 --detection.fake.fnr 0.0
$RUN_KOFT_ENV expyrun config/track/sinetra.yml --simulation_name hydra_flow --seed $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
$RUN_KOFT_ENV expyrun config/track/sinetra.yml --simulation_name hydra_flow --seed $@ --detection.detector fake --detection.fake.fpr 0.2 --detection.fake.fnr 0.2
$RUN_KOFT_ENV expyrun config/track/sinetra.yml --simulation_name hydra_flow --seed $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
$RUN_KOFT_ENV expyrun config/track/sinetra.yml --simulation_name hydra_flow --seed $@ --detection.detector fake --detection.fake.fpr 0.4 --detection.fake.fnr 0.4