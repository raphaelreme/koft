set -e

# Springs - 1000 particles
$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.snr 1.8 --seed $@

# Flow - 1000 particles
$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 1000 --simulator.imaging_config.snr 1.8 --seed $@
