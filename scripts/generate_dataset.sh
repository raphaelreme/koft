set -e

# Springs - 100 particles
$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 100 --simulator.imaging_config.snr 1 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 100 --simulator.imaging_config.snr 2 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 100 --simulator.imaging_config.snr 4 --seed $@


# Springs - 400 particles
$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 400 --simulator.imaging_config.snr 1 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 400 --simulator.imaging_config.snr 2 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 400 --simulator.imaging_config.snr 4 --seed $@


# Springs - 1000 particles
$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.snr 1 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.snr 2 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.snr 4 --seed $@


# Flow - 100 particles
$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 100 --simulator.imaging_config.snr 1 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 100 --simulator.imaging_config.snr 2 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 100 --simulator.imaging_config.snr 4 --seed $@


# Flow - 400 particles
$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 400 --simulator.imaging_config.snr 1 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 400 --simulator.imaging_config.snr 2 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 400 --simulator.imaging_config.snr 4 --seed $@


# Flow - 1000 particles
$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 1000 --simulator.imaging_config.snr 1 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 1000 --simulator.imaging_config.snr 2 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 1000 --simulator.imaging_config.snr 4 --seed $@
