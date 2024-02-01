set -e

### DEFAULT (PSNR = 1.5db, delta=50, 1000 particles, flow and springs)

# Springs
$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 1.5 --seed $@

# Flow
$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 1.5 --seed $@


### IMAGE DEGRADATION (PSNR from 0.5db to 10db, delta from 5 to 500)

# Springs

$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 0.1 --simulator.imaging_config.delta 5 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 0.1 --simulator.imaging_config.delta 50 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 0.1 --simulator.imaging_config.delta 500 --seed $@

$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 0.5 --simulator.imaging_config.delta 5 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 0.5 --simulator.imaging_config.delta 50 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 0.5 --simulator.imaging_config.delta 500 --seed $@

$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 1.5 --simulator.imaging_config.delta 5 --seed $@
# $RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 1.5 --simulator.imaging_config.delta 50 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 1.5 --simulator.imaging_config.delta 500 --seed $@

$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 3.0 --simulator.imaging_config.delta 5 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 3.0 --simulator.imaging_config.delta 50 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 3.0 --simulator.imaging_config.delta 500 --seed $@

$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 10.0 --simulator.imaging_config.delta 5 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 10.0 --simulator.imaging_config.delta 50 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 10.0 --simulator.imaging_config.delta 500 --seed $@


# FLow

$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 0.1 --simulator.imaging_config.delta 5 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 0.1 --simulator.imaging_config.delta 50 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 0.1 --simulator.imaging_config.delta 500 --seed $@

$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 0.5 --simulator.imaging_config.delta 5 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 0.5 --simulator.imaging_config.delta 50 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 0.5 --simulator.imaging_config.delta 500 --seed $@

$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 1.5 --simulator.imaging_config.delta 5 --seed $@
# $RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 1.5 --simulator.imaging_config.delta 50 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 1.5 --simulator.imaging_config.delta 500 --seed $@

$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 3.0 --simulator.imaging_config.delta 5 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 3.0 --simulator.imaging_config.delta 50 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 3.0 --simulator.imaging_config.delta 500 --seed $@

$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 10.0 --simulator.imaging_config.delta 5 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 10.0 --simulator.imaging_config.delta 50 --seed $@
$RUN_KOFT_ENV expyrun config/simulator/flow_dupre.yml --simulator.particle.n 1000 --simulator.imaging_config.psnr 10.0 --simulator.imaging_config.delta 500 --seed $@
